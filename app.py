import json
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import yfinance as yf
from groq import Groq, APIError

# ------------------------------------------------------------------------------
# Configuration & Session State
# ------------------------------------------------------------------------------
st.set_page_config(page_title="AI Trading Assistant & Position Tracker", layout="wide")

STATE_FILE = "trade_state.json"

ASSET_MAP = {
    "Gold (XAU/USD)": "GC=F",
    "GBP/USD": "GBPUSD=X",
    "EUR/USD": "EURUSD=X",
    "Bitcoin (BTC/USD)": "BTC-USD"
}

if "groq_key" not in st.session_state:
    st.session_state["groq_key"] = st.secrets.get("GROQ_API_KEY", "")
if "telegram_token" not in st.session_state:
    st.session_state["telegram_token"] = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
if "telegram_chat_id" not in st.session_state:
    st.session_state["telegram_chat_id"] = st.secrets.get("TELEGRAM_CHAT_ID", "")

# ------------------------------------------------------------------------------
# Persistence Layer (Remembers Signals)
# ------------------------------------------------------------------------------
def save_active_signal(signal_data: dict):
    """Saves the latest generated signal to Streamlit state and trade_state.json."""
    st.session_state["active_signal"] = signal_data
    with open(STATE_FILE, "w") as f:
        json.dump(signal_data, f, indent=4)

def load_active_signal() -> dict:
    """Retrieves active signal from session state or trade_state.json file."""
    if "active_signal" in st.session_state:
        return st.session_state["active_signal"]
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                st.session_state["active_signal"] = data
                return data
        except Exception:
            return None
    return None

def clear_active_signal():
    """Removes stored signal state when a trade is closed."""
    st.session_state.pop("active_signal", None)
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

# ------------------------------------------------------------------------------
# Data Fetching & Telegram Notifications
# ------------------------------------------------------------------------------
def fetch_live_price(ticker_symbol: str) -> dict:
    """Fetch latest price and daily change from yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            return {"price": 0.0, "change": 0.0, "status": "No Data"}
        
        latest_price = float(data["Close"].iloc[-1])
        open_price = float(data["Open"].iloc[0])
        pct_change = ((latest_price - open_price) / open_price) * 100
        
        return {
            "price": latest_price,
            "change": pct_change,
            "status": "Success"
        }
    except Exception as e:
        return {"price": 0.0, "change": 0.0, "status": str(e)}

def scrape_forex_factory_news() -> list:
    """Scrapes high-impact economic calendar events."""
    url = "https://www.forexfactory.com/calendar"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    events = []
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.find_all("tr", class_="calendar__row")
            for row in rows:
                impact = row.find("td", class_="calendar__impact")
                if impact and impact.find("span", class_="icon--ff-impact-red"):
                    currency = row.find("td", class_="calendar__currency")
                    title = row.find("td", class_="calendar__event")
                    time_elem = row.find("td", class_="calendar__time")
                    
                    events.append({
                        "time": time_elem.text.strip() if time_elem else "",
                        "currency": currency.text.strip() if currency else "",
                        "title": title.text.strip() if title else ""
                    })
    except Exception:
        pass
    return events

def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> bool:
    """Sends Markdown formatted alert to your phone via Telegram."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=5)
        return res.status_code == 200
    except Exception:
        return False

# ------------------------------------------------------------------------------
# Groq Signal Generator
# ------------------------------------------------------------------------------
MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b"
]

def generate_groq_signal(asset: str, price_data: dict, news_data: list, timeframe: str, api_key: str) -> dict:
    if not api_key:
        return {"signal": "ERROR", "confidence": 0, "reasons": ["Groq API key missing"]}

    client = Groq(api_key=api_key)
    news_str = json.dumps(news_data, indent=2)
    current_price = str(price_data.get("price"))
    daily_change = str(price_data.get("change"))

    prompt = f"""
You are an institutional trading analyst. Analyze parameters for {asset}:
- Timeframe: {timeframe}
- Current Price: {current_price}
- Daily Change (%): {daily_change}
- High-Impact News Today: {news_str}

Return ONLY a raw JSON object matching this schema without markdown fences or text:
{{
    "signal": "BUY",
    "order_type": "BUY LIMIT",
    "entry_price": 0.0,
    "stop_loss": 0.0,
    "take_profit_1": 0.0,
    "take_profit_2": 0.0,
    "risk_reward_ratio": "1:2.5",
    "confidence": 85,
    "reasons": ["Key structure level sweep", "News sentiment aligned"]
}}
"""

    for model in MODEL_FALLBACKS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw_content = response.choices[0].message.content.strip()
            
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
                
            return json.loads(raw_content.strip())
        except APIError as e:
            if e.status_code in [400, 404] or "decommissioned" in str(e).lower():
                continue
            return {"signal": "ERROR", "confidence": 0, "reasons": [str(e)]}
        except Exception:
            continue

    return {"signal": "ERROR", "confidence": 0, "reasons": ["All AI model fallbacks failed."]}

# ------------------------------------------------------------------------------
# Check Status Logic
# ------------------------------------------------------------------------------
def evaluate_trade_status(saved_signal: dict, current_price: float) -> tuple:
    """Compares saved signal with live market price and produces trade advice."""
    action = saved_signal.get("signal", "BUY").upper()
    entry = float(saved_signal.get("entry_price", 0.0))
    sl = float(saved_signal.get("stop_loss", 0.0))
    tp1 = float(saved_signal.get("take_profit_1", 0.0))

    if entry == 0.0 or sl == 0.0:
        return "UNKNOWN", "Invalid entry or stop loss level in recorded trade."

    # Pip/Point determination
    is_gold = "GC=F" in saved_signal.get("ticker", "") or "Gold" in saved_signal.get("asset", "")
    point = 0.1 if is_gold else 0.0001

    if "BUY" in action:
        pnl_pips = (current_price - entry) / point
        risk_pips = (entry - sl) / point
    else:
        pnl_pips = (entry - current_price) / point
        risk_pips = (sl - entry) / point

    rr_achieved = pnl_pips / risk_pips if risk_pips > 0 else 0

    if rr_achieved >= 1.0:
        recommendation = "TRAIL SL TO BREAKEVEN"
        analysis = f"🎯 1:1 Risk-to-Reward reached (+{pnl_pips:.1f} pips). Move Stop Loss to Entry ({entry}) to lock in a risk-free trade."
    elif pnl_pips < -(risk_pips * 0.75):
        recommendation = "EARLY EXIT / CLOSE"
        analysis = f"⚠️ Trade is in severe drawdown (-{abs(pnl_pips):.1f} pips), approaching SL ({sl}). Market structure has invalidated."
    else:
        recommendation = "HOLD POSITION"
        analysis = f"⏳ Setup structure is healthy. Current PnL: {pnl_pips:+.1f} pips. Target TP: {tp1}."

    report = {
        "action": action,
        "asset": saved_signal.get("asset", "Asset"),
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "current_price": current_price,
        "pnl_pips": round(pnl_pips, 1),
        "rr_achieved": round(rr_achieved, 2),
        "recommendation": recommendation,
        "analysis": analysis
    }

    return recommendation, report

# ------------------------------------------------------------------------------
# Dashboard UI Layout
# ------------------------------------------------------------------------------
st.title("📊 AI Trading Assistant & Position Tracker")

# Sidebar
st.sidebar.header("🔑 Credentials")
groq_key = st.sidebar.text_input("Groq API Key", value=st.session_state["groq_key"], type="password")
st.session_state["groq_key"] = groq_key

telegram_token = st.sidebar.text_input("Telegram Bot Token", value=st.session_state["telegram_token"], type="password")
st.session_state["telegram_token"] = telegram_token

telegram_chat_id = st.sidebar.text_input("Telegram Chat ID", value=st.session_state["telegram_chat_id"])
st.session_state["telegram_chat_id"] = telegram_chat_id

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuration")
selected_asset_label = st.sidebar.selectbox("Select Asset", list(ASSET_MAP.keys()))
selected_ticker = ASSET_MAP[selected_asset_label]
selected_timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1D"])

news_list = scrape_forex_factory_news()

# Main App Tabs
tab1, tab2 = st.tabs(["⚡ Generate Signal", "🔍 Check Status"])

# ------------------------------------------------------------------------------
# TAB 1: SIGNAL GENERATION
# ------------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Market Data: {selected_asset_label}")
        price_data = fetch_live_price(selected_ticker)
        if price_data["status"] == "Success":
            st.metric("Live Price", f"{price_data['price']:.4f}", f"{price_data['change']:.2f}%")
        else:
            st.error(f"Price Error: {price_data['status']}")

    with col2:
        st.subheader("Generate & Dispatch Signal")
        if st.button("Generate Signal & Push Alert", type="primary"):
            if not groq_key:
                st.error("Missing Groq API Key!")
            else:
                with st.spinner("Analyzing market structure..."):
                    setup = generate_groq_signal(
                        selected_asset_label, price_data, news_list, selected_timeframe, groq_key
                    )
                    
                    sig = setup.get("signal", "NEUTRAL")
                    if sig in ["BUY", "SELL"]:
                        setup["asset"] = selected_asset_label
                        setup["ticker"] = selected_ticker
                        setup["timeframe"] = selected_timeframe
                        
                        # Save signal to memory and disk
                        save_active_signal(setup)

                        st.success(f"### {setup.get('order_type')} ({sig}) — {setup.get('confidence')}% Confidence")
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Entry", setup.get("entry_price"))
                        m2.metric("Stop Loss", setup.get("stop_loss"))
                        m3.metric("Take Profit 1", setup.get("take_profit_1"))
                        m4.metric("Take Profit 2", setup.get("take_profit_2"))

                        st.markdown("**Rationale:**")
                        for r in setup.get("reasons", []):
                            st.write(f"- {r}")

                        # Push alert to Telegram
                        if telegram_token and telegram_chat_id:
                            msg = (
                                f"📲 *NEW AI SIGNAL DISPATCH*\n\n"
                                f"📌 *Asset:* {selected_asset_label} ({selected_timeframe})\n"
                                f"⚡ *Action:* `{setup.get('order_type')}`\n"
                                f"📊 *Confidence:* {setup.get('confidence')}%\n\n"
                                f"🎯 *Entry:* `{setup.get('entry_price')}`\n"
                                f"🛑 *Stop Loss:* `{setup.get('stop_loss')}`\n"
                                f"🟢 *TP 1:* `{setup.get('take_profit_1')}`\n"
                                f"🚀 *TP 2:* `{setup.get('take_profit_2')}`\n"
                                f"⚖️ *R:R:* {setup.get('risk_reward_ratio')}\n\n"
                                f"💡 *Rationale:*\n" + "\n".join([f"• {reason}" for reason in setup.get("reasons", [])])
                            )
                            if send_telegram_alert(telegram_token, telegram_chat_id, msg):
                                st.success("Signal alert pushed to Telegram!")
                            else:
                                st.error("Failed to deliver Telegram notification.")
                    else:
                        st.warning(f"No clear setup found: {setup.get('reasons')}")

# ------------------------------------------------------------------------------
# TAB 2: CHECK TRADE STATUS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🔍 Active Position Status Check")
    saved_trade = load_active_signal()

    if not saved_trade:
        st.info("No recorded signal currently active. Generate a signal in Tab 1 first.")
    else:
        st.markdown(
            f"**Active Signal Memory:** `{saved_trade.get('signal')}` on `{saved_trade.get('asset')}` "
            f"| Entry: `{saved_trade.get('entry_price')}` | SL: `{saved_trade.get('stop_loss')}` | TP1: `{saved_trade.get('take_profit_1')}`"
        )

        if st.button("Check Trade Status Now", type="primary"):
            ticker_to_check = saved_trade.get("ticker", selected_ticker)
            live_data = fetch_live_price(ticker_to_check)

            if live_data["status"] != "Success":
                st.error("Could not pull live price to evaluate status.")
            else:
                curr_price = live_data["price"]
                rec, report = evaluate_trade_status(saved_trade, curr_price)

                # Visual recommendation banner
                if "TRAIL" in rec:
                    st.success(f"### AI Recommendation: {rec}")
                elif "EXIT" in rec or "CLOSE" in rec:
                    st.error(f"### AI Recommendation: {rec}")
                else:
                    st.info(f"### AI Recommendation: {rec}")

                # Metric cards
                c1, c2, c3 = st.columns(3)
                c1.metric("Live Market Price", f"{report['current_price']:.4f}")
                c2.metric("Floating PnL (Pips)", f"{report['pnl_pips']:+} pips")
                c3.metric("R:R Ratio Reached", f"1:{report['rr_achieved']}")

                st.write(f"**Detailed Analysis:** {report['analysis']}")

                # Dispatch recommendation update to Telegram
                if telegram_token and telegram_chat_id:
                    tg_update = (
                        f"📊 *POSITION STATUS UPDATE*\n\n"
                        f"📌 *Asset:* {report['asset']}\n"
                        f"⚡ *Signal:* {report['action']} @ {report['entry']}\n"
                        f"📈 *Current Price:* {report['current_price']:.4f}\n"
                        f"💰 *PnL:* {report['pnl_pips']:+} pips\n\n"
                        f"💡 *AI Action:* *{report['recommendation']}*\n"
                        f"📝 {report['analysis']}"
                    )
                    send_telegram_alert(telegram_token, telegram_chat_id, tg_update)
                    st.caption("Status update pushed to your Telegram.")

        st.markdown("---")
        if st.button("Clear / Close Saved Trade Record"):
            clear_active_signal()
            st.success("Trade state cleared from memory.")
            st.rerun()
