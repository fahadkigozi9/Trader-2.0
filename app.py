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
st.set_page_config(page_title="AI Trading Assistant & Multi-Trade Tracker", layout="wide")

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
# Persistence Layer (Stores ALL Trades Permanently)
# ------------------------------------------------------------------------------
def load_trade_history() -> list:
    """Loads trade history from session state or trade_state.json file."""
    if "trade_history" in st.session_state:
        return st.session_state["trade_history"]
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                history = json.load(f)
                if isinstance(history, list):
                    st.session_state["trade_history"] = history
                    return history
        except Exception:
            pass
    st.session_state["trade_history"] = []
    return []

def save_new_signal(signal_data: dict):
    """Appends a new signal, storing ALL trades without a limit."""
    history = load_trade_history()
    history.append(signal_data)
        
    st.session_state["trade_history"] = history
    with open(STATE_FILE, "w") as f:
        json.dump(history, f, indent=4)

def save_all_trades(history: list):
    """Saves updated trade history list directly to file and session."""
    st.session_state["trade_history"] = history
    with open(STATE_FILE, "w") as f:
        json.dump(history, f, indent=4)

def clear_all_trade_records():
    """Wipes stored trade history."""
    st.session_state["trade_history"] = []
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

# ------------------------------------------------------------------------------
# Data Fetching & Telegram Dispatcher
# ------------------------------------------------------------------------------
def fetch_live_price(ticker_symbol: str) -> dict:
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            return {"price": 0.0, "change": 0.0, "status": "No Data"}
        
        latest_price = float(data["Close"].iloc[-1])
        open_price = float(data["Open"].iloc[0])
        pct_change = ((latest_price - open_price) / open_price) * 100
        
        return {"price": latest_price, "change": pct_change, "status": "Success"}
    except Exception as e:
        return {"price": 0.0, "change": 0.0, "status": str(e)}

def scrape_forex_factory_news() -> list:
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
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=5)
        return res.status_code == 200
    except Exception:
        return False

# ------------------------------------------------------------------------------
# Order Type Resolver & Groq Signal Generator
# ------------------------------------------------------------------------------
def determine_order_type(action: str, entry_price: float, current_price: float) -> str:
    """Determines exact execution order type based on current market price."""
    if action == "BUY":
        return "BUY STOP" if entry_price > current_price else "BUY LIMIT"
    else:  # SELL
        return "SELL STOP" if entry_price < current_price else "SELL LIMIT"

MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b"
]

def generate_groq_signal(asset: str, price_data: dict, news_data: list, timeframe: str, api_key: str) -> dict:
    if not api_key:
        return {"signal": "ERROR", "confidence": 0, "reasons": ["Groq API key missing"]}

    client = Groq(api_key=api_key)
    current_price = price_data.get("price", 0.0)

    prompt = f"""
You are an institutional trading analyst. Analyze parameters for {asset}:
- Timeframe: {timeframe}
- Current Price: {current_price}
- Daily Change (%): {price_data.get('change')}
- High-Impact News: {json.dumps(news_data, indent=2)}

Return ONLY a raw JSON object matching this schema without markdown formatting:
{{
    "signal": "BUY",
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
            
            if raw_content.startswith("```json"): raw_content = raw_content[7:]
            if raw_content.startswith("```"): raw_content = raw_content[3:]
            if raw_content.endswith("```"): raw_content = raw_content[:-3]
                
            signal_obj = json.loads(raw_content.strip())
            
            sig_action = signal_obj.get("signal", "BUY").upper()
            entry_p = float(signal_obj.get("entry_price", current_price))
            signal_obj["order_type"] = determine_order_type(sig_action, entry_p, current_price)
            
            return signal_obj
        except APIError as e:
            if e.status_code in [400, 404] or "decommissioned" in str(e).lower():
                continue
            return {"signal": "ERROR", "confidence": 0, "reasons": [str(e)]}
        except Exception:
            continue

    return {"signal": "ERROR", "confidence": 0, "reasons": ["All AI model fallbacks failed."]}

# ------------------------------------------------------------------------------
# Post-Mortem Failure Analysis Engine
# ------------------------------------------------------------------------------
def analyze_failed_trade(trade: dict, live_price: float, api_key: str) -> str:
    """Uses Groq AI to evaluate why a trade hit its Stop Loss."""
    if not api_key:
        return "Groq API key is required to perform failure analysis."

    client = Groq(api_key=api_key)
    prompt = f"""
You are an expert forex risk manager. Perform a post-mortem failure analysis on a trade setup that hit its STOP LOSS:
- Asset: {trade.get('asset')}
- Order Type: {trade.get('order_type')}
- Entry Price: {trade.get('entry_price')}
- Stop Loss: {trade.get('stop_loss')}
- Take Profit: {trade.get('take_profit_1')}
- Current/Exit Price: {live_price}
- Original Confidence: {trade.get('confidence')}%
- Original Setup Reasons: {json.dumps(trade.get('reasons', []))}

Provide a concise 3-bullet point breakdown explaining:
1. What went wrong (e.g. liquidity sweep, news reaction, tight SL).
2. Market structure/bias error.
3. Key takeaway/lesson to avoid similar stop loss hits in future setups.
"""

    for model in MODEL_FALLBACKS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            continue

    return "Unable to generate post-mortem analysis at this time."

# ------------------------------------------------------------------------------
# Multi-Trade Status Evaluator
# ------------------------------------------------------------------------------
def evaluate_trade_status(saved_signal: dict, current_price: float) -> dict:
    action = saved_signal.get("signal", "BUY").upper()
    entry = float(saved_signal.get("entry_price", 0.0))
    sl = float(saved_signal.get("stop_loss", 0.0))
    tp1 = float(saved_signal.get("take_profit_1", 0.0))

    if entry == 0.0 or sl == 0.0:
        return {"recommendation": "UNKNOWN", "analysis": "Invalid levels."}

    is_gold = "GC=F" in saved_signal.get("ticker", "") or "Gold" in saved_signal.get("asset", "")
    point = 0.1 if is_gold else 0.0001

    if "BUY" in action:
        pnl_pips = (current_price - entry) / point
        risk_pips = (entry - sl) / point
        hit_sl = current_price <= sl
    else:
        pnl_pips = (entry - current_price) / point
        risk_pips = (sl - entry) / point
        hit_sl = current_price >= sl

    rr_achieved = pnl_pips / risk_pips if risk_pips > 0 else 0

    if hit_sl:
        recommendation = "STOP LOSS HIT"
        analysis = f"❌ Trade invalidated. Price hit SL ({sl}). PnL: {pnl_pips:.1f} pips."
    elif rr_achieved >= 1.0:
        recommendation = "TRAIL SL TO BREAKEVEN"
        analysis = f"🎯 1:1 R:R achieved (+{pnl_pips:.1f} pips). Lock SL to entry ({entry})."
    elif pnl_pips < -(risk_pips * 0.75):
        recommendation = "EARLY EXIT / CLOSE"
        analysis = f"⚠️ Severe drawdown (-{abs(pnl_pips):.1f} pips). Structure invalidated."
    else:
        recommendation = "HOLD POSITION"
        analysis = f"⏳ Setup intact. PnL: {pnl_pips:+.1f} pips. TP: {tp1}."

    return {
        "id": saved_signal.get("id"),
        "timestamp": saved_signal.get("timestamp"),
        "asset": saved_signal.get("asset"),
        "order_type": saved_signal.get("order_type"),
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "current_price": current_price,
        "pnl_pips": round(pnl_pips, 1),
        "rr_achieved": round(rr_achieved, 2),
        "recommendation": recommendation,
        "analysis": analysis,
        "hit_sl": hit_sl
    }

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

tab1, tab2, tab3 = st.tabs(["⚡ Generate Signal", "🔍 Check Status", "🩺 Post-Mortem Analysis"])

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
                        setup["id"] = f"TRADE-{pd.Timestamp.now().strftime('%M%S')}"
                        setup["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                        setup["asset"] = selected_asset_label
                        setup["ticker"] = selected_ticker
                        setup["timeframe"] = selected_timeframe
                        setup["status"] = "ACTIVE"
                        
                        save_new_signal(setup)

                        st.success(f"### {setup.get('order_type')} ({sig}) — {setup.get('confidence')}% Confidence")
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Entry Price", setup.get("entry_price"))
                        m2.metric("Stop Loss", setup.get("stop_loss"))
                        m3.metric("Take Profit 1", setup.get("take_profit_1"))
                        m4.metric("Take Profit 2", setup.get("take_profit_2"))

                        st.markdown("**Rationale:**")
                        for r in setup.get("reasons", []):
                            st.write(f"- {r}")

                        if telegram_token and telegram_chat_id:
                            msg = (
                                f"📲 *NEW AI SIGNAL DISPATCH*\n\n"
                                f"📌 *Asset:* {selected_asset_label} ({selected_timeframe})\n"
                                f"⚡ *Order Type:* `{setup.get('order_type')}`\n"
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
# TAB 2: CHECK STATUS (ALL TRADES)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🔍 Status Check — All Stored Trades")
    trade_history = load_trade_history()

    if not trade_history:
        st.info("No saved trades found in memory. Generate a signal in Tab 1 first.")
    else:
        st.write(f"Showing **{len(trade_history)} total trade(s)** stored in `trade_state.json`:")

        if st.button("🔄 Check Status For All Trades", type="primary"):
            reports = []
            
            for trade in reversed(trade_history):
                ticker = trade.get("ticker", selected_ticker)
                live_data = fetch_live_price(ticker)
                
                if live_data["status"] == "Success":
                    rep = evaluate_trade_status(trade, live_data["price"])
                    reports.append(rep)
                    
                    if rep["hit_sl"] and trade.get("status") != "STOP LOSS HIT":
                        trade["status"] = "STOP LOSS HIT"
                        save_all_trades(trade_history)

                    with st.expander(f"📌 {rep['asset']} | {rep['order_type']} @ {rep['entry']} ({trade.get('timestamp')})", expanded=True):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Live Price", f"{rep['current_price']:.4f}")
                        c2.metric("Floating PnL", f"{rep['pnl_pips']:+} pips")
                        c3.metric("R:R Reached", f"1:{rep['rr_achieved']}")
                        c4.write(f"**Action:** `{rep['recommendation']}`")
                        
                        st.write(f"**Analysis:** {rep['analysis']}")
            
            if reports and telegram_token and telegram_chat_id:
                summary_lines = ["📊 *MULTI-TRADE STATUS REPORT*\n"]
                for r in reports:
                    summary_lines.append(
                        f"• *{r['asset']}* ({r['order_type']})\n"
                        f"  PnL: `{r['pnl_pips']:+} pips` | Action: *{r['recommendation']}*\n"
                    )
                send_telegram_alert(telegram_token, telegram_chat_id, "\n".join(summary_lines))
                st.caption("Combined report pushed to Telegram!")

        st.markdown("---")
        if st.button("🗑️ Clear All Stored Trades"):
            clear_all_trade_records()
            st.success("Trade state file cleared!")
            st.rerun()

# ------------------------------------------------------------------------------
# TAB 3: POST-MORTEM ANALYSIS (FAILED TRADES)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🩺 AI Post-Mortem Analysis on Failed Trades")
    trade_history = load_trade_history()
    
    sl_trades = [t for t in trade_history if t.get("status") == "STOP LOSS HIT"]

    if not sl_trades:
        st.info("No trades marked as 'STOP LOSS HIT' yet. Execute 'Check Status' in Tab 2 to detect trades that hit SL.")
    else:
        st.write(f"Found **{len(sl_trades)} trade(s)** that hit Stop Loss:")

        for trade in reversed(sl_trades):
            with st.expander(f"❌ {trade.get('asset')} ({trade.get('order_type')}) — Entry: {trade.get('entry_price')} | SL: {trade.get('stop_loss')}", expanded=True):
                st.write(f"**Timestamp:** {trade.get('timestamp')}")
                st.write(f"**Original Reasons:** {', '.join(trade.get('reasons', []))}")

                if "post_mortem" in trade:
                    st.markdown("### 📋 AI Failure Analysis:")
                    st.write(trade["post_mortem"])
                else:
                    if st.button(f"Analyze Why This Trade Failed ({trade.get('id')})", key=trade.get('id')):
                        live_data = fetch_live_price(trade.get("ticker", selected_ticker))
                        with st.spinner("AI analyzing setup structure and market failure..."):
                            analysis_result = analyze_failed_trade(
                                trade, live_data.get("price", 0.0), groq_key
                            )
                            trade["post_mortem"] = analysis_result
                            save_all_trades(trade_history)
                            st.rerun()
