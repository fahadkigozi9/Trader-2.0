import datetime
import json
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import yfinance as yf
from groq import Groq, APIError

# Configuration & Setup
st.set_page_config(page_title="AI Trading Assistant", layout="wide")

ASSET_MAP = {
    "GBP/USD": "GBPUSD=X",
    "Gold (XAU/USD)": "GC=F",
    "Bitcoin (BTC/USD)": "BTC-USD"
}

# ------------------------------------------------------------------------------
# Persistent State Initialization
# ------------------------------------------------------------------------------
if "groq_key" not in st.session_state:
    st.session_state["groq_key"] = st.secrets.get("GROQ_API_KEY", "")
if "telegram_token" not in st.session_state:
    st.session_state["telegram_token"] = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
if "telegram_chat_id" not in st.session_state:
    st.session_state["telegram_chat_id"] = st.secrets.get("TELEGRAM_CHAT_ID", "")

# ------------------------------------------------------------------------------
# Data Fetching Functions
# ------------------------------------------------------------------------------
def fetch_live_price(ticker_symbol: str) -> dict:
    """Fetch live price and daily change using yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            return {"price": "None", "change": "None", "status": "No Data"}
        
        latest_price = float(data["Close"].iloc[-1])
        open_price = float(data["Open"].iloc[0])
        pct_change = ((latest_price - open_price) / open_price) * 100
        
        return {
            "price": latest_price,
            "change": pct_change,
            "status": "Success"
        }
    except Exception as e:
        return {"price": "None", "change": "None", "status": str(e)}


def scrape_forex_factory_news() -> list:
    """Scrape high-impact economic news events from Forex Factory."""
    url = "https://www.forexfactory.com/calendar"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
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
                    time = row.find("td", class_="calendar__time")
                    
                    currency_text = currency.text.strip() if currency else ""
                    event_title = title.text.strip() if title else ""
                    time_val = time.text.strip() if time else ""
                    
                    events.append({
                        "time": time_val,
                        "currency": currency_text,
                        "title": event_title
                    })
    except Exception:
        pass
    return events


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    """Send an automated alert message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=5)
        return res.status_code == 200
    except Exception:
        return False


# ------------------------------------------------------------------------------
# Groq AI Inference Function
# ------------------------------------------------------------------------------
MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b"
]


def generate_groq_signal(asset: str, price_data: dict, news_data: list, timeframe: str, api_key: str) -> dict:
    """Generate precise order parameters and signal reasoning using Groq API."""
    if not api_key:
        return {"signal": "ERROR", "confidence": 0, "reasons": ["Groq API key missing"]}

    client = Groq(api_key=api_key)

    news_str = json.dumps(news_data, indent=2)
    current_price = str(price_data.get("price"))
    daily_change = str(price_data.get("change"))

    prompt_template = """
You are an expert institutional trading analyst. Analyze the market structure and news for __ASSET__:
- Timeframe: __TIMEFRAME__
- Current Market Price: __PRICE__
- Daily Change (%): __CHANGE__
- Today's High-Impact News: __NEWS__

Evaluate technical key levels (support/resistance, liquidity sweeps) and macroeconomic risk to calculate trade setup parameters.

Return ONLY a valid, raw JSON object matching this EXACT schema without markdown commentary:
{
    "signal": "BUY",
    "order_type": "BUY LIMIT",
    "entry_price": 0.0,
    "stop_loss": 0.0,
    "take_profit_1": 0.0,
    "take_profit_2": 0.0,
    "risk_reward_ratio": "1:2.5",
    "confidence": 85,
    "reasons": [
        "Reason 1: Technical confluence or key level sweep",
        "Reason 2: Fundamental/news catalyst analysis"
    ]
}
"""

    prompt = (
        prompt_template.replace("__ASSET__", asset)
        .replace("__TIMEFRAME__", timeframe)
        .replace("__PRICE__", current_price)
        .replace("__CHANGE__", daily_change)
        .replace("__NEWS__", news_str)
    )

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
            err_msg = str(e).lower()
            if e.status_code in [400, 404] or "model" in err_msg or "decommissioned" in err_msg:
                st.warning(f"Model `{model}` unavailable. Trying fallback...")
                continue
            else:
                st.error(f"Groq API Error on {model}: {e.message}")
                return {"signal": "ERROR", "confidence": 0, "reasons": [str(e)]}
        except Exception as e:
            st.warning(f"Unexpected error on model `{model}`: {str(e)}. Trying fallback...")
            continue

    st.error("All model fallbacks failed. Verify your Groq API key.")
    return {"signal": "ERROR", "confidence": 0, "reasons": ["All model fallbacks failed"]}


# ------------------------------------------------------------------------------
# Streamlit User Interface
# ------------------------------------------------------------------------------
st.title("📈 AI Trading Setup & Signal Dashboard")

# Sidebar - Settings & Credentials with persistent Session Keys
st.sidebar.header("🔑 Credentials & Setup")

groq_key = st.sidebar.text_input(
    "Groq API Key",
    value=st.session_state["groq_key"],
    type="password",
    key="groq_key_input"
)
st.session_state["groq_key"] = groq_key

telegram_token = st.sidebar.text_input(
    "Telegram Bot Token",
    value=st.session_state["telegram_token"],
    type="password",
    key="telegram_token_input"
)
st.session_state["telegram_token"] = telegram_token

telegram_chat_id = st.sidebar.text_input(
    "Telegram Chat ID",
    value=st.session_state["telegram_chat_id"],
    key="telegram_chat_id_input"
)
st.session_state["telegram_chat_id"] = telegram_chat_id

st.sidebar.markdown("---")
st.sidebar.header("📌 Asset Selection")
selected_asset_label = st.sidebar.selectbox("Select Asset", list(ASSET_MAP.keys()))
selected_ticker = ASSET_MAP[selected_asset_label]
selected_timeframe = st.sidebar.selectbox("Select Timeframe", ["15m", "1h", "4h", "1D"])

# Fetch News Side Panel
st.sidebar.markdown("---")
st.sidebar.header("📅 High-Impact News Today")
news_list = scrape_forex_factory_news()
if news_list:
    for item in news_list:
        st.sidebar.caption(f"⏱️ **{item['time']}** [{item['currency']}] - {item['title']}")
else:
    st.sidebar.caption("No high-impact news scraped or available.")

# Main Dashboard Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Live Market Data: {selected_asset_label}")
    price_data = fetch_live_price(selected_ticker)
    
    if price_data["status"] == "Success":
        st.metric(
            label="Current Price",
            value=f"{price_data['price']:.4f}",
            delta=f"{price_data['change']:.2f}%"
        )
    else:
        st.error(f"Error fetching price data: {price_data['status']}")

with col2:
    st.subheader("Signal & Parameter Generator")
    if st.button("Generate Trade Parameters & Send Alert"):
        if not groq_key:
            st.error("Please enter your Groq API Key in the sidebar or app secrets.")
        else:
            with st.spinner("Calculating entry levels, stop loss, and targets..."):
                setup = generate_groq_signal(
                    selected_asset_label, price_data, news_list, selected_timeframe, groq_key
                )
                
                sig = setup.get("signal", "NEUTRAL")
                conf = setup.get("confidence", 0)
                order_type = setup.get("order_type", "MARKET")
                entry = setup.get("entry_price", 0.0)
                sl = setup.get("stop_loss", 0.0)
                tp1 = setup.get("take_profit_1", 0.0)
                tp2 = setup.get("take_profit_2", 0.0)
                rr = setup.get("risk_reward_ratio", "N/A")

                if sig == "BUY":
                    st.success(f"### {order_type} ({sig}) — {conf}% Confidence")
                elif sig == "SELL":
                    st.error(f"### {order_type} ({sig}) — {conf}% Confidence")
                else:
                    st.warning(f"### Signal: {sig} — {conf}% Confidence")

                # Metrics Grid for Parameters
                if sig in ["BUY", "SELL"]:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Entry Price", f"{entry}")
                    m2.metric("Stop Loss (SL)", f"{sl}")
                    m3.metric("Take Profit 1", f"{tp1}")
                    m4.metric("Take Profit 2", f"{tp2}")
                    st.caption(f"**Risk:Reward Ratio:** `{rr}`")

                st.markdown("**Trade Strategy Rationale:**")
                for reason in setup.get("reasons", []):
                    st.write(f"- {reason}")

                # Dispatch formatted message to Telegram
                if telegram_token and telegram_chat_id:
                    msg = (
                        f"🚨 *NEW TRADING SETUP*\n\n"
                        f"📌 *Asset:* {selected_asset_label} ({selected_timeframe})\n"
                        f"⚡ *Type:* `{order_type}`\n"
                        f"📊 *Confidence:* {conf}%\n\n"
                        f"🎯 *Entry:* `{entry}`\n"
                        f"🛑 *Stop Loss:* `{sl}`\n"
                        f"🟢 *Take Profit 1:* `{tp1}`\n"
                        f"🚀 *Take Profit 2:* `{tp2}`\n"
                        f"⚖️ *R:R Ratio:* {rr}\n\n"
                        f"💡 *Rationale:*\n" + "\n".join([f"• {r}" for r in setup.get("reasons", [])])
                    )
                    sent = send_telegram_message(telegram_token, telegram_chat_id, msg)
                    if sent:
                        st.success("Trade setup sent to Telegram!")
                    else:
                        st.error("Failed to deliver Telegram alert. Verify Bot Token & Chat ID.")
                else:
                    st.info("Telegram credentials not provided. Displaying parameters on dashboard only.")
