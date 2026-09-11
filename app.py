import datetime
import json
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import yfinance as yf
from groq import Groq

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

ASSET_MAP = {
    "GBP/USD": "GBPUSD=X",
    "Gold (XAU/USD)": "GC=F",
    "Bitcoin (BTC/USD)": "BTC-USD"
}

# ------------------------------------------------------------------------------
# Data Fetching Functions
# ------------------------------------------------------------------------------
def fetch_live_price(ticker_symbol: str) -> dict:
    """Fetch live price and daily change using yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            return {"price": None, "change": None, "status": "No data found"}
        
        latest_price = float(data["Close"].iloc[-1])
        open_price = float(data["Open"].iloc[0])
        pct_change = ((latest_price - open_price) / open_price) * 100
        
        return {
            "price": latest_price,
            "change": pct_change,
            "status": "Success"
        }
    except Exception as e:
        return {"price": None, "change": None, "status": str(e)}

def scrape_forex_factory_news() -> list:
    """Scrape high-impact economic news events from Forex Factory."""
    url = "https://www.forexfactory.com/calendar"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    events = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.find_all("tr", class_="calendar__row")
            for row in rows:
                impact = row.find("td", class_="calendar__impact")
                if impact and impact.find("span", class_="high"):
                    currency = row.find("td", class_="calendar__currency")
                    currency_text = currency.text.strip() if currency else "USD"
                    
                    event_title = row.find("td", class_="calendar__event")
                    title_text = event_title.text.strip() if event_title else "High Impact Event"
                    
                    time_val = row.find("td", class_="calendar__time")
                    time_text = time_val.text.strip() if time_val else "TBA"
                    
                    events.append({
                        "time": time_text,
                        "currency": currency_text,
                        "title": title_text
                    })
    except Exception:
        pass
    return events

def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    """Send an automated alert message via Telegram."""
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
def generate_groq_signal(asset: str, price_data: dict, news_data: list, api_key: str) -> dict:
    """Generate trade signal and reasoning using Groq API."""
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
You are an expert AI trading strategist. Analyze the current market context for {asset}.

Market Data:
- Current Price: {price_data.get('price')}
- Daily Change (%): {price_data.get('change'):.2f}%

High-Impact Economic News:
{json.dumps(news_data, indent=2)}

Provide your response in strictly VALID JSON format matching this schema:
{{
    "signal": "BUY" | "SELL" | "NEUTRAL",
    "confidence": number between 0 and 100,
    "reasons": ["reason 1", "reason 2", "reason 3"]
}}
Return ONLY raw JSON, with no markdown formatting or extra text.
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        raw_content = response.choices[0].message.content.strip()
        parsed_json = json.loads(raw_content)
        return parsed_json
    except Exception as e:
        return {
            "signal": "ERROR",
            "confidence": 0,
            "reasons": [f"AI Processing Error: {str(e)}"]
        }

# ------------------------------------------------------------------------------
# Streamlit User Interface
# ------------------------------------------------------------------------------
st.title("📈 AI Trading Assistant Dashboard")

# Sidebar - Settings & Credentials
st.sidebar.header("🔑 Credentials & Setup")
groq_key = st.sidebar.text_input("Groq API Key", type="password")
telegram_token = st.sidebar.text_input("Telegram Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID")

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

# Main Dashboard Grid
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
    st.subheader("Signal Generation")
    if st.button("Generate Signal & Dispatch Alert"):
        if not groq_key:
            st.error("Please provide your Groq API Key in the sidebar.")
        else:
            with st.spinner("Analyzing market structure & news..."):
                signal_data = generate_groq_signal(
                    selected_asset_label, price_data, news_list, groq_key
                )
                
                sig = signal_data.get("signal", "NEUTRAL")
                conf = signal_data.get("confidence", 0)
                
                if sig == "BUY":
                    st.success(f"### Signal: {sig} ({conf}% Confidence)")
                elif sig == "SELL":
                    st.error(f"### Signal: {sig} ({conf}% Confidence)")
                else:
                    st.warning(f"### Signal: {sig} ({conf}% Confidence)")
                
                st.markdown("**Reasoning:**")
                for reason in signal_data.get("reasons", []):
                    st.write(f"- {reason}")
                
                # Telegram Dispatch
                if telegram_token and telegram_chat_id:
                    msg = (
                        f"🚨 *NEW TRADING SIGNAL*\n\n"
                        f"📌 *Asset:* {selected_asset_label} ({selected_timeframe})\n"
                        f"🎯 *Action:* {sig}\n"
                        f"📊 *Confidence:* {conf}%\n\n"
                        f"💡 *Key Reasons:*\n" + "\n".join([f"• {r}" for r in signal_data.get("reasons", [])])
                    )
                    sent = send_telegram_message(telegram_token, telegram_chat_id, msg)
                    if sent:
                        st.success("Signal successfully sent to Telegram!")
                    else:
                        st.error("Failed to send Telegram alert.")
                else:
                    st.info("Telegram credentials not provided. Signal displayed on dashboard only.")
