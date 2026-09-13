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
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-specdec",
    "llama3-70b-8192"
]


def generate_groq_signal(asset: str, price_data: dict, news_data: list, api_key: str) -> dict:
    """Generate trade signal and reasoning using Groq API with active model fallbacks."""
    if not api_key:
        return {"signal": "ERROR", "confidence": 0, "reasons": ["Groq API key missing"]}

    client = Groq(api_key=api_key)

    prompt = f"""
    You are an expert AI trading strategist. Analyze the following market data for {asset}:
    - Current Price: {price_data.get('price')}
    - Daily Change (%): {price_data.get('change'):.2f}%
    - High-Impact News: {json.dumps(news_data, indent=2)}

    Provide your response in strictly VALID JSON format matching this schema:
    {{
        "signal": "BUY" | "SELL" | "NEUTRAL",
        "confidence": number between 0 and 100,
        "reasons": ["reason 1", "reason 2"]
    }}
    Return ONLY raw JSON, with no markdown formatting.
    """

    for model in MODEL_FALLBACKS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw_content = response.choices[0].message.content
            
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
                
            return json.loads(raw_content.strip())
        except APIError as e:
            err_msg = str(e).lower()
            if e.status_code in [400, 404] or "model" in err_msg or "decommissioned" in err_msg:
                st.warning(f"Model `{model}` unavailable/decommissioned. Trying fallback...")
                continue
            else:
                st.error(f"Groq API Error on {model}: {e.message}")
                return {"signal": "ERROR", "confidence": 0, "reasons": [str(e)]}
        except Exception as e:
            st.warning(f"Unexpected error on model `{model}`: {str(e)}. Trying fallback...")
            continue

    st.error("All model fallbacks failed. Check your Groq API Key validity in console.groq.com.")
    return {"signal": "ERROR", "confidence": 0, "reasons": ["All model fallbacks failed"]}


# ------------------------------------------------------------------------------
# Streamlit User Interface
# ------------------------------------------------------------------------------
st.title("📈 AI Trading Assistant Dashboard")

# Load Credentials from st.secrets if available
default_groq_key = st.secrets.get("GROQ_API_KEY", "")
default_telegram_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
default_telegram_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")

# Sidebar - Settings & Credentials
st.sidebar.header("🔑 Credentials & Setup")
groq_key = st.sidebar.text_input("Groq API Key", value=default_groq_key, type="password")
telegram_token = st.sidebar.text_input("Telegram Bot Token", value=default_telegram_token, type="password")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID", value=default_telegram_chat_id)

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
            st.error("Please provide your Groq API Key in the sidebar or app secrets.")
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
        return False


# ------------------------------------------------------------------------------
# Groq AI Inference Function
# ------------------------------------------------------------------------------
MODEL_FALLBACKS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-specdec",
    "llama3-70b-8192"
]


def generate_groq_signal(asset: str, price_data: dict, news_data: list, api_key: str) -> dict:
    """Generate trade signal and reasoning using Groq API with active model fallbacks."""
    if not api_key:
        return {"signal": "ERROR", "confidence": 0, "reasons": ["Groq API key missing"]}

    client = Groq(api_key=api_key)

    prompt = f"""
    You are an expert AI trading strategist. Analyze the following market data for {asset}:
    - Current Price: {price_data.get('price')}
    - Daily Change (%): {price_data.get('change'):.2f}%
    - High-Impact News: {json.dumps(news_data, indent=2)}

    Provide your response in strictly VALID JSON format matching this schema:
    {{
        "signal": "BUY" | "SELL" | "NEUTRAL",
        "confidence": number between 0 and 100,
        "reasons": ["reason 1", "reason 2"]
    }}
    Return ONLY raw JSON, with no markdown formatting.
    """

    for model in MODEL_FALLBACKS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw_content = response.choices[0].message.content
            
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
                
            return json.loads(raw_content.strip())
        except APIError as e:
            err_msg = str(e).lower()
            if e.status_code in [400, 404] or "model" in err_msg or "decommissioned" in err_msg:
                st.warning(f"Model `{model}` unavailable/decommissioned. Trying fallback...")
                continue
            else:
                st.error(f"Groq API Error on {model}: {e.message}")
                return {"signal": "ERROR", "confidence": 0, "reasons": [str(e)]}
        except Exception as e:
            st.warning(f"Unexpected error on model `{model}`: {str(e)}. Trying fallback...")
            continue

    st.error("All model fallbacks failed. Check your Groq API Key validity in console.groq.com.")
    return {"signal": "ERROR", "confidence": 0, "reasons": ["All model fallbacks failed"]}


# ------------------------------------------------------------------------------
# Streamlit User Interface
# ------------------------------------------------------------------------------
st.title("📈 AI Trading Assistant Dashboard")

# Load Credentials from st.secrets if available
default_groq_key = st.secrets.get("GROQ_API_KEY", "")
default_telegram_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
default_telegram_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")

# Sidebar - Settings & Credentials
st.sidebar.header("🔑 Credentials & Setup")
groq_key = st.sidebar.text_input("Groq API Key", value=default_groq_key, type="password")
telegram_token = st.sidebar.text_input("Telegram Bot Token", value=default_telegram_token, type="password")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID", value=default_telegram_chat_id)

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
            st.error("Please provide your Groq API Key in the sidebar or app secrets.")
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
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192"
]


def generate_groq_signal(asset: str, price_data: dict, news_data: list, api_key: str) -> dict:
    """Generate trade signal and reasoning using Groq API with active model fallbacks."""
    if not api_key:
        return {"signal": "ERROR", "confidence": 0, "reasons": ["Groq API key missing"]}

    client = Groq(api_key=api_key)

    prompt = f"""
    You are an expert AI trading strategist. Analyze the following market data for {asset}:
    - Current Price: {price_data.get('price')}
    - Daily Change (%): {price_data.get('change'):.2f}%
    - High-Impact News: {json.dumps(news_data, indent=2)}

    Provide your response in strictly VALID JSON format matching this schema:
    {{
        "signal": "BUY" | "SELL" | "NEUTRAL",
        "confidence": number between 0 and 100,
        "reasons": ["reason 1", "reason 2"]
    }}
    Return ONLY raw JSON, with no markdown formatting.
    """

    for model in MODEL_FALLBACKS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw_content = response.choices[0].message.content
            
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
                
            return json.loads(raw_content.strip())
        except APIError as e:
            if e.status_code in [404, 400] or "model_not_found" in str(e) or "model_decommissioned" in str(e):
                st.warning(f"Model {model} unavailable/decommissioned. Trying fallback...")
                continue
            else:
                st.error(f"Groq API Error on {model}: {e.message}")
                return {"signal": "ERROR", "confidence": 0, "reasons": [str(e)]}
        except Exception as e:
            return {"signal": "ERROR", "confidence": 0, "reasons": [str(e)]}

    st.error("All model fallbacks failed.")
    return {"signal": "ERROR", "confidence": 0, "reasons": ["All model fallbacks failed"]}


# ------------------------------------------------------------------------------
# Streamlit User Interface
# ------------------------------------------------------------------------------
st.title("📈 AI Trading Assistant Dashboard")

# Load Credentials from st.secrets if available
default_groq_key = st.secrets.get("GROQ_API_KEY", "")
default_telegram_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
default_telegram_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")

# Sidebar - Settings & Credentials
st.sidebar.header("🔑 Credentials & Setup")
groq_key = st.sidebar.text_input("Groq API Key", value=default_groq_key, type="password")
telegram_token = st.sidebar.text_input("Telegram Bot Token", value=default_telegram_token, type="password")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID", value=default_telegram_chat_id)

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
            st.error("Please provide your Groq API Key in the sidebar or app secrets.")
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
