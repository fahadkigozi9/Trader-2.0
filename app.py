import json
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from groq import Groq, APIError

# ------------------------------------------------------------------------------
# Configuration & Session State
# ------------------------------------------------------------------------------
st.set_page_config(page_title="AI Trading Assistant & Strategy Backtester", layout="wide")

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
# Persistence Layer
# ------------------------------------------------------------------------------
def load_trade_history() -> list:
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
    history = load_trade_history()
    history.append(signal_data)
    st.session_state["trade_history"] = history
    with open(STATE_FILE, "w") as f:
        json.dump(history, f, indent=4)

def save_all_trades(history: list):
    st.session_state["trade_history"] = history
    with open(STATE_FILE, "w") as f:
        json.dump(history, f, indent=4)

def clear_all_trade_records():
    st.session_state["trade_history"] = []
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

# ------------------------------------------------------------------------------
# Market Data & Telegram Operations
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
    if action == "BUY":
        return "BUY STOP" if entry_price > current_price else "BUY LIMIT"
    else:
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
# Fixed Backtesting Engine (Sequential Trades, Scalar Fix, & Asset Point Scaling)
# ------------------------------------------------------------------------------
def run_strategy_backtest(ticker_symbol: str, timeframe: str, lookback_days: int, signal_data: dict) -> dict:
    yf_interval_map = {"15m": "15m", "1h": "1h", "4h": "1h", "1D": "1d"}
    interval = yf_interval_map.get(timeframe, "1h")
    
    try:
        df = yf.download(ticker_symbol, period=f"{lookback_days}d", interval=interval)
        if df.empty:
            return {"error": "Failed to fetch historical data for backtesting."}
            
        # Flatten MultiIndex columns returned by newer yfinance releases
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        action = signal_data.get("signal", "BUY").upper()
        entry = float(signal_data.get("entry_price", 0.0))
        sl = float(signal_data.get("stop_loss", 0.0))
        tp = float(signal_data.get("take_profit_1", 0.0))
        
        if entry == 0.0 or sl == 0.0 or tp == 0.0:
            return {"error": "Invalid target price levels for strategy backtest."}
            
        # Asset-specific point scaling resolution
        if "GC=F" in ticker_symbol or "Gold" in ticker_symbol:
            point = 0.1
        elif "BTC" in ticker_symbol:
            point = 1.0
        else:
            point = 0.0001
        
        trades = []
        in_position = False
        
        for i in range(len(df)):
            high_val = df["High"].iloc[i]
            low_val = df["Low"].iloc[i]
            
            # Safely extract scalar value whether returned as Series or float
            high = float(high_val.iloc[0]) if isinstance(high_val, pd.Series) else float(high_val)
            low = float(low_val.iloc[0]) if isinstance(low_val, pd.Series) else float(low_val)
            
            # 1. Trigger Entry if not currently in a trade
            if not in_position:
                if (action == "BUY" and low <= entry <= high) or (action == "SELL" and low <= entry <= high):
                    in_position = True
                continue

            # 2. Evaluate Exit conditions for active position
            if in_position:
                if action == "BUY":
                    if low <= sl:
                        pips = (sl - entry) / point
                        trades.append({"result": "LOSS", "pips": pips})
                        in_position = False
                    elif high >= tp:
                        pips = (tp - entry) / point
                        trades.append({"result": "WIN", "pips": pips})
                        in_position = False
                else:  # SELL
                    if high >= sl:
                        pips = (entry - sl) / point
                        trades.append({"result": "LOSS", "pips": pips})
                        in_position = False
                    elif low <= tp:
                        pips = (entry - tp) / point
                        trades.append({"result": "WIN", "pips": pips})
                        in_position = False

        if not trades:
            return {"error": "No completed trade cycles triggered in this historical window."}

        total_trades = len(trades)
        wins = sum(1 for t in trades if t["result"] == "WIN")
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100
        net_pips = sum(t["pips"] for t in trades)

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "net_pips": round(net_pips, 1),
            "trades_detail": trades
        }
    except Exception as e:
        return {"error": f"Backtest failure: {str(e)}"}

# ------------------------------------------------------------------------------
# Post-Mortem Failure Analysis Engine
# ------------------------------------------------------------------------------
def analyze_failed_trade(trade: dict, live_price: float, api_key: str) -> str:
    if not api_key:
        return "Groq API key required."

    client = Groq(api_key=api_key)
    prompt = f"""
Perform a post-mortem failure analysis on a trade setup that hit its STOP LOSS:
- Asset: {trade.get('asset')}
- Order Type: {trade.get('order_type')}
- Entry Price: {trade.get('entry_price')}
- Stop Loss: {trade.get('stop_loss')}
- Exit Price: {live_price}

Provide a concise 3-bullet point breakdown explaining what went wrong and key learnings.
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

    return "Unable to generate failure analysis."

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

    ticker_str = saved_signal.get("ticker", "")
    asset_str = saved_signal.get("asset", "")
    
    if "GC=F" in ticker_str or "Gold" in asset_str:
        point = 0.1
    elif "BTC" in ticker_str or "Bitcoin" in asset_str:
        point = 1.0
    else:
        point = 0.0001

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
        analysis = f"❌ Trade invalidated. Hit SL ({sl}). PnL: {pnl_pips:.1f} pips."
    elif rr_achieved >= 1.0:
        recommendation = "TRAIL SL TO BREAKEVEN"
        analysis = f"🎯 1:1 R:R achieved (+{pnl_pips:.1f} pips). Lock SL to entry ({entry})."
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
st.title("📊 AI Trading Assistant & Strategy Backtester")

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

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Generate Signal", 
    "🔍 Check Status", 
    "🩺 Post-Mortem Analysis", 
    "🧪 Strategy Backtester"
])

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
                        st.session_state["latest_signal"] = setup

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
                            send_telegram_alert(telegram_token, telegram_chat_id, msg)
                    else:
                        st.warning(f"No clear setup found: {setup.get('reasons')}")

# ------------------------------------------------------------------------------
# TAB 2: CHECK STATUS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🔍 Status Check — All Stored Trades")
    trade_history = load_trade_history()

    if not trade_history:
        st.info("No saved trades found in memory.")
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

# ------------------------------------------------------------------------------
# TAB 3: POST-MORTEM ANALYSIS
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🩺 AI Post-Mortem Analysis on Failed Trades")
    trade_history = load_trade_history()
    sl_trades = [t for t in trade_history if t.get("status") == "STOP LOSS HIT"]

    if not sl_trades:
        st.info("No trades marked as 'STOP LOSS HIT' yet.")
    else:
        for trade in reversed(sl_trades):
            with st.expander(f"❌ {trade.get('asset')} ({trade.get('order_type')}) — Entry: {trade.get('entry_price')}", expanded=True):
                if "post_mortem" in trade:
                    st.write(trade["post_mortem"])
                else:
                    if st.button(f"Analyze Why This Trade Failed ({trade.get('id')})", key=trade.get('id')):
                        live_data = fetch_live_price(trade.get("ticker", selected_ticker))
                        analysis_result = analyze_failed_trade(trade, live_data.get("price", 0.0), groq_key)
                        trade["post_mortem"] = analysis_result
                        save_all_trades(trade_history)
                        st.rerun()

# ------------------------------------------------------------------------------
# TAB 4: STRATEGY BACKTESTER
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🧪 Backtest AI Signal Setup against Historical Data")
    
    trade_history = load_trade_history()
    
    if not trade_history:
        st.warning("Generate a trade signal first or select parameters manually to test.")
    else:
        selected_trade_id = st.selectbox(
            "Select AI Signal to Backtest", 
            options=[t.get("id") for t in reversed(trade_history)]
        )
        
        target_trade = next((t for t in trade_history if t.get("id") == selected_trade_id), None)
        
        if target_trade:
            st.json({
                "Asset": target_trade.get("asset"),
                "Signal": target_trade.get("signal"),
                "Entry": target_trade.get("entry_price"),
                "SL": target_trade.get("stop_loss"),
                "TP": target_trade.get("take_profit_1")
            })
            
            lookback = st.slider("Historical Lookback (Days)", min_value=5, max_value=60, value=30)
            
            if st.button("▶️ Run Backtest Engine", type="primary"):
                with st.spinner("Simulating strategy across historical candles..."):
                    res = run_strategy_backtest(
                        target_trade.get("ticker", selected_ticker),
                        target_trade.get("timeframe", "1h"),
                        lookback,
                        target_trade
                    )
                    
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        st.success("Backtest Complete!")
                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric("Total Executions", res["total_trades"])
                        b2.metric("Win Rate", f"{res['win_rate']}%")
                        b3.metric("Wins / Losses", f"{res['wins']} W / {res['losses']} L")
                        b4.metric("Net Cumulative Pips", f"{res['net_pips']:+} pips")
