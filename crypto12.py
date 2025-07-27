import datetime
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
import ccxt
import ta
from streamlit_autorefresh import st_autorefresh


# Set up page title, icon and wide layout
st.set_page_config(
    page_title="Crypto Market Dashboard",
    page_icon="📊",
    layout="wide",
)


PRIMARY_BG = "#0D1117"        # overall app background – near‑black
CARD_BG    = "#16213E"        # cards and panel backgrounds – dark blue
ACCENT     = "#FFFFFF"        # white colour
POSITIVE   = "#06D6A0"        # green for positive change
NEGATIVE   = "#EF476F"        # red for negative change
WARNING    = "#FFD166"        # yellow for neutral or caution
TEXT_LIGHT = "#E5E7EB"        # light text colour
TEXT_MUTED = "#8CA1B6"        # muted grey for secondary text


st.markdown(
    f"""
    <style>
    /* Global styles */
    .stApp {{
        background-colour: {PRIMARY_BG};
        colour: {TEXT_LIGHT};
        font-family: 'Segoe UI', sans-serif;
    }}

    /* Titles and subtitles */
    h1.title {{
        font-size: 2.4rem;
        font-weight: 700;
        colour: {ACCENT};
        margin-bottom: 0.4rem;
    }}
    p.subtitle {{
        font-size: 1.05rem;
        colour: {TEXT_MUTED};
        margin-top: 0;
        margin-bottom: 2rem;
    }}

    /* Card styling */
    .metric-card {{
        background-colour: {CARD_BG};
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 24px 20px;
        text-align: centre;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }}
    .metric-label {{
        font-size: 0.9rem;
        colour: {TEXT_MUTED};
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 600;
        colour: {ACCENT};
    }}
    .metric-delta-positive, .metric-delta-negative {{
        font-size: 0.9rem;
        font-weight: 500;
    }}
    .metric-delta-positive {{ colour: {POSITIVE}; }}
    .metric-delta-negative {{ colour: {NEGATIVE}; }}

    /* Panel boxes for larger sections */
    .panel-box {{
        background-colour: {CARD_BG};
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 32px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
    }}

    /* Table styling */
    .table-container {{ overflow-x: auto; }}
    table.dataframe {{
        width: 100% !important;
        border-collapse: collapse;
        background-colour: {CARD_BG};
    }}
    table.dataframe thead tr {{
        background-colour: {CARD_BG};
    }}
    table.dataframe th {{
        colour: {ACCENT};
        padding: 10px;
        text-align: left;
        font-size: 0.9rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}
    table.dataframe td {{
        padding: 10px;
        font-size: 0.9rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        colour: {TEXT_LIGHT};
    }}

    /* Remove row hover highlight from Streamlit table */
    table.dataframe tbody tr:hover {{
        background-colour: rgba(255, 255, 255, 0.03);
    }}
    </style>
    """,
    unsafe_allow_html=True
)



# Exchange set up with caching
@st.cache_resource(show_spinner=False)
def get_exchange():
    """Load the Binance exchange via ccxt."""
    return ccxt.binance()

EXCHANGE = get_exchange()

# Fetch BTC and ETH prices in USD from CoinGecko
@st.cache_data(ttl=120, show_spinner=False)
def get_btc_eth_prices():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin,ethereum", "vs_currencies": "usd"}
        response = requests.get(url, params=params).json()
        return response.get("bitcoin", {}).get("usd", 0), response.get("ethereum", {}).get("usd", 0)
    except Exception:
        return 0, 0

# Fetch market dominance and total/alt market cap from CoinGecko
@st.cache_data(ttl=1800, show_spinner=False)
def get_market_indices():
    try:
        data = requests.get("https://api.coingecko.com/api/v3/global").json().get("data", {})
        btc_dom = data.get("market_cap_percentage", {}).get("btc", 0) * 100
        eth_dom = data.get("market_cap_percentage", {}).get("eth", 0) * 100
        total_mcap = data.get("total_market_cap", {}).get("usd", 0)
        alt_mcap = total_mcap * (1 - btc_dom / 100)
        return round(btc_dom, 2), round(eth_dom, 2), int(total_mcap), int(alt_mcap)
    except Exception:
        return 0.0, 0.0, 0, 0

# Fetch fear and greed index from alternative.me
@st.cache_data(ttl=300, show_spinner=False)
def get_fear_greed():
    try:
        data = requests.get("https://api.alternative.me/fng/?limit=1").json()
        value = int(data.get("data", [{}])[0].get("value", 0))
        label = data.get("data", [{}])[0].get("value_classification", "Unknown")
        return value, label
    except Exception:
        return 0, "Unknown"

@st.cache_data(ttl=300, show_spinner=False)
def get_social_sentiment(symbol: str) -> tuple[int, str]:
    """Return a naive sentiment score (0–100) and label based on 24h price change.

    The score is centred at 50 with each percentage point of change shifting
    the score by one point.  For example, a +5% move yields a score of 55,
    while a −10% move yields 40.  The score is clipped between 0 and 100.
    """
    try:
        change = get_price_change(symbol) or 0.0
    except Exception:
        change = 0.0
    # Map change to a 0–100 scale around 50
    score = int(max(0, min(100, 50 + change)))
    # Determine sentiment category
    if score >= 75:
        label = "Strongly Bullish"
    elif score >= 55:
        label = "Bullish"
    elif score >= 45:
        label = "Neutral"
    elif score >= 25:
        label = "Bearish"
    else:
        label = "Strongly Bearish"
    return score, label

def get_trending_topics() -> list[str]:
    """Return a list of current trending crypto topics (placeholder)."""
    return [
        "ETF approval",
        "SEC lawsuit",
        "Layer‑2 scaling",
        "CBDCs discussion",
        "DeFi regulation",
    ]

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR) for given price dataframe."""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


# Fetch previous day's total market cap from CoinGecko
@st.cache_data(ttl=3600, show_spinner=False)
def get_yesterday_mcap():
    try:
        data = requests.get("https://api.coingecko.com/api/v3/global").json().get("data", {})
        return int(data.get("total_market_cap", {}).get("usd", 0))
    except Exception:
        return 0

# Fetch price change percentage for a given symbol via ccxt
def get_price_change(symbol: str) -> float | None:
    try:
        ticker = EXCHANGE.fetch_ticker(symbol)
        percent = ticker.get("percentage")
        return round(percent, 2) if percent is not None else None
    except Exception:
        return None

# Fetch OHLCV data for a symbol and timeframe
@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame | None:
    try:
        data = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception:
        return None

# Calculate strength of a coin based on EMA and RSI divergence
def calculate_strength(latest, max_ema_diff: float, max_rsi_dist: float) -> float:
    ema_diff = abs(latest["ema9"] - latest["ema21"])
    rsi_dist = abs(latest["rsi"] - 50)
    ema_score = min((ema_diff / max(max_ema_diff, 1)) * 5, 5)
    rsi_score = min((rsi_dist / max(max_rsi_dist, 1)) * 5, 5)
    return round(min(ema_score + rsi_score, 10), 2)

# Convert numerical strength to confidence levels for sorting
def strength_to_confidence_level(strength: float) -> int:
    if strength >= 8:
        return 3  # HIGH
    elif strength >= 5:
        return 2  # MEDIUM
    else:
        return 1  # LOW

def confidence_level_badge(strength: float) -> str:
    if strength >= 9.5:
        return "🔵 MAXIMUM"
    elif strength >= 7:
        return "🟢 HIGH"
    elif strength >= 4:
        return "🟠 MEDIUM"
    else:
        return "🔴 LOW"

def signal_badge(signal: str) -> str:
    """Return a simplified badge for the given signal."""
    if signal in ("STRONG BUY", "BUY"):
        return "🟢 LONG"
    elif signal in ("STRONG SELL", "SELL"):
        return "🔴 SHORT"
    else:
        return "⚪ WAIT"


def leverage_badge(lev: int) -> str:
    """Display leverage as a formatted badge (e.g. x5)."""
    return f"x{lev}"


def strength_badge(strength: float) -> str:
    """Return a colour‑coded strength badge."""
    if strength >= 8:
        return f"🟢 {strength:.2f} / 10 (Strong)"
    elif strength >= 5:
        return f"🟠 {strength:.2f} / 10 (Medium)"
    else:
        return f"🔴 {strength:.2f} / 10 (Weak)"


def signal_plain(signal: str) -> str:
    """Map detailed signals to a plain LONG/SHORT/WAIT label."""
    if signal in ("STRONG BUY", "BUY"):
        return "LONG"
    elif signal in ("STRONG SELL", "SELL"):
        return "SHORT"
    else:
        return "WAIT"


def strength_label(strength: float) -> str:
    """Return a plain strength description without emojis."""
    if strength >= 8:
        return f"{strength:.2f} / 10 (Strong)"
    elif strength >= 5:
        return f"{strength:.2f} / 10 (Medium)"
    else:
        return f"{strength:.2f} / 10 (Weak)"


def confidence_label(strength: float) -> str:
    """Return a plain confidence label based on strength."""
    if strength >= 9.5:
        return "MAXIMUM"
    elif strength >= 7:
        return "HIGH"
    elif strength >= 4:
        return "MEDIUM"
    else:
        return "LOW"

def format_delta(delta):
    """Format delta with unicode triangle and 2 decimals for DataFrame display."""
    if delta is None:
        return ''
    triangle = "▲" if delta > 0 else "▼"
    return f"{triangle} {abs(delta):.2f}%"

def readable_market_cap(value):
    trillion = 1_000_000_000_000
    billion = 1_000_000_000
    million = 1_000_000

    if value >= trillion:
        return f"{value / trillion:.2f}T"
    elif value >= billion:
        return f"{value / billion:.2f}B"
    elif value >= million:
        return f"{value / million:.2f}M"
    else:
        return f"{value:,}"
def detect_volume_spike(df, window: int = 20, multiplier: float = 2.0) -> bool:
    """
    Detects if the last volume bar is a significant spike compared to recent average.

    Parameters:
    - df: DataFrame containing 'volume' column.
    - window: Number of past bars to average.
    - multiplier: Threshold multiplier for detecting spike.

    Returns:
    - True if spike detected, otherwise False.
    """
    if 'volume' not in df or len(df) < window + 1:
        return False

    recent_volumes = df['volume'][-(window+1):-1]
    avg_volume = recent_volumes.mean()
    last_volume = df['volume'].iloc[-1]

    return last_volume > avg_volume * multiplier

def detect_candle_pattern(df: pd.DataFrame) -> str:
    if df is None or len(df) < 2:
        return ""

    last = df.iloc[-1]
    prev = df.iloc[-2]

    body_last = abs(last['close'] - last['open'])
    body_prev = abs(prev['close'] - prev['open'])

    # Bullish Engulfing
    if prev['close'] < prev['open'] and last['close'] > last['open'] and \
       last['close'] > prev['open'] and last['open'] < prev['close']:
        return "Bullish Engulfing (strong reversal up)"

    # Bearish Engulfing
    if prev['close'] > prev['open'] and last['close'] < last['open'] and \
       last['open'] > prev['close'] and last['close'] < prev['open']:
        return "Bearish Engulfing (strong reversal down)"

    # Hammer
    lower_shadow = last['open'] - last['low'] if last['open'] > last['close'] else last['close'] - last['low']
    upper_shadow = last['high'] - last['close'] if last['open'] > last['close'] else last['high'] - last['open']
    if body_last < lower_shadow and upper_shadow < lower_shadow * 0.5:
        return "Hammer (bullish bottom wick)"

    # Doji
    if body_last / (last['high'] - last['low'] + 1e-9) < 0.1:
        return "Doji (market indecision)"

    return ""


def explain_candle_pattern(pattern: str) -> str:
    explanations = {
        "Bullish Engulfing": "strong reversal up",
        "Bearish Engulfing": "strong reversal down",
        "Hammer": "bullish bottom wick",
        "Doji": "market indecision"
    }
    return explanations.get(pattern, "")


# Analyse a dataframe of price data and return signal, leverage, strength, and comment
def analyse(df: pd.DataFrame) -> tuple[str, int, float, str, bool, str, str]:
    if df is None or len(df) < 30:
        return "NO DATA", 1, 0.0, "Insufficient data", False, "", ""

    df["ema5"] = ta.trend.ema_indicator(df["close"], window=5)
    df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    latest = df.iloc[-1]

    volume_spike = detect_volume_spike(df)
    candle_pattern = detect_candle_pattern(df)  # örn: "Bullish Engulfing (strong reversal up)"
    candle_label = candle_pattern.split(" (")[0] if candle_pattern else ""

    atr_latest = latest["atr"]
    if atr_latest > latest["close"] * 0.05:
        atr_comment = "▲ High"
    elif atr_latest < latest["close"] * 0.02:
        atr_comment = "▼ Low"
    else:
        atr_comment = "– Moderate"

    # === Strength points ===
    strength_score = 0.0
    
    # EMA trend check (5 > 9 > 21 > 50)
    ema_trend = [latest["ema5"] > latest["ema9"],
                 latest["ema9"] > latest["ema21"],
                 latest["ema21"] > latest["ema50"]]
    strength_score += 2 if sum(ema_trend) >= 2 else 0
    
    # RSI (14) scoring
    if latest["rsi"] > 60:
        strength_score += 2
    elif latest["rsi"] > 55:
        strength_score += 1
    
    # MACD: only if MACD > signal and histogram is positive
    if latest["macd"] > latest["macd_signal"] and latest["macd_diff"] > 0:
        strength_score += 1
    
    # OBV increase
    if df["obv"].iloc[-1] > df["obv"].iloc[-5]:
        strength_score += 1
    
    # Volume spike
    if volume_spike:
        strength_score += 1
    
    # ATR volatility
    if atr_comment == "– Moderate":
        strength_score += 1
    elif atr_comment == "▲ High":
        strength_score += 0.5
    
    # Candle pattern + EMA trend alignment
    bullish_patterns = ["Bullish Engulfing", "Hammer"]
    if candle_label in bullish_patterns and sum(ema_trend) >= 2:
        strength_score += 2
    
    # Final strength
    strength = round(min(strength_score, 10), 2)

    # === Leverage points ===
    risk_score = 0.0
    bollinger_width = (df["close"].rolling(20).std() * 2).iloc[-1]
    volatility_factor = min(bollinger_width / latest["close"], 0.1)
    rsi_factor = 0.1 if latest["rsi"] > 70 or latest["rsi"] < 30 else 0
    obv_factor = 0.1 if df["obv"].iloc[-1] > df["obv"].iloc[-5] and latest["close"] > latest["ema21"] else 0
    recent = df.tail(20)
    support = recent["low"].min()
    resistance = recent["high"].max()
    current_price = latest["close"]
    sr_factor = 0.1 if abs(current_price - support) / current_price < 0.02 or abs(current_price - resistance) / current_price < 0.02 else 0
    risk_score = volatility_factor + rsi_factor + obv_factor + sr_factor

    if risk_score < 0.15:
        lev_base = int(round(np.interp(risk_score, [0, 0.15], [3, 7])))
    elif risk_score < 0.25:
        lev_base = int(round(np.interp(risk_score, [0.15, 0.25], [8, 12])))
    else:
        lev_base = int(round(np.interp(min(risk_score, 0.4), [0.25, 0.4], [13, 20])))

    if strength < 4:
        lev_base = min(lev_base, 4)
    elif strength < 6:
        lev_base = min(lev_base, 8)

    # === Signal & Comment ===
    if strength >= 8:
        signal = "STRONG BUY"
        comment = "🚀 Strong bullish momentum detected. High confidence to go LONG."
    elif strength >= 6:
        signal = "BUY"
        comment = "📈 Multiple bullish signals. Consider LONG with moderate risk."
    elif strength >= 4:
        signal = "WAIT"
        comment = "⏳ Mixed signals. Wait for trend confirmation."
    elif strength >= 2:
        signal = "SELL"
        comment = "📉 Bearish pressure forming. SHORT may be considered."
    else:
        signal = "STRONG SELL"
        comment = "⚠️ Strong bearish signal. SHORT with high conviction."

    return signal, lev_base, strength, comment, volume_spike, atr_comment, candle_pattern

def render_market_tab():
    """Render the Market Dashboard tab containing top‑level crypto metrics and scanning."""

    # Fetch global market data
    btc_dom, eth_dom, total_mcap, alt_mcap = get_market_indices()
    fg_value, fg_label = get_fear_greed()
    btc_price, eth_price = get_btc_eth_prices()
    yesterday_mcap = get_yesterday_mcap()

    # Compute percentage change for market cap
    delta_mcap = 0.0
    if yesterday_mcap:
        delta_mcap = ((total_mcap - yesterday_mcap) / yesterday_mcap) * 100

    # Compute price change percentages using ccxt
    btc_change = get_price_change("BTC/USDT")
    eth_change = get_price_change("ETH/USDT")

    # Display headline and subtitle
    st.markdown("<h1 class='title'>Crypto Market Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-size:0.94rem;'>"
        "Live metrics for BTC, ETH and the broader market. "
        "Top coins are dynamically selected based on 24h volume rankings from CoinGecko, "
        "and filtered to include only USDT pairs actively traded on the exchange. "
        "Each coin is scored based on real-time technical signals."
        "</p>",
        unsafe_allow_html=True,
    )

    # Top row: Price and market cap metrics
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        delta_class = "metric-delta-positive" if (btc_change or 0) >= 0 else "metric-delta-negative"
        delta_text = f"({btc_change:+.2f}%)" if btc_change is not None else ""
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Bitcoin Price</div>"
            f"  <div class='metric-value'>${btc_price:,.2f}</div>"
            f"  <div class='{delta_class}'>{delta_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m2:
        delta_class = "metric-delta-positive" if (eth_change or 0) >= 0 else "metric-delta-negative"
        delta_text = f"({eth_change:+.2f}%)" if eth_change is not None else ""
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Ethereum Price</div>"
            f"  <div class='metric-value'>${eth_price:,.2f}</div>"
            f"  <div class='{delta_class}'>{delta_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m3:
        delta_class = "metric-delta-positive" if delta_mcap >= 0 else "metric-delta-negative"
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Total Market Cap</div>"
            f"  <div class='metric-value'>${total_mcap / 1e12:.2f}T</div>"
            f"  <div class='{delta_class}'>({delta_mcap:+.2f}%)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m4:
        # Colour for fear & greed based on sentiment
        sentiment_colour = POSITIVE if "Greed" in fg_label else (NEGATIVE if "Fear" in fg_label else WARNING)
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-label'>Fear &amp; Greed</div>"
            f"  <div class='metric-value'>{fg_value}</div>"
            f"  <div style='color:{sentiment_colour};font-size:0.9rem;'>{fg_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Second row: Dominance gauges
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        btc_dom_int = round(btc_dom / 100)
        fig_btc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=btc_dom_int,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 40], 'color': NEGATIVE},
                    {'range': [40, 60], 'color': WARNING},
                    {'range': [60, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'BTC Dominance (%)', 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': ACCENT, 'size': 38}},
        ))
        fig_btc.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
        )
        st.plotly_chart(fig_btc, use_container_width=True)
    with g2:
        eth_dom_int = round(eth_dom / 100)
        fig_eth = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eth_dom_int,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 15], 'color': NEGATIVE},
                    {'range': [15, 25], 'color': WARNING},
                    {'range': [25, 100], 'color': POSITIVE},
                ],
            },
            title={'text': 'ETH Dominance (%)', 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': ACCENT, 'size': 38}},
        ))
        fig_eth.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            
        )
        st.plotly_chart(fig_eth, use_container_width=True)

    # Divider
    st.markdown("\n\n")

    # Top coin scanner controls
    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.5rem;'>Top Coin Finder</h2>",
        unsafe_allow_html=True,
    )
    controls = st.columns([1.5, 1.5, 1.5, 1], gap="medium")
    with controls[0]:
        timeframe = st.selectbox("Select timeframe", ['5m', '15m', '1h', '4h', '1d'], index=2)
    with controls[1]:
        sort_option = st.selectbox("Sort by", ['Strength', 'Confidence', 'Combined'], index=0)
    with controls[2]:
        signal_filter = st.selectbox("Signal", ['LONG', 'SHORT', 'BOTH'], index=2)
    with controls[3]:
        top_n = st.slider("Top N", min_value=3, max_value=20, value=20)

    # Fetch top coins
    with st.spinner(f"Scanning {top_n} coins ({signal_filter}) [{timeframe}] ..."):
        # Obtain top USDT trading pairs from CoinGecko and filter by exchange markets
        def get_top_volume_usdt_symbols(top_n: int = 100, vs_currency: str = "usd"):
            try:
                url = "https://api.coingecko.com/api/v3/coins/markets"
                params = {
                    "vs_currency": vs_currency,
                    "order": "volume_desc",
                    "per_page": min(top_n, 250),
                    "page": 1,
                    "sparkline": False,
                }
                data = requests.get(url, params=params).json()
                markets = EXCHANGE.load_markets()
                valid = []
                seen = set()
        
                for coin in data:
                    symbol = coin.get("symbol", "").upper()
                    if symbol in seen:
                        continue  # skip if the same symbol
                    seen.add(symbol)
        
                    pair = f"{symbol}/USDT"
                    if pair in markets:
                        valid.append(pair)
        
                return valid, data
            except Exception:
                return [], []


        # Fetch mapping of symbols to market cap for the top coins
        TOP_VOLUME_LIMIT = 150
        usdt_symbols, market_data = get_top_volume_usdt_symbols(TOP_VOLUME_LIMIT)
        
        # Filter same symbol coins (eg. SOL vs Wrapped SOL) and skip wrapped coins
        seen_symbols = set()
        unique_market_data = []
        
        for coin in market_data:
            coin_id = coin.get("id", "").lower()
            symbol = coin.get("symbol", "").upper()
        
            if "wrapped" in coin_id:
                continue  # wrapped coin'leri atla
        
            if symbol in seen_symbols:
                continue  # aynı symbol varsa atla
        
            seen_symbols.add(symbol)
            unique_market_data.append(coin)
        
        # Create market cap data
        mcap_map = {}
        for coin in unique_market_data:
            symbol = coin.get("symbol", "").upper()
            name = coin.get("name", "").upper()
            mcap = coin.get("market_cap", 0)
        
            if symbol:
                if symbol not in mcap_map or mcap > mcap_map[symbol]:
                    mcap_map[symbol] = int(mcap)
            if name:
                if name not in mcap_map or mcap > mcap_map[name]:
                    mcap_map[name] = int(mcap)

        usdt_symbols, market_data = get_top_volume_usdt_symbols(max(top_n, 50))
        # Precompute EMA and RSI differences and keep track of market cap for each pair
        ema_diffs: list[float] = []
        rsi_dists: list[float] = []
        coin_data: list[tuple[str, pd.Series, pd.DataFrame, int]] = []
        for sym in usdt_symbols:
            df = fetch_ohlcv(sym, timeframe)
            if df is not None and len(df) > 30:
                df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
                df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
                latest = df.iloc[-1]
                ema_diffs.append(abs(latest['ema9'] - latest['ema21']))
                rsi_dists.append(abs(latest['rsi'] - 50))
                base = sym.split('/')[0]
                mcap_val = mcap_map.get(base.upper(), 0)

                if mcap_val == 0:
                    st.write("⚠️ MCAP 0:", sym, "| base:", base, "| lookup:", base.upper(), "| Known Keys:", list(mcap_map.keys())[:5])
                coin_data.append((sym, latest, df, mcap_val))
        max_ema = max(ema_diffs) if ema_diffs else 1
        max_rsi = max(rsi_dists) if rsi_dists else 1

        results: list[dict] = []
        for sym, latest, df, mcap_val in coin_data:
            price = latest['close']
            price_change = get_price_change(sym)
            # Analyse coin using original logic
            signal, lev, strength, comment, volume_spike, atr_comment, candle_pattern = analyse(df)

            # Compute scalping entry and target prices to guide leverage tiers
            entry_price = 0.0
            target_price = 0.0
            df_scalp = fetch_ohlcv(sym, timeframe, limit=100)
            if df_scalp is not None and len(df_scalp) > 30:
                df_scalp['ema_fast'] = df_scalp['close'].ewm(span=5).mean()
                df_scalp['ema_slow'] = df_scalp['close'].ewm(span=13).mean()
                fast = df_scalp['ema_fast'].iloc[-1]
                slow = df_scalp['ema_slow'].iloc[-1]
                support = df_scalp['low'].tail(20).min()
                resistance = df_scalp['high'].tail(20).max()
                if fast > slow:
                    entry_price = support * 1.005
                    target_price = resistance
                else:
                    entry_price = resistance * 0.995
                    target_price = support

            # Determine leverage tiers
            conservative_lev = min(lev, 7)
            medium_risk_lev = min(lev + 3, 14)
            high_risk_lev = min(lev + 6, 20)

            # Only include coins based on the selected signal filter
            include = False
            if signal_filter == 'BOTH':
                include = signal in ['STRONG BUY', 'BUY', 'WAIT', 'SELL', 'STRONG SELL']
            elif signal_filter == 'LONG':
                include = signal in ['STRONG BUY', 'BUY']
            elif signal_filter == 'SHORT':
                include = signal in ['STRONG SELL', 'SELL']
            if include:
                results.append({
                    'Coin': sym.split('/')[0],
                    'Price ($)': f"{price:,.2f}",
                    # Use plain labels (no emojis) for signal, strength and confidence
                    'Signal': signal_plain(signal),
                    'Strength': strength_label(strength),
                    'Confidence': confidence_label(strength),
                    'Market Cap ($)': readable_market_cap(mcap_val),
                    'Low Risk (X)': leverage_badge(conservative_lev),
                    'Medium Risk (X)': leverage_badge(medium_risk_lev),
                    'High Risk (X)': leverage_badge(high_risk_lev),
                    'Entry Price': f"${entry_price:,.2f}" if entry_price else '',
                    'Target Price': f"${target_price:,.2f}" if target_price else '',
                    'Δ (%)': format_delta(price_change) if price_change is not None else '',
                    'Spike Alert': '▲ Spike' if volume_spike else '',
                    'Volatility': atr_comment,
                    'Candle Pattern': candle_pattern,
                    '__strength_val': strength,
                    '__confidence_val': strength_to_confidence_level(strength),
                })

        # Sort results according to user selection
        if sort_option == "Strength":
            results = sorted(results, key=lambda x: x['__strength_val'], reverse=True)
        elif sort_option == "Confidence":
            results = sorted(results, key=lambda x: x['__confidence_val'], reverse=True)
        else:
            results = sorted(results, key=lambda x: x['__strength_val'] * x['__confidence_val'], reverse=True)

        # Limit to top_n
        results = results[:top_n]

        # Prepare DataFrame for display
        if results:
            df_results = pd.DataFrame(results)
            df_display = df_results.drop(columns=['__strength_val', '__confidence_val'])
  
            def style_signal(val: str) -> str:
                if 'LONG' in val:
                    return f'color: {POSITIVE}; font-weight: 600;'
                if 'SHORT' in val:
                    return f'color: {NEGATIVE}; font-weight: 600;'
                return f'color: {WARNING}; font-weight: 600;'

            def style_strength(val: str) -> str:
                if 'Strong' in val:
                    return f'color: {POSITIVE}; font-weight: 600;'
                if 'Medium' in val:
                    return f'color: {WARNING}; font-weight: 600;'
                return f'color: {NEGATIVE}; font-weight: 600;'

            def style_confidence(val: str) -> str:
                # Colour coding for confidence: treat MAXIMUM the same as HIGH (green)
                if 'MAXIMUM' in val:
                    return f'color: {POSITIVE}; font-weight: 600;'
                if 'HIGH' in val:
                    return f'color: {POSITIVE}; font-weight: 600;'
                if 'MEDIUM' in val:
                    return f'color: {WARNING}; font-weight: 600;'
                return f'color: {NEGATIVE}; font-weight: 600;'

            styled = (
                df_display.style
                .applymap(style_signal, subset=['Signal'])
                .applymap(style_strength, subset=['Strength'])
                .applymap(style_confidence, subset=['Confidence'])
            )
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("No coins matched the criteria.")



def render_spot_tab():
    """Render the Spot Trading tab which allows instant analysis of a selected coin."""
    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.5rem;'>Spot Trading</h2>",
        unsafe_allow_html=True,
    )
    coin = st.text_input("Enter coin symbol (e.g., BTC/USDT)", value="BTC/USDT").upper()
    timeframe = st.selectbox("Timeframe", ['1m', '3m', '5m', '15m', '1h', '4h', '1d'], index=4)
    if st.button("Analyse", type="primary"):
        df = fetch_ohlcv(coin, timeframe)
        if df is None or len(df) < 30:
            st.error("Could not fetch data or not enough candles. Try another symbol/timeframe.")
            return
        signal, lev, strength, comment, volume_spike, atr_comment, candle_pattern = analyse(df)
        current_price = df['close'].iloc[-1]

        # Display summary
        st.markdown(f"**Signal:** {signal} | **Leverage:** x{lev} | **Strength:** {strength:.2f} / 10", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{TEXT_MUTED};'>{comment}</p>", unsafe_allow_html=True)

        # Volume / Volatility / Pattern explanations
        explanations = []

        if volume_spike:
            explanations.append("📈 <b>Volume Spike detected</b> – sudden increase in trading activity.")

        # Clean ATR comment (strip emoji/symbols)
        atr_clean = atr_comment.replace("▲", "").replace("▼", "").replace("–", "").strip()
        if atr_clean == "Moderate":
            explanations.append("🔄 <b>Volatility is moderate</b> – stable price conditions.")
        elif atr_clean == "High":
            explanations.append("⚠️ <b>Volatility is high</b> – expect sharp moves.")
        elif atr_clean == "Low":
            explanations.append("🟢 <b>Volatility is low</b> – steady market behaviour.")

        if candle_pattern:
            explanations.append(f"🕯️ <b>Candle pattern:</b> {candle_pattern}")

        if explanations:
            st.markdown("<br/>".join(explanations), unsafe_allow_html=True)

        # Price box
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Current Price</div><div class='metric-value'>${current_price:,.2f}</div></div>", unsafe_allow_html=True)


        sent_score, sent_label = get_social_sentiment(coin)
        gauge_sent = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sent_score,
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': TEXT_MUTED},
                'bar': {'color': ACCENT},
                'bgcolor': CARD_BG,
                'steps': [
                    {'range': [0, 25], 'color': NEGATIVE},
                    {'range': [25, 45], 'color': WARNING},
                    {'range': [45, 55], 'color': TEXT_MUTED},
                    {'range': [55, 75], 'color': POSITIVE},
                    {'range': [75, 100], 'color': POSITIVE},
                ],
            },
            title={'text': f"Sentiment ({sent_label})", 'font': {'size': 16, 'color': ACCENT}},
            number={'font': {'color': TEXT_LIGHT, 'size': 36}}
        ))
        gauge_sent.update_layout(
            height=170,
            margin=dict(l=10, r=10, t=40, b=15),
            template='plotly_dark',
            paper_bgcolor=CARD_BG
        )
        st.plotly_chart(gauge_sent, use_container_width=True)
        # Plot candlestick with EMAs
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE, name="Price"
        ))
        for window, colour in [(5, '#F472B6'), (9, '#60A5FA'), (21, '#FBBF24'), (50, '#FCD34D')]:
            ema_series = ta.trend.ema_indicator(df['close'], window=window)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=ema_series, mode='lines',
                                     name=f"EMA{window}", line=dict(color=colour, width=1.5)))
        # Place legend at top left for candlestick chart
        fig.update_layout(
            height=380,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        # RSI chart
        rsi_fig = go.Figure()
        for period, colour in [(6, '#D8B4FE'), (14, '#A78BFA'), (24, '#818CF8')]:
            rsi_series = ta.momentum.rsi(df['close'], window=period)
            rsi_fig.add_trace(go.Scatter(
                x=df['timestamp'], y=rsi_series, mode='lines', name=f"RSI {period}",
                line=dict(color=colour, width=2)
            ))
        # Add overbought/oversold bands
        rsi_fig.add_hline(y=70, line=dict(color=NEGATIVE, dash='dot', width=1), name="Overbought")
        rsi_fig.add_hline(y=30, line=dict(color=POSITIVE, dash='dot', width=1), name="Oversold")
        rsi_fig.update_layout(
            height=180,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=30),
            yaxis=dict(title="RSI"),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(rsi_fig, use_container_width=True)

        # MACD chart
        macd_ind = ta.trend.MACD(df['close'])
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_diff'] = macd_ind.macd_diff()
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd'], name="MACD",
            line=dict(color=ACCENT, width=2)
        ))
        macd_fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd_signal'], name="Signal",
            line=dict(color=WARNING, width=2, dash='dot')
        ))
        macd_fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['macd_diff'], name="Histogram",
            marker_color=CARD_BG
        ))
        macd_fig.update_layout(
            height=200,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=30),
            yaxis=dict(title="MACD"),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(macd_fig, use_container_width=True)

        # Volume & OBV chart
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['volume'], name="Volume", marker_color="#6B7280"
        ))
        volume_fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['obv'], name="OBV",
            line=dict(color=WARNING, width=1.5, dash='dot'),
            yaxis='y2'
        ))
        volume_fig.update_layout(
            height=180,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=20, b=30),
            yaxis=dict(title="Volume"),
            yaxis2=dict(overlaying='y', side='right', title='OBV', showgrid=False),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )
        st.plotly_chart(volume_fig, use_container_width=True)

        # Technical snapshot
        # Compute indicators for snapshot
        df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
        latest = df.iloc[-1]
        ema9 = latest['ema9']
        ema21 = latest['ema21']
        macd_val = df['macd'].iloc[-1]
        rsi_val = latest['rsi14']
        obv_change = ((df['obv'].iloc[-1] - df['obv'].iloc[-5]) / abs(df['obv'].iloc[-5]) * 100) if df['obv'].iloc[-5] != 0 else 0
        recent = df.tail(20)
        support = recent['low'].min()
        resistance = recent['high'].max()
        current_price = latest['close']
        support_dist = abs(current_price - support) / current_price * 100
        resistance_dist = abs(current_price - resistance) / current_price * 100
        # Build snapshot HTML
        snapshot_html = f"""
        <div class='panel-box'>
          <b style='color:{ACCENT}; font-size:1.05rem;'>📊 Technical Snapshot</b><br>
          <ul style='color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.5; list-style-position:inside; margin-top:6px;'>
            <li>EMA Trend (9 vs 21): <b>{ema9:.2f}</b> vs <b>{ema21:.2f}</b> {('🟢' if ema9 > ema21 else '🔴')} — When EMA9 is above EMA21 the short‑term trend is bullish; otherwise bearish.</li>
            <li>MACD: <b>{macd_val:.2f}</b> {('🟢' if macd_val > 0 else '🔴')} — Positive MACD indicates upward momentum; negative values suggest downward pressure.</li>
            <li>RSI (14): <b>{rsi_val:.2f}</b> {('🟢' if rsi_val > 55 else ('🟠' if 45 <= rsi_val <= 55 else '🔴'))} — Above 70 may signal overbought, below 30 oversold. Values above 50 favour bulls.</li>
            <li>OBV change (last 5 candles): <b>{obv_change:+.2f}%</b> {('🟢' if obv_change > 0 else '🔴')} — Rising OBV supports the price move; falling OBV warns against continuation.</li>
            <li>Support / Resistance: support at <b>${support:,.2f}</b> ({support_dist:.2f}% away), resistance at <b>${resistance:,.2f}</b> ({resistance_dist:.2f}% away).</li>
          </ul>
        </div>
        """
        st.markdown(snapshot_html, unsafe_allow_html=True)


def render_position_tab():
    """Render the Position Analyser tab for evaluating open positions."""
    st.markdown(
        f"<h2 style='color:{ACCENT};margin-bottom:0.5rem;'>Position Analyser</h2>",
        unsafe_allow_html=True,
    )
    coin = st.text_input("Coin Symbol (e.g. BTC/USDT)", value="BTC/USDT").upper()
    timeframe = st.selectbox("Timeframe", ['1m', '3m', '5m', '15m', '1h', '4h', '1d'], index=2)
    # Prefill the entry price with the current market price to ease input.  We fetch
    # a single candle to determine the latest close for the selected symbol and timeframe.
    default_entry_price: float = 0.0
    try:
        df_current = fetch_ohlcv(coin, timeframe, limit=1)
        if df_current is not None and len(df_current) > 0:
            default_entry_price = float(df_current['close'].iloc[-1])
    except Exception:
        default_entry_price = 0.0
    entry_price = st.number_input(
        "Entry Price", min_value=0.0, format="%.4f", value=default_entry_price
    )
    direction = st.selectbox("Position Direction", ["LONG", "SHORT"])
    if st.button("Analyse Position", type="primary"):
        df = fetch_ohlcv(coin, timeframe, limit=100)
        if df is None or len(df) < 30:
            st.error("Not enough data to analyse position.")
            return
        signal, lev, strength, comment, volume_spike, atr_comment, candle_pattern = analyse(df)
        current_price = df['close'].iloc[-1]
        # Compute PnL
        pnl = current_price - entry_price
        pnl_percent = (pnl / entry_price * 100) if entry_price else 0
        # Determine colour and icon
        col = POSITIVE if pnl_percent > 0 else (WARNING if abs(pnl_percent) < 1 else NEGATIVE)
        icon = '🟢' if pnl_percent > 0 else ('🟠' if abs(pnl_percent) < 1 else '🔴')
        st.markdown(
            f"<div class='panel-box' style='background-color:{col};color:{PRIMARY_BG};'>"
            f"  {icon} <strong>{direction} Position</strong><br>"
            f"  Entry: ${entry_price:,.4f} | Current: ${current_price:,.4f} ({pnl_percent:+.2f}%)"
            f"</div>",
            unsafe_allow_html=True,
        )
        # Show strength and leverage badges
        st.markdown(f"**Signal:** {signal} | **Leverage:** x{lev} | **Strength:** {strength:.2f} / 10", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{TEXT_MUTED};'>{comment}</p>", unsafe_allow_html=True)

        # === Extra Commentary ===
        explanations = []
        
        if volume_spike:
            explanations.append("📈 <b>Volume Spike detected</b> – sudden increase in trading activity.")
        
        atr_clean = atr_comment.replace("▲", "").replace("▼", "").replace("–", "").strip()
        if atr_clean == "Moderate":
            explanations.append("🔄 <b>Volatility is moderate</b> – stable price conditions.")
        elif atr_clean == "High":
            explanations.append("⚠️ <b>Volatility is high</b> – expect sharp moves.")
        elif atr_clean == "Low":
            explanations.append("🟢 <b>Volatility is low</b> – steady market behaviour.")
        
        if candle_pattern:
            explanations.append(f"🕯️ <b>Candle pattern:</b> {candle_pattern}")
        
        if explanations:
            st.markdown("<br/>".join(explanations), unsafe_allow_html=True)

        # ===== Prepare technical indicators for the current position =====

        df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
        df['ema13'] = ta.trend.ema_indicator(df['close'], window=13)
        df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
        # MACD and RSI across multiple periods
        macd_ind = ta.trend.MACD(df['close'])
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_diff'] = macd_ind.macd_diff()
        df['rsi6'] = ta.momentum.rsi(df['close'], window=6)
        df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi24'] = ta.momentum.rsi(df['close'], window=24)
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

        # === Strategy Suggestion (displayed before charts) ===
        # Determine suggestion based on price relative to support/resistance and entry
        suggestion = ""
        # Support and resistance computed from recent candles for suggestion purposes
        recent_sr = df.tail(20)
        support_sr = recent_sr['low'].min()
        resistance_sr = recent_sr['high'].max()
        if direction == "LONG":
            if current_price < support_sr:
                suggestion = (f"🔻 Price broke below recent support at <b>${support_sr:,.4f}</b>.<br>"
                              f"<b>Consider closing the position (STOP).</b>")
            elif current_price < entry_price:
                suggestion = (f"⚠️ Price is under entry.<br>"
                              f"Watch support at <b>${support_sr:,.4f}</b>. If it breaks, stopping out may be wise.<br>"
                              f"Price is also near resistance at <b>${resistance_sr:,.4f}</b>. A breakout could signal further upside.")
            else:
                if current_price < resistance_sr:
                    suggestion = (f"📈 Price above entry.<br>"
                                  f"If <b>${resistance_sr:,.4f}</b> breaks, consider taking profit.")
                else:
                    suggestion = (f"🟢 Price broke above resistance.<br>"
                                  f"Consider holding or taking partial profits.")
        else:  # SHORT
            if current_price > resistance_sr:
                suggestion = (f"🔺 Price broke above resistance at <b>${resistance_sr:,.4f}</b>.<br>"
                              f"<b>Consider closing the position (STOP).</b>")
            elif current_price > entry_price:
                suggestion = (f"⚠️ Price is above entry.<br>"
                              f"Watch resistance at <b>${resistance_sr:,.4f}</b>. If it breaks, stopping may be necessary.<br>"
                              f"Price is also near support at <b>${support_sr:,.4f}</b>. Breakdown would support a SHORT.")
            else:
                if current_price > support_sr:
                    suggestion = (f"📉 Price below entry.<br>"
                                  f"If <b>${support_sr:,.4f}</b> breaks down, consider taking profit.")
                else:
                    suggestion = (f"🟢 Price broke below support.<br>"
                                  f"Consider holding or taking partial profits.")
        st.markdown(
            f"<div class='panel-box'>"
            f"  <b style='color:{ACCENT}; font-size:1.05rem;'>🧠 Strategy Suggestion</b><br>"
            f"  <p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:6px;'>{suggestion}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        # === Primary Candlestick Chart ===
        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE, name="Price"
        ))
        for window, colour in [(5, '#F472B6'), (9, '#60A5FA'), (13, '#A78BFA'), (21, '#FBBF24'), (50, '#FCD34D')]:
            ema_series = ta.trend.ema_indicator(df['close'], window=window)
            fig_candle.add_trace(go.Scatter(x=df['timestamp'], y=ema_series, mode='lines', name=f"EMA{window}", line=dict(color=colour, width=1.5)))
        fig_candle.update_layout(
            height=380,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )

        # === Scalping Opportunity ===
        df_scalp = fetch_ohlcv(coin, timeframe, limit=100)
        if df_scalp is not None and len(df_scalp) > 30:
            df_scalp['ema_fast'] = df_scalp['close'].ewm(span=5).mean()
            df_scalp['ema_slow'] = df_scalp['close'].ewm(span=13).mean()
            df_scalp['ema9'] = df_scalp['close'].ewm(span=9).mean()
            df_scalp['ema21'] = df_scalp['close'].ewm(span=21).mean()
            # MACD and indicators for scalping
            macd_s = ta.trend.MACD(df_scalp['close'])
            df_scalp['macd'] = macd_s.macd()
            df_scalp['macd_signal'] = macd_s.macd_signal()
            df_scalp['macd_diff'] = df_scalp['macd'] - df_scalp['macd_signal']
            df_scalp['rsi6'] = ta.momentum.rsi(df_scalp['close'], window=6)
            df_scalp['obv'] = ta.volume.on_balance_volume(df_scalp['close'], df_scalp['volume'])
            # Direction and colours
            fast = df_scalp['ema_fast'].iloc[-1]
            slow = df_scalp['ema_slow'].iloc[-1]
            scalp_direction = "LONG" if fast > slow else "SHORT"
            colour = POSITIVE if scalp_direction == "LONG" else NEGATIVE
            icon = "🟢" if scalp_direction == "LONG" else "🔴"
            support_s = df_scalp['low'].tail(20).min()
            resistance_s = df_scalp['high'].tail(20).max()
            if scalp_direction == "LONG":

                entry_s = support_s * 1.005
                stop_s = df_scalp['low'].rolling(5).min().iloc[-1]

                target_s = resistance_s
            else:

                entry_s = resistance_s * 0.995
                stop_s = df_scalp['high'].rolling(5).max().iloc[-1]

                target_s = support_s

            st.markdown(
                f"<div class='panel-box' style='background-color:{colour};color:{PRIMARY_BG};'>"
                f"  {icon} <b>Scalping {scalp_direction}</b><br>"
                f"  Entry: <b>${entry_s:,.4f}</b><br>"
                f"  Stop Loss: <b>${stop_s:,.4f}</b><br>"
                f"  Target: <b>${target_s:,.4f}</b>"
                f"</div>",
                unsafe_allow_html=True
            )
            # Immediately display the scalping technical snapshot
            latest_s = df_scalp.iloc[-1]
            ema5_val = df_scalp['ema_fast'].iloc[-1]
            ema13_val = df_scalp['ema_slow'].iloc[-1]
            macd_hist_s = df_scalp['macd_diff'].iloc[-1]
            rsi6_val = df_scalp['rsi6'].iloc[-1]
            obv_change_s = ((df_scalp['obv'].iloc[-1] - df_scalp['obv'].iloc[-3]) / abs(df_scalp['obv'].iloc[-3]) * 100) if df_scalp['obv'].iloc[-3] != 0 else 0
            support_dist_s = abs(latest_s['close'] - support_s) / latest_s['close'] * 100
            resistance_dist_s = abs(latest_s['close'] - resistance_s) / latest_s['close'] * 100
            # Build and display the scalping snapshot
            scalping_snapshot_html = f"""
            <div class='panel-box'>
              <b style='color:{ACCENT}; font-size:1.05rem;'>📊 Technical Snapshot (Scalping)</b><br>
              <ul style='color:{TEXT_MUTED}; font-size:0.9rem; line-height:1.5; list-style-position:inside; margin-top:6px;'>
                <li>EMA Trend (5 vs 13): <b>{ema5_val:,.2f}</b> vs <b>{ema13_val:,.2f}</b> {('🟢' if ema5_val > ema13_val else '🔴')} — When EMA5 crosses above EMA13, momentum shifts bullish; below, bearish.</li>
                <li>MACD Histogram: <b>{macd_hist_s:.2f}</b> {('🟢' if macd_hist_s > 0 else '🔴')} — Positive values indicate strengthening momentum; negative values suggest weakening.</li>
                <li>RSI (6): <b>{rsi6_val:.2f}</b> {('🟢' if rsi6_val > 50 else '🔴')} — Short‑term RSI above 70 may be overbought, below 30 oversold.</li>
                <li>OBV Change (last 3 candles): <b>{obv_change_s:+.2f}%</b> {('🟢' if obv_change_s > 0 else '🔴')} — Rising OBV lends support to the trend.</li>
                <li>Support / Resistance: support at <b>${support_s:,.4f}</b> ({support_dist_s:.2f}% away), resistance at <b>${resistance_s:,.4f}</b> ({resistance_dist_s:.2f}% away).</li>
              </ul>
            </div>
            """
            st.markdown(scalping_snapshot_html, unsafe_allow_html=True)


        else:
            latest_s = None  # no scalping data
            ema5_val = ema13_val = macd_hist_s = rsi6_val = obv_change_s = support_dist_s = resistance_dist_s = None

        st.plotly_chart(fig_candle, use_container_width=True)

def render_guide_tab():
    """Render an Analysis Guide explaining the calculations used in the dashboard."""

    st.markdown(
        f"<h2 style='color:{ACCENT}; font-size:1.6rem; margin-bottom:1rem;'>Analysis Guide</h2>",
        unsafe_allow_html=True,
    )

    signal_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>Signal &amp; Strength</b>
      <ul style='color:{TEXT_MUTED}; font-size:0.92rem; line-height:1.6; margin-top:0.5rem;'>
        <li><span style='color:{ACCENT};'>EMA Trend:</span> A bullish structure (e.g. EMA5 &gt; EMA9 &gt; EMA21 &gt; EMA50) gives +2 strength points if at least two relationships are true.</li>
        <li><span style='color:{ACCENT};'>RSI:</span> RSI14 above 60 adds +2 points; between 55–60 adds +1 point (momentum improving).</li>
        <li><span style='color:{ACCENT};'>MACD:</span> MACD line above signal and positive histogram adds +1 point (momentum confirmation).</li>
        <li><span style='color:{ACCENT};'>OBV:</span> A rising On‑Balance Volume over the last five candles adds +1 strength point.</li>
        <li><span style='color:{ACCENT};'>Volume Spike:</span> If current volume &gt; 2× recent average, +1 point is added.</li>
        <li><span style='color:{ACCENT};'>Volatility (ATR):</span> Moderate ATR adds +1 point, high ATR adds +0.5, low adds 0.</li>
        <li><span style='color:{ACCENT};'>Candle Pattern:</span> If a bullish pattern (e.g. Bullish Engulfing or Hammer) aligns with EMA trend, +2 points.</li>
      </ul>
      <p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:0.5rem;'>All strength points are capped at 10 and define signal type and leverage limits.</p>
    </div>
    """

    confidence_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>Confidence Levels</b>
      <p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:0.5rem;'>Confidence reflects how reliable a signal is, based on strength score:</p>
      <ul style='color:{TEXT_MUTED}; font-size:0.92rem; line-height:1.6;'>
        <li><span style='color:{ACCENT};'>MAXIMUM:</span> Strength ≥ 9.5</li>
        <li><span style='color:{ACCENT};'>HIGH:</span> Strength ≥ 7</li>
        <li><span style='color:{ACCENT};'>MEDIUM:</span> Strength ≥ 4</li>
        <li><span style='color:{ACCENT};'>LOW:</span> Strength &lt; 4</li>
      </ul>
    </div>
    """

    risk_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>Risk Score &amp; Leverage</b>
      <p style='color:{TEXT_MUTED}; font-size:0.9rem; margin-top:0.5rem;'>Risk is evaluated using:</p>
      <ul style='color:{TEXT_MUTED}; font-size:0.92rem; line-height:1.6;'>
        <li><span style='color:{ACCENT};'>Bollinger Band Width:</span> Wider bands indicate high volatility (adds up to +0.1).</li>
        <li><span style='color:{ACCENT};'>RSI Extremes:</span> RSI above 70 or below 30 adds +0.1 risk.</li>
        <li><span style='color:{ACCENT};'>OBV Change:</span> Rising OBV <b>only</b> when price is above EMA21 adds +0.1 to risk.</li>
        <li><span style='color:{ACCENT};'>Support/Resistance Proximity:</span> If price is within 2% of support or resistance, +0.1 risk is added.</li>
      </ul>
      <p style='color:{TEXT_MUTED}; font-size:0.9rem;'>
        Risk &lt; 0.15 → <span style='{POSITIVE};'>3–7×</span> leverage<br/>
        0.15–0.25 → <span style='{WARNING};'>8–12×</span> leverage<br/>
        ≥0.25 → <span style='{NEGATIVE};'>13–20×</span> leverage
      </p>
      <p style='color:{TEXT_MUTED}; font-size:0.9rem;'>Leverage is capped at 4× if strength &lt; 4 and at 8× if strength &lt; 6. Final tiers are:<br/>
      <b>Low Risk:</b> lev_base<br/>
      <b>Medium Risk:</b> lev_base + 3 (max 14)<br/>
      <b>High Risk:</b> lev_base + 6 (max 20)</p>
    </div>
    """


    position_html = f"""
    <div class='panel-box'>
      <b style='color:{ACCENT}; font-size:1.2rem;'>Position &amp; Scalping Analysis</b>
      <p style='color:{TEXT_MUTED}; font-size:0.92rem; margin-top:0.5rem;'>In the <b>Position Analyser</b>, enter a symbol, timeframe and your entry price. The dashboard recalculates signal and strength, shows your unrealised PnL, and provides strategy suggestions based on support/resistance breakout or breakdown.</p>
      <p style='color:{TEXT_MUTED}; font-size:0.92rem;'>For <b>scalping</b>, LONG trades use entry = support × 1.005 and SHORT = resistance × 0.995. Stop = recent extreme (lowest/highest of last 5 candles), and Target = opposite SR level. This approach keeps logic simple and fast for intraday entries.</p>
    </div>
    """

    st.markdown(signal_html, unsafe_allow_html=True)
    st.markdown(confidence_html, unsafe_allow_html=True)
    st.markdown(risk_html, unsafe_allow_html=True)
    st.markdown(position_html, unsafe_allow_html=True)


def main():
    """Entry point for the Streamlit app."""
    # Create tabs for different panels
    tabs = st.tabs(["Market", "Spot", "Position", "Analysis Guide"])
    with tabs[0]:
        render_market_tab()
    with tabs[1]:
        render_spot_tab()
    with tabs[2]:
        render_position_tab()
    with tabs[3]:
        render_guide_tab()


if __name__ == "__main__":
    main()