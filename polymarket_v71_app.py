
import json
import math
import time
from datetime import datetime, timezone
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# POLYMARKET MOBILE AUTO PAPER TRADER V7.1
# This is the full app. If your GitHub app.py does not say V7,
# you are still running old code.
# ============================================================

GAMMA_API = "https://gamma-api.polymarket.com"
DEFAULT_BANKROLL = 100.00

st.set_page_config(
    page_title="V7.1 Mobile Auto Paper Trader",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 0.75rem;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
    max-width: 760px;
}
div.stButton > button {
    width: 100%;
    min-height: 3.2rem;
    border-radius: 16px;
    font-size: 1.05rem;
    font-weight: 800;
}
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 14px;
    margin-bottom: 13px;
}
.title {
    font-size: 1rem;
    font-weight: 850;
    line-height: 1.25;
    margin-top: 7px;
    margin-bottom: 7px;
}
.small {
    font-size: 0.83rem;
    opacity: 0.82;
    line-height: 1.35;
}
.pill {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    margin-right: 5px;
    margin-bottom: 5px;
    font-size: 0.76rem;
    font-weight: 800;
}
.green { background: #dcfce7; color: #166534; }
.yellow { background: #fef3c7; color: #854d0e; }
.red { background: #fee2e2; color: #991b1b; }
.blue { background: #dbeafe; color: #1e40af; }
.gray { background: #e5e7eb; color: #374151; }
.money {
    font-size: 1.05rem;
    font-weight: 900;
    margin-top: 6px;
}
a { text-decoration: none; font-weight: 800; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# State
# ----------------------------
def init_state():
    defaults = {
        "cash": DEFAULT_BANKROLL,
        "initial_bankroll": DEFAULT_BANKROLL,
        "trades": [],
        "previous_prices": {},
        "last_candidates": [],
        "scan_count": 0,
        "last_scan": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ----------------------------
# API and parsing
# ----------------------------
def safe_get(url, params=None, retries=2):
    last_error = None
    for _ in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_error = e
            time.sleep(0.5)
    raise last_error

@st.cache_data(ttl=90)
def get_markets(limit):
    data = safe_get(
        f"{GAMMA_API}/markets",
        {
            "active": "true",
            "closed": "false",
            "limit": int(limit),
            "order": "volume",
            "ascending": "false",
        },
    )
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame(data.get("markets", data.get("data", [])))

def parse_jsonish(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return [x.strip() for x in value.split(",") if x.strip()]
    return []

def fnum(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default

def get_prices(row):
    prices = parse_jsonish(row.get("outcomePrices", row.get("outcome_prices", None)))
    out = []
    for p in prices:
        try:
            out.append(float(p))
        except Exception:
            pass
    return out

def get_outcomes(row):
    return parse_jsonish(row.get("outcomes", None))

def market_url(slug):
    return f"https://polymarket.com/market/{slug}" if slug else "https://polymarket.com/markets"

def news_url(question):
    return "https://www.google.com/search?tbm=nws&q=" + quote_plus(str(question))

def days_until(end_date):
    try:
        if not end_date:
            return None
        dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
        return (dt - datetime.now(timezone.utc)).days
    except Exception:
        return None

# ----------------------------
# Strategy
# ----------------------------
def focus_match(question, focus):
    q = str(question).lower()
    if focus == "All":
        return True
    buckets = {
        "Politics / News": ["trump", "biden", "iran", "israel", "china", "russia", "ukraine", "election", "president", "leader", "congress", "senate", "war", "ceasefire", "tariff", "fed", "court"],
        "Sports": ["ufc", "nba", "nfl", "mlb", "nhl", "soccer", "fight", "match", "championship", "wins", "score", "goal"],
        "Business / AI": ["openai", "anthropic", "cursor", "anysphere", "ipo", "valuation", "apple", "tesla", "nvidia", "google", "meta", "microsoft", "ai", "company"],
        "Weather / Science": ["weather", "temperature", "hurricane", "storm", "rain", "snow", "earthquake", "climate", "science", "nasa"],
    }
    return any(w in q for w in buckets.get(focus, []))

def risk_flags(question, price, volume, volume_24h, liquidity, spread_proxy, end_date):
    q = str(question).lower()
    flags = []

    if liquidity < 2500:
        flags.append("Low liquidity")
    if volume < 5000:
        flags.append("Low volume")
    if spread_proxy > 0.18:
        flags.append("Wide pricing")
    if price < 0.06 or price > 0.94:
        flags.append("Extreme price")
    if volume_24h > 0 and volume_24h < 1000:
        flags.append("Weak 24h activity")
    if any(w in q for w in ["fdv", "meme", "coin", "airdrop", "token", "launch"]):
        flags.append("Crypto launch risk")
    if any(w in q for w in ["ufc", "nba", "nfl", "mlb", "nhl", "fight", "match"]):
        flags.append("Sports variance")
    if len(q) > 95:
        flags.append("Complex wording")
    if any(w in q for w in ["resign", "out as", "ceo", "leader", "president", "minister"]):
        flags.append("Political/personnel uncertainty")
    if "before 2027" in q or "by december 31" in q:
        flags.append("Long time horizon")

    d = days_until(end_date)
    if d is not None:
        if d <= 2:
            flags.append("Near expiration")
        elif d > 180:
            flags.append("Long dated")

    return flags

def catalyst_type(question):
    q = str(question).lower()
    if any(w in q for w in ["iran", "israel", "ukraine", "russia", "china", "trump", "election", "tariff", "fed", "court", "ceasefire"]):
        return "News"
    if any(w in q for w in ["ufc", "nba", "nfl", "mlb", "nhl", "fight", "match"]):
        return "Sports"
    if any(w in q for w in ["fdv", "token", "coin", "airdrop", "launch", "btc", "eth", "sol"]):
        return "Crypto"
    if any(w in q for w in ["ipo", "valuation", "openai", "anthropic", "cursor", "anysphere", "nvidia", "tesla"]):
        return "Business/AI"
    return "General"

def confidence(edge_score, risk_count, liquidity, volume, price):
    if risk_count >= 4:
        return "Low"
    if edge_score >= 78 and liquidity >= 15000 and volume >= 25000 and 0.12 <= price <= 0.88 and risk_count <= 1:
        return "High"
    if edge_score >= 64 and liquidity >= 5000 and volume >= 10000 and risk_count <= 3:
        return "Medium"
    return "Low"

def action_label(conf, edge_score, flags):
    hard_avoid = {"Low liquidity", "Wide pricing", "Complex wording", "Crypto launch risk"}
    if any(f in hard_avoid for f in flags):
        return "Research Only"
    if "Sports variance" in flags:
        return "Watch" if conf in ["High", "Medium"] and edge_score >= 64 else "Research Only"
    if conf == "High" and edge_score >= 78 and len(flags) <= 1:
        return "Possible Bet"
    if conf in ["High", "Medium"] and edge_score >= 64:
        return "Watch"
    return "Research Only"

def stake_size(cash, max_bet_pct, conf, edge_score, flags):
    if cash <= 0.25 or conf == "Low" or len(flags) >= 4:
        return 0.0

    base = cash * max_bet_pct
    if conf == "High":
        mult = min(1.0, max(0.30, (edge_score - 62) / 26))
    elif conf == "Medium":
        mult = min(0.45, max(0.10, (edge_score - 55) / 42))
    else:
        mult = 0

    penalties = {
        "Sports variance": 0.70,
        "Political/personnel uncertainty": 0.75,
        "Near expiration": 0.70,
        "Long dated": 0.70,
        "Weak 24h activity": 0.80,
    }
    for f, p in penalties.items():
        if f in flags:
            mult *= p

    raw = base * mult
    if raw <= 0:
        return 0.0
    return round(min(cash, max(0.25, raw)), 2)

def score_market(row, settings):
    question = row.get("question", row.get("title", ""))
    if not focus_match(question, settings["focus"]):
        return None

    slug = row.get("slug", "")
    key = slug or question
    end_date = row.get("endDate", row.get("end_date", ""))

    volume = fnum(row.get("volume", row.get("volumeNum", 0)))
    volume_24h = fnum(row.get("volume24hr", row.get("volume24hrClob", 0)))
    liquidity = fnum(row.get("liquidity", row.get("liquidityNum", 0)))

    prices = get_prices(row)
    outcomes = get_outcomes(row)
    if not prices:
        return None

    best_i = int(np.nanargmax(prices))
    price = float(prices[best_i])
    outcome = outcomes[best_i] if best_i < len(outcomes) else f"Outcome {best_i + 1}"

    if price < settings["min_price"] or price > settings["max_price"]:
        return None
    if volume < settings["min_volume"] or liquidity < settings["min_liquidity"]:
        return None

    prev = st.session_state.previous_prices.get(f"{key}||{outcome}")
    movement = 0.0 if prev is None else price - float(prev)

    spread_proxy = abs(sum(prices[:2]) - 1) if len(prices) >= 2 else 0.12

    vol_score = min(1, math.log10(max(volume, 1)) / 7)
    liq_score = min(1, math.log10(max(liquidity, 1)) / 6)
    price_score = max(0, min(1, 1 - abs(price - 0.5) * 1.35))
    v24_score = min(1, math.log10(max(volume_24h, 1)) / 6) if volume_24h else vol_score * 0.65
    spread_score = 1 - min(1, spread_proxy * 4)
    movement_score = min(1, abs(movement) / 0.08)

    cat = catalyst_type(question)
    flags = risk_flags(question, price, volume, volume_24h, liquidity, spread_proxy, end_date)

    edge_score = 100 * (
        0.25 * vol_score
        + 0.25 * liq_score
        + 0.17 * price_score
        + 0.13 * v24_score
        + 0.08 * spread_score
        + 0.07 * movement_score
        + (0.05 if cat != "General" else 0)
    )
    edge_score -= 6.5 * len(flags)
    edge_score = max(0, min(100, edge_score))

    conf = confidence(edge_score, len(flags), liquidity, volume, price)
    action = action_label(conf, edge_score, flags)
    stake = stake_size(st.session_state.cash, settings["max_bet_pct"], conf, edge_score, flags)

    reasons = []
    if liquidity >= 15000:
        reasons.append("good liquidity")
    if volume >= 25000:
        reasons.append("strong volume")
    if volume_24h >= 5000:
        reasons.append("recent activity")
    if abs(movement) >= 0.03:
        reasons.append(f"moved {movement:+.1%}")
    if cat != "General":
        reasons.append(f"{cat} catalyst")
    if flags:
        reasons.append("risks: " + ", ".join(flags[:2]))

    return {
        "Action": action,
        "Confidence": conf,
        "Edge Score": round(edge_score, 1),
        "Market Price": round(price, 3),
        "Change": round(movement, 3),
        "Suggested Stake": stake,
        "Question": question,
        "Outcome": outcome,
        "Catalyst": cat,
        "Liquidity": round(liquidity, 0),
        "Volume": round(volume, 0),
        "24h Volume": round(volume_24h, 0),
        "Risk Flags": ", ".join(flags) if flags else "None",
        "Why": "; ".join(reasons) if reasons else "passed base filters",
        "End Date": end_date,
        "Open Market": market_url(slug),
        "News Search": news_url(question),
        "_market_key": key,
        "_trade_key": f"{key}||{outcome}",
        "_price": price,
    }

def build_price_lookup(markets):
    lookup = {}
    for _, row in markets.iterrows():
        slug = row.get("slug", "")
        market_key = slug or row.get("question", row.get("title", ""))
        prices = get_prices(row)
        outcomes = get_outcomes(row)
        for i, price in enumerate(prices):
            outcome = outcomes[i] if i < len(outcomes) else f"Outcome {i + 1}"
            lookup[f"{market_key}||{outcome}"] = float(price)
    return lookup

def open_trade(candidate):
    stake = min(float(candidate["Suggested Stake"]), float(st.session_state.cash))
    if stake <= 0 or candidate["Action"] == "Research Only":
        return False

    price = float(candidate["Market Price"])
    trade = {
        "id": len(st.session_state.trades) + 1,
        "status": "OPEN",
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "closed_at": "",
        "question": candidate["Question"],
        "outcome": candidate["Outcome"],
        "trade_key": candidate["_trade_key"],
        "entry_price": price,
        "last_price": price,
        "exit_price": None,
        "stake": round(stake, 2),
        "shares": round(stake / price, 6),
        "unrealized_pnl": 0.0,
        "pnl": 0.0,
        "close_reason": "",
        "action": candidate["Action"],
        "confidence": candidate["Confidence"],
        "edge_score": candidate["Edge Score"],
        "open_market": candidate["Open Market"],
        "news_search": candidate["News Search"],
    }
    st.session_state.cash -= stake
    st.session_state.trades.append(trade)
    return True

def close_trade(trade, exit_price, reason):
    value = float(trade["shares"]) * float(exit_price)
    pnl = value - float(trade["stake"])
    trade["status"] = "CLOSED"
    trade["closed_at"] = datetime.now(timezone.utc).isoformat()
    trade["exit_price"] = round(float(exit_price), 4)
    trade["last_price"] = round(float(exit_price), 4)
    trade["pnl"] = round(pnl, 2)
    trade["unrealized_pnl"] = 0.0
    trade["close_reason"] = reason
    st.session_state.cash += value

def update_positions(price_lookup, settings):
    for trade in st.session_state.trades:
        if trade["status"] != "OPEN":
            continue
        current = price_lookup.get(trade["trade_key"])
        if current is None:
            continue

        entry = float(trade["entry_price"])
        current = float(current)
        trade["last_price"] = round(current, 4)
        trade["unrealized_pnl"] = round(float(trade["shares"]) * current - float(trade["stake"]), 2)

        pct = (current - entry) / entry if entry else 0
        opened = datetime.fromisoformat(trade["opened_at"])
        age_days = (datetime.now(timezone.utc) - opened).total_seconds() / 86400

        if pct >= settings["take_profit"]:
            close_trade(trade, current, f"Take profit +{settings['take_profit']:.0%}")
        elif pct <= -settings["stop_loss"]:
            close_trade(trade, current, f"Stop loss -{settings['stop_loss']:.0%}")
        elif age_days >= settings["max_hold_days"]:
            close_trade(trade, current, f"Max hold {settings['max_hold_days']}d")

def already_open(trade_key):
    return any(t["status"] == "OPEN" and t["trade_key"] == trade_key for t in st.session_state.trades)

def auto_enter(candidates, settings):
    opened = 0
    open_count = sum(1 for t in st.session_state.trades if t["status"] == "OPEN")
    allowed = ["Possible Bet"] if settings["entry_quality"] == "Possible Bet only" else ["Possible Bet", "Watch"]

    for c in candidates:
        if open_count >= settings["max_open"]:
            break
        if c["Action"] not in allowed:
            continue
        if already_open(c["_trade_key"]):
            continue
        if c["Suggested Stake"] <= 0:
            continue
        if st.session_state.cash < 0.25:
            break
        if open_trade(c):
            opened += 1
            open_count += 1
    return opened

def run_cycle(settings):
    markets = get_markets(settings["market_limit"])
    if markets.empty:
        return pd.DataFrame(), 0, "No markets returned"

    price_lookup = build_price_lookup(markets)
    update_positions(price_lookup, settings)

    rows = []
    latest = {}
    for _, row in markets.iterrows():
        scored = score_market(row, settings)
        if scored:
            rows.append(scored)
            latest[scored["_trade_key"]] = scored["_price"]

    if latest:
        st.session_state.previous_prices = latest

    if not rows:
        return pd.DataFrame(), 0, "No candidates passed filters"

    candidates = pd.DataFrame(rows)
    action_order = {"Possible Bet": 0, "Watch": 1, "Research Only": 2}
    conf_order = {"High": 0, "Medium": 1, "Low": 2}
    candidates["_action_sort"] = candidates["Action"].map(action_order)
    candidates["_conf_sort"] = candidates["Confidence"].map(conf_order)
    candidates = candidates.sort_values(
        ["_action_sort", "_conf_sort", "Edge Score", "Liquidity"],
        ascending=[True, True, False, False],
    )

    opened = auto_enter(candidates.to_dict("records"), settings) if settings["auto_mode"] else 0
    st.session_state.last_candidates = candidates.to_dict("records")
    st.session_state.scan_count += 1
    st.session_state.last_scan = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return candidates, opened, "Cycle complete"

def perf():
    trades = pd.DataFrame(st.session_state.trades)
    cash = float(st.session_state.cash)

    if trades.empty:
        equity = cash
        ret = (equity - float(st.session_state.initial_bankroll)) / float(st.session_state.initial_bankroll) * 100
        return trades, cash, equity, 0.0, 0.0, 0, 0, 0.0, ret

    open_df = trades[trades["status"] == "OPEN"]
    closed_df = trades[trades["status"] == "CLOSED"]

    open_value = float((open_df["shares"] * open_df["last_price"]).sum()) if not open_df.empty else 0.0
    realized = float(closed_df["pnl"].sum()) if not closed_df.empty else 0.0
    unrealized = float(open_df["unrealized_pnl"].sum()) if not open_df.empty else 0.0
    equity = cash + open_value
    closed_count = len(closed_df)
    win_rate = float((closed_df["pnl"] > 0).mean() * 100) if closed_count else 0.0
    open_count = len(open_df)
    ret = (equity - float(st.session_state.initial_bankroll)) / float(st.session_state.initial_bankroll) * 100

    return trades, cash, equity, realized, unrealized, open_count, closed_count, win_rate, ret

def pill(action):
    if action == "Possible Bet":
        return "green"
    if action == "Watch":
        return "yellow"
    return "red"

def cpill(conf):
    if conf == "High":
        return "blue"
    if conf == "Medium":
        return "yellow"
    return "gray"

# ----------------------------
# UI
# ----------------------------
st.title("📈 V7.1 Mobile Auto Paper Trader")
st.caption("Automated paper trading for Polymarket. No wallet. No real trades.")

st.warning("Paper-trading only. Use this for 14 days or 30–50 closed trades before risking real money.")

trades_df, cash, equity, realized, unrealized, open_count, closed_count, win_rate, ret = perf()

a, b = st.columns(2)
a.metric("Equity", f"${equity:.2f}", f"{ret:.2f}%")
b.metric("Cash", f"${cash:.2f}")

c, d = st.columns(2)
c.metric("Open", int(open_count))
d.metric("Closed", int(closed_count))

if closed_count:
    st.metric("Win Rate", f"{win_rate:.1f}%")

if st.session_state.last_scan:
    st.caption(f"Last run: {st.session_state.last_scan}")

tab_run, tab_settings, tab_trades, tab_help = st.tabs(["🚀 Run", "⚙️ Settings", "📒 Trades", "🧠 Rules"])

with tab_settings:
    st.subheader("Bot settings")

    auto_mode = st.toggle("Auto-open paper trades", value=True)
    focus = st.selectbox("Focus", ["All", "Politics / News", "Sports", "Business / AI", "Weather / Science"], index=0)
    entry_quality = st.selectbox("Entry quality", ["Possible Bet only", "Possible Bet + Watch"], index=1)

    x, y = st.columns(2)
    max_open = x.slider("Max open", 1, 10, 4)
    max_bet_pct = y.slider("Stake %", 1.0, 15.0, 5.0, step=1.0) / 100

    x, y = st.columns(2)
    take_profit = x.slider("Take profit", 5, 100, 25, step=5) / 100
    stop_loss = y.slider("Stop loss", 5, 80, 25, step=5) / 100

    max_hold_days = st.slider("Max hold days", 1, 30, 7)

    with st.expander("Advanced filters"):
        market_limit = st.slider("Markets scanned", 25, 500, 300, step=25)
        min_liquidity = st.number_input("Min liquidity $", min_value=0, value=5000, step=1000)
        min_volume = st.number_input("Min total volume $", min_value=0, value=10000, step=1000)
        min_price = st.slider("Min price", 0.01, 0.50, 0.08, step=0.01)
        max_price = st.slider("Max price", 0.50, 0.99, 0.92, step=0.01)
        show_research = st.checkbox("Show research-only markets", value=False)
        top_n = st.slider("Candidates shown", 5, 50, 12, step=1)

    st.subheader("Reset account")
    new_bankroll = st.number_input("New starting bankroll", min_value=10.0, value=float(st.session_state.initial_bankroll), step=10.0)
    if st.button("Reset paper account"):
        st.session_state.cash = float(new_bankroll)
        st.session_state.initial_bankroll = float(new_bankroll)
        st.session_state.trades = []
        st.session_state.previous_prices = {}
        st.session_state.last_candidates = []
        st.session_state.scan_count = 0
        st.session_state.last_scan = None
        st.success("Reset complete")

settings = {
    "auto_mode": auto_mode,
    "focus": focus,
    "entry_quality": entry_quality,
    "max_open": max_open,
    "max_bet_pct": max_bet_pct,
    "take_profit": take_profit,
    "stop_loss": stop_loss,
    "max_hold_days": max_hold_days,
    "market_limit": market_limit,
    "min_liquidity": min_liquidity,
    "min_volume": min_volume,
    "min_price": min_price,
    "max_price": max_price,
}

with tab_run:
    st.subheader("Run bot")

    if st.button("🚀 Run Auto Cycle Now", type="primary"):
        with st.spinner("Scanning markets and updating paper trades..."):
            candidates, opened, msg = run_cycle(settings)
        if candidates.empty:
            st.warning(msg)
        else:
            st.success(f"{msg}. Opened {opened} paper trade(s).")

    records = st.session_state.last_candidates
    if not records:
        st.info("Tap **Run Auto Cycle Now** to start.")
    else:
        df = pd.DataFrame(records)
        if not show_research:
            df = df[df["Action"] != "Research Only"]
        df = df.head(top_n)

        st.subheader("Top candidates")
        for _, r in df.iterrows():
            st.markdown(
                f"""
<div class="card">
    <span class="pill {pill(r['Action'])}">{r['Action']}</span>
    <span class="pill {cpill(r['Confidence'])}">{r['Confidence']}</span>
    <span class="pill gray">Score {r['Edge Score']}</span>
    <div class="title">{r['Question']}</div>
    <div class="small"><b>Outcome:</b> {r['Outcome']}</div>
    <div class="small"><b>Price:</b> {r['Market Price']} | <b>Stake:</b> ${r['Suggested Stake']} | <b>Change:</b> {r['Change']}</div>
    <div class="small"><b>Why:</b> {r['Why']}</div>
    <div class="small"><b>Risk:</b> {r['Risk Flags']}</div>
    <br>
    <a href="{r['Open Market']}" target="_blank">Open Market</a> &nbsp; | &nbsp;
    <a href="{r['News Search']}" target="_blank">News</a>
</div>
""",
                unsafe_allow_html=True,
            )

with tab_trades:
    st.subheader("Trade ledger")
    trades_df, cash, equity, realized, unrealized, open_count, closed_count, win_rate, ret = perf()

    if trades_df.empty:
        st.info("No trades yet.")
    else:
        open_df = trades_df[trades_df["status"] == "OPEN"]
        closed_df = trades_df[trades_df["status"] == "CLOSED"]

        st.metric("Realized P/L", f"${realized:.2f}")
        st.metric("Unrealized P/L", f"${unrealized:.2f}")

        st.markdown("### Open positions")
        if open_df.empty:
            st.info("No open positions.")
        else:
            for _, t in open_df.iterrows():
                pnl = float(t["unrealized_pnl"])
                st.markdown(
                    f"""
<div class="card">
    <span class="pill yellow">OPEN</span>
    <span class="pill gray">{t['confidence']}</span>
    <div class="title">{t['question']}</div>
    <div class="small"><b>Outcome:</b> {t['outcome']}</div>
    <div class="small"><b>Entry:</b> {t['entry_price']} | <b>Last:</b> {t['last_price']} | <b>Stake:</b> ${t['stake']}</div>
    <div class="money">Unrealized P/L: ${pnl:.2f}</div>
    <br>
    <a href="{t['open_market']}" target="_blank">Open Market</a> &nbsp; | &nbsp;
    <a href="{t['news_search']}" target="_blank">News</a>
</div>
""",
                    unsafe_allow_html=True,
                )

        st.markdown("### Recent closed trades")
        if closed_df.empty:
            st.info("No closed trades yet.")
        else:
            for _, t in closed_df.tail(10).iloc[::-1].iterrows():
                pnl = float(t["pnl"])
                cls = "green" if pnl >= 0 else "red"
                st.markdown(
                    f"""
<div class="card">
    <span class="pill {cls}">CLOSED</span>
    <span class="pill gray">{t['close_reason']}</span>
    <div class="title">{t['question']}</div>
    <div class="small"><b>Entry:</b> {t['entry_price']} | <b>Exit:</b> {t['exit_price']} | <b>Stake:</b> ${t['stake']}</div>
    <div class="money">P/L: ${pnl:.2f}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        csv = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download trades CSV", csv, "v7_paper_trades.csv", "text/csv")

with tab_help:
    st.subheader("How to test")
    st.markdown(
        """
Use this setup first:

- Starting bankroll: **$100**
- Stake size: **5%**
- Max open positions: **4**
- Take profit: **25%**
- Stop loss: **25%**
- Max hold: **7 days**
- Minimum entry: **Possible Bet + Watch**
- Test period: **14 days or 30–50 closed trades**

This is not a real-money bot. It paper-trades to see whether the strategy has any edge.
"""
    )
