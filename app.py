
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

st.set_page_config(page_title="Polymarket Edge Dashboard", layout="wide")

st.title("Polymarket Edge Dashboard")
st.caption("Read-only research tool. It does not place trades.")

with st.sidebar:
    st.header("Scanner Settings")
    category = st.selectbox(
        "Leaderboard category",
        ["OVERALL", "POLITICS", "SPORTS", "CRYPTO", "CULTURE", "ECONOMICS", "TECH", "FINANCE", "WEATHER"],
        index=0
    )
    time_period = st.selectbox("Leaderboard period", ["DAY", "WEEK", "MONTH", "ALL"], index=1)
    market_limit = st.slider("Markets to scan", 25, 500, 150, step=25)
    min_liquidity = st.number_input("Minimum liquidity $", min_value=0, value=5000, step=1000)
    min_volume = st.number_input("Minimum volume $", min_value=0, value=10000, step=1000)
    max_price = st.slider("Ignore markets priced above", 0.50, 0.99, 0.92, step=0.01)
    min_price = st.slider("Ignore markets priced below", 0.01, 0.50, 0.08, step=0.01)
    bankroll = st.number_input("Bankroll $", min_value=100, value=1000, step=100)
    max_bet_pct = st.slider("Max stake per idea", 0.5, 10.0, 2.0, step=0.5) / 100
    run = st.button("Scan markets", type="primary")

def safe_get(url, params=None, retries=2):
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise last_err

@st.cache_data(ttl=120)
def get_markets(limit):
    data = safe_get(
        f"{GAMMA_API}/markets",
        {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "order": "volume",
            "ascending": "false",
        },
    )
    if not isinstance(data, list):
        data = data.get("markets", data.get("data", []))
    return pd.DataFrame(data)

@st.cache_data(ttl=120)
def get_leaderboard(category, time_period, limit=100):
    # Current official docs describe a trader leaderboard endpoint under the core/data API.
    # Endpoint names have moved before, so this tries common public paths.
    paths = [
        f"{DATA_API}/leaderboard",
        f"{DATA_API}/traders/leaderboard",
        f"{DATA_API}/leaderboard/rankings",
        f"{DATA_API}/rankings",
    ]
    for path in paths:
        try:
            data = safe_get(path, {
                "category": category,
                "timePeriod": time_period,
                "orderBy": "PNL",
                "limit": limit,
            }, retries=1)
            if isinstance(data, dict):
                data = data.get("data", data.get("leaderboard", data.get("rankings", [])))
            df = pd.DataFrame(data)
            if not df.empty:
                return df, path
        except Exception:
            pass
    return pd.DataFrame(), None

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

def pick_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def implied_from_prices(outcome_prices):
    prices = parse_jsonish(outcome_prices)
    out = []
    for p in prices:
        try:
            out.append(float(p))
        except Exception:
            out.append(np.nan)
    return out

def score_market(row):
    volume = float(row.get("volume", row.get("volumeNum", row.get("volume24hr", 0))) or 0)
    liquidity = float(row.get("liquidity", row.get("liquidityNum", 0)) or 0)
    prices = implied_from_prices(row.get("outcomePrices", row.get("outcome_prices", None)))
    outcomes = parse_jsonish(row.get("outcomes", None))

    if len(prices) == 0:
        last = row.get("lastTradePrice", row.get("bestAsk", None))
        try:
            prices = [float(last)]
        except Exception:
            prices = []

    if len(prices) == 0:
        return None

    best_i = int(np.nanargmax(prices))
    price = float(prices[best_i])
    outcome = outcomes[best_i] if best_i < len(outcomes) else "Likely side"

    if not (min_price <= price <= max_price):
        return None
    if volume < min_volume or liquidity < min_liquidity:
        return None

    # Heuristic score:
    # - high volume/liquidity lowers execution risk
    # - avoid super extreme prices
    # - recent activity is useful if available
    vol_score = min(1.0, math.log10(max(volume, 1)) / 7)
    liq_score = min(1.0, math.log10(max(liquidity, 1)) / 6)
    uncertainty_bonus = 1 - abs(price - 0.5) * 1.25
    uncertainty_bonus = max(0, min(1, uncertainty_bonus))

    score = 100 * (0.45 * vol_score + 0.35 * liq_score + 0.20 * uncertainty_bonus)

    # This is NOT a true win probability. It is a conservative research estimate
    # that tells you when the market deserves manual investigation.
    fair_estimate = min(0.95, max(0.05, price + ((score - 60) / 1000)))
    edge = fair_estimate - price

    suggested_stake = 0
    if edge > 0:
        suggested_stake = min(bankroll * max_bet_pct, bankroll * edge * 0.20)

    return {
        "Question": row.get("question", row.get("title", "")),
        "Outcome to inspect": outcome,
        "Market price": round(price, 3),
        "Research fair estimate": round(fair_estimate, 3),
        "Estimated edge": round(edge, 3),
        "Volume": round(volume, 0),
        "Liquidity": round(liquidity, 0),
        "Score": round(score, 1),
        "Suggested max stake": round(suggested_stake, 2),
        "End date": row.get("endDate", row.get("end_date", "")),
        "Slug": row.get("slug", ""),
    }

def market_url(slug):
    if not slug:
        return ""
    return f"https://polymarket.com/market/{slug}"

if run:
    col1, col2 = st.columns([2, 1])

    with st.spinner("Fetching markets and leaderboard data..."):
        markets = get_markets(market_limit)
        leaderboard, leaderboard_path = get_leaderboard(category, time_period)

    st.subheader("Leaderboard check")
    if leaderboard.empty:
        st.warning("Leaderboard endpoint did not return data. The market scanner still works.")
    else:
        st.success(f"Leaderboard data loaded from: {leaderboard_path}")
        st.dataframe(leaderboard.head(25), use_container_width=True)

    st.subheader("Market scanner")
    if markets.empty:
        st.error("No markets returned. Try again later or reduce filters.")
    else:
        rows = []
        for _, r in markets.iterrows():
            scored = score_market(r)
            if scored:
                rows.append(scored)

        df = pd.DataFrame(rows)

        if df.empty:
            st.warning("No markets passed your filters. Lower minimum liquidity/volume or expand scan size.")
        else:
            df = df.sort_values(["Estimated edge", "Score", "Liquidity"], ascending=False)
            df["Open"] = df["Slug"].apply(market_url)
            st.dataframe(
                df.drop(columns=["Slug"]),
                use_container_width=True,
                column_config={
                    "Open": st.column_config.LinkColumn("Open", display_text="Open market")
                }
            )

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv, "polymarket_edge_results.csv", "text/csv")

            st.subheader("How to use the output")
            st.write("""
            Treat the top rows as markets to research manually. The dashboard is intentionally conservative:
            it favors liquidity, volume, and non-extreme pricing. Do not bet only because a row appears here.
            Before entering, check news, resolution rules, spread, order book depth, and whether the market wording has traps.
            """)

else:
    st.info("Set your filters in the sidebar, then click **Scan markets**.")
