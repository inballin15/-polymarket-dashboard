
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

GAMMA_API = "https://gamma-api.polymarket.com"

st.set_page_config(page_title="Polymarket Edge Dashboard V2", layout="wide")

st.title("Polymarket Edge Dashboard V2")
st.caption("Read-only betting research tool. It does not place trades.")

with st.sidebar:
    st.header("Scanner Settings")
    market_limit = st.slider("Markets to scan", 25, 500, 250, step=25)
    min_liquidity = st.number_input("Minimum liquidity $", min_value=0, value=5000, step=1000)
    min_volume = st.number_input("Minimum volume $", min_value=0, value=10000, step=1000)
    min_price = st.slider("Ignore prices below", 0.01, 0.50, 0.08, step=0.01)
    max_price = st.slider("Ignore prices above", 0.50, 0.99, 0.92, step=0.01)
    bankroll = st.number_input("Bankroll $", min_value=100, value=1000, step=100)
    max_bet_pct = st.slider("Max bet per market", 0.5, 5.0, 2.0, step=0.5) / 100
    show_do_not_bet = st.checkbox("Show Do Not Bet markets", value=False)
    run = st.button("Scan Markets", type="primary")


def safe_get(url, params=None, retries=2):
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.6)
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


def market_url(slug):
    if not slug:
        return "https://polymarket.com/markets"
    return f"https://polymarket.com/market/{slug}"


def get_prices(row):
    prices = parse_jsonish(row.get("outcomePrices", row.get("outcome_prices", None)))
    clean = []
    for p in prices:
        try:
            clean.append(float(p))
        except Exception:
            pass
    return clean


def get_outcomes(row):
    return parse_jsonish(row.get("outcomes", None))


def risk_flags(question, price, volume, liquidity, spread_proxy, end_date):
    q = str(question).lower()
    flags = []

    if liquidity < 2500:
        flags.append("Low liquidity")
    if volume < 5000:
        flags.append("Low volume")
    if spread_proxy > 0.18:
        flags.append("Wide/uncertain pricing")
    if price < 0.06 or price > 0.94:
        flags.append("Extreme price")
    if any(w in q for w in ["fdv", "meme", "coin", "airdrop", "token", "launch"]):
        flags.append("Meme/crypto launch risk")
    if any(w in q for w in ["ufc", "nba", "nfl", "mlb", "nhl", "fight", "match"]):
        flags.append("Sports variance")
    if any(w in q for w in ["will ", "by ", "before ", "end "]) and len(q) > 90:
        flags.append("Check wording carefully")
    if any(w in q for w in ["resign", "out as", "ceo", "leader", "president"]):
        flags.append("Political/personnel uncertainty")

    try:
        if end_date:
            dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
            days_left = (dt - datetime.now(timezone.utc)).days
            if days_left <= 2:
                flags.append("Near expiration")
    except Exception:
        pass

    return flags


def confidence_rating(edge_score, risk_count, liquidity, volume, price):
    if risk_count >= 3:
        return "Low"
    if edge_score >= 72 and liquidity >= 15000 and volume >= 25000 and 0.15 <= price <= 0.85 and risk_count <= 1:
        return "High"
    if edge_score >= 58 and liquidity >= 5000 and volume >= 10000 and risk_count <= 2:
        return "Medium"
    return "Low"


def action_label(confidence, edge_score, flags):
    if "Low liquidity" in flags or "Wide/uncertain pricing" in flags:
        return "Research Only"
    if confidence == "High" and edge_score >= 72:
        return "Possible Bet"
    if confidence == "Medium" and edge_score >= 60:
        return "Watch / Small Bet"
    return "Research Only"


def suggested_stake(bankroll, max_bet_pct, confidence, edge_score, flags):
    if confidence == "Low" or len(flags) >= 3:
        return 0.0

    base = bankroll * max_bet_pct

    if confidence == "High":
        mult = min(1.0, max(0.35, (edge_score - 60) / 25))
    elif confidence == "Medium":
        mult = min(0.50, max(0.15, (edge_score - 50) / 40))
    else:
        mult = 0

    if "Sports variance" in flags:
        mult *= 0.70
    if "Check wording carefully" in flags:
        mult *= 0.70
    if "Political/personnel uncertainty" in flags:
        mult *= 0.75
    if "Meme/crypto launch risk" in flags:
        mult *= 0.50

    return round(base * mult, 2)


def score_market(row, bankroll, max_bet_pct):
    question = row.get("question", row.get("title", ""))
    slug = row.get("slug", "")
    end_date = row.get("endDate", row.get("end_date", ""))

    volume = fnum(row.get("volume", row.get("volumeNum", row.get("volume24hr", 0))))
    volume_24h = fnum(row.get("volume24hr", row.get("volume24hrClob", 0)))
    liquidity = fnum(row.get("liquidity", row.get("liquidityNum", 0)))

    prices = get_prices(row)
    outcomes = get_outcomes(row)

    if not prices:
        return None

    best_i = int(np.nanargmax(prices))
    price = float(prices[best_i])
    outcome = outcomes[best_i] if best_i < len(outcomes) else "Likely side"

    if price < min_price or price > max_price:
        return None
    if volume < min_volume or liquidity < min_liquidity:
        return None

    # Spread proxy: if binary market has two prices summing far from 1, treat as risky/unclear.
    if len(prices) >= 2:
        spread_proxy = abs(sum(prices[:2]) - 1)
    else:
        spread_proxy = 0.12

    vol_score = min(1, math.log10(max(volume, 1)) / 7)
    liq_score = min(1, math.log10(max(liquidity, 1)) / 6)
    price_score = max(0, min(1, 1 - abs(price - 0.5) * 1.35))
    v24_score = min(1, math.log10(max(volume_24h, 1)) / 6) if volume_24h else vol_score * 0.7

    flags = risk_flags(question, price, volume, liquidity, spread_proxy, end_date)

    # Edge Score is not a true probability. It is a ranking score for bet research.
    edge_score = 100 * (
        0.30 * vol_score +
        0.28 * liq_score +
        0.20 * price_score +
        0.12 * v24_score +
        0.10 * (1 - min(1, spread_proxy * 4))
    )

    # Penalties
    edge_score -= 7 * len(flags)
    edge_score = max(0, min(100, edge_score))

    confidence = confidence_rating(edge_score, len(flags), liquidity, volume, price)
    action = action_label(confidence, edge_score, flags)
    stake = suggested_stake(bankroll, max_bet_pct, confidence, edge_score, flags)

    # Not a real model probability. This is a research estimate to compare against price.
    research_estimate = price + ((edge_score - 55) / 1200)
    research_estimate = max(0.03, min(0.97, research_estimate))
    edge = research_estimate - price

    return {
        "Action": action,
        "Confidence": confidence,
        "Edge Score": round(edge_score, 1),
        "Question": question,
        "Outcome to inspect": outcome,
        "Market Price": round(price, 3),
        "Research Estimate": round(research_estimate, 3),
        "Estimated Edge": round(edge, 3),
        "Suggested Max Bet": stake,
        "Liquidity": round(liquidity, 0),
        "Volume": round(volume, 0),
        "24h Volume": round(volume_24h, 0),
        "Risk Flags": ", ".join(flags) if flags else "None",
        "End Date": end_date,
        "Open": market_url(slug),
    }


if run:
    with st.spinner("Scanning live Polymarket markets..."):
        markets = get_markets(market_limit)

    if markets.empty:
        st.error("No markets returned. Try again later.")
    else:
        rows = []
        for _, row in markets.iterrows():
            scored = score_market(row, bankroll, max_bet_pct)
            if scored:
                rows.append(scored)

        df = pd.DataFrame(rows)

        if df.empty:
            st.warning("No markets passed your filters. Lower the liquidity/volume filters and scan again.")
        else:
            if not show_do_not_bet:
                df = df[df["Action"] != "Research Only"]

            if df.empty:
                st.warning("Only research-only markets were found. Turn on 'Show Do Not Bet markets' to view them.")
            else:
                action_order = {"Possible Bet": 0, "Watch / Small Bet": 1, "Research Only": 2}
                conf_order = {"High": 0, "Medium": 1, "Low": 2}
                df["_action_sort"] = df["Action"].map(action_order)
                df["_conf_sort"] = df["Confidence"].map(conf_order)

                df = df.sort_values(
                    ["_action_sort", "_conf_sort", "Edge Score", "Liquidity"],
                    ascending=[True, True, False, False]
                ).drop(columns=["_action_sort", "_conf_sort"])

                st.subheader("Best research candidates")
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        "Open": st.column_config.LinkColumn("Open Market", display_text="Open"),
                    }
                )

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "polymarket_edge_v2_results.csv", "text/csv")

                st.subheader("How to read this")
                st.markdown("""
                **Possible Bet** = worth serious research before entering.  
                **Watch / Small Bet** = maybe, but keep size small.  
                **Research Only** = do not bet blindly.

                **Confidence is not a guarantee.** It is based on liquidity, volume, price structure, and risk flags.
                Before betting, always check the market rules, the latest news, and the order book spread.
                """)
else:
    st.info("Use the sidebar settings, then click **Scan Markets**.")
