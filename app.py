
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

GAMMA_API = "https://gamma-api.polymarket.com"

st.set_page_config(page_title="Polymarket Edge Dashboard V3", layout="wide")

st.title("Polymarket Edge Dashboard V3")
st.caption("Read-only betting research tool. It does not place trades or connect to your wallet.")

with st.sidebar:
    st.header("Scanner Settings")
    market_limit = st.slider("Markets to scan", 25, 500, 300, step=25)
    min_liquidity = st.number_input("Minimum liquidity $", min_value=0, value=5000, step=1000)
    min_volume = st.number_input("Minimum total volume $", min_value=0, value=10000, step=1000)
    min_price = st.slider("Ignore prices below", 0.01, 0.50, 0.08, step=0.01)
    max_price = st.slider("Ignore prices above", 0.50, 0.99, 0.92, step=0.01)
    bankroll = st.number_input("Bankroll $", min_value=100, value=1000, step=100)
    max_bet_pct = st.slider("Max bet per market", 0.5, 5.0, 2.0, step=0.5) / 100

    st.header("Focus")
    focus = st.selectbox(
        "Market focus",
        ["All", "Politics / News", "Sports", "Crypto / Token", "Business / AI", "Weather / Science"],
        index=0
    )
    show_research_only = st.checkbox("Show research-only / risky markets", value=False)
    top_n = st.slider("Show top N rows", 5, 50, 15, step=5)
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


def days_until(end_date):
    try:
        if not end_date:
            return None
        dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
        return (dt - datetime.now(timezone.utc)).days
    except Exception:
        return None


def focus_match(question, focus):
    q = str(question).lower()
    if focus == "All":
        return True
    if focus == "Politics / News":
        words = ["trump", "biden", "iran", "israel", "china", "russia", "ukraine", "election", "president", "leader", "congress", "senate", "war", "ceasefire", "tariff", "fed", "court"]
        return any(w in q for w in words)
    if focus == "Sports":
        words = ["ufc", "nba", "nfl", "mlb", "nhl", "soccer", "fight", "match", "championship", "wins", "score", "goal", "holloway", "mcgregor"]
        return any(w in q for w in words)
    if focus == "Crypto / Token":
        words = ["crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "token", "coin", "fdv", "airdrop", "launch", "stablecoin"]
        return any(w in q for w in words)
    if focus == "Business / AI":
        words = ["openai", "anthropic", "cursor", "anysphere", "ipo", "valuation", "apple", "tesla", "nvidia", "google", "meta", "microsoft", "ai", "company"]
        return any(w in q for w in words)
    if focus == "Weather / Science":
        words = ["weather", "temperature", "hurricane", "storm", "rain", "snow", "earthquake", "climate", "science", "nasa"]
        return any(w in q for w in words)
    return True


def risk_flags(question, price, volume, volume_24h, liquidity, spread_proxy, end_date):
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


def catalyst_category(question):
    q = str(question).lower()
    if any(w in q for w in ["iran", "israel", "ukraine", "russia", "china", "trump", "election", "tariff", "fed", "court", "ceasefire"]):
        return "News catalyst likely"
    if any(w in q for w in ["ufc", "nba", "nfl", "mlb", "nhl", "fight", "match"]):
        return "Sports event"
    if any(w in q for w in ["fdv", "token", "coin", "airdrop", "launch", "btc", "eth", "sol"]):
        return "Crypto/token event"
    if any(w in q for w in ["ipo", "valuation", "openai", "anthropic", "cursor", "anysphere", "nvidia", "tesla"]):
        return "Business/AI event"
    return "General market"


def confidence_rating(edge_score, risk_count, liquidity, volume, price, volume_24h):
    if risk_count >= 4:
        return "Low"
    if edge_score >= 76 and liquidity >= 15000 and volume >= 25000 and 0.12 <= price <= 0.88 and risk_count <= 1:
        return "High"
    if edge_score >= 62 and liquidity >= 5000 and volume >= 10000 and risk_count <= 3:
        return "Medium"
    return "Low"


def action_label(confidence, edge_score, flags):
    hard_avoid = {"Low liquidity", "Wide/uncertain pricing", "Complex wording"}
    if any(f in hard_avoid for f in flags):
        return "Research Only"
    if confidence == "High" and edge_score >= 76:
        return "Possible Bet"
    if confidence in ["High", "Medium"] and edge_score >= 62:
        return "Watch / Small Bet"
    return "Research Only"


def suggested_stake(bankroll, max_bet_pct, confidence, edge_score, flags):
    if confidence == "Low" or len(flags) >= 4:
        return 0.0

    base = bankroll * max_bet_pct

    if confidence == "High":
        mult = min(1.0, max(0.35, (edge_score - 60) / 25))
    elif confidence == "Medium":
        mult = min(0.55, max(0.15, (edge_score - 50) / 40))
    else:
        mult = 0

    penalty_map = {
        "Sports variance": 0.70,
        "Complex wording": 0.50,
        "Political/personnel uncertainty": 0.75,
        "Crypto launch risk": 0.50,
        "Near expiration": 0.70,
        "Long dated": 0.70,
        "Weak 24h activity": 0.80,
    }

    for flag, penalty in penalty_map.items():
        if flag in flags:
            mult *= penalty

    return round(base * mult, 2)


def why_showing(row, confidence, edge_score, flags, catalyst, liquidity, volume, volume_24h):
    reasons = []
    if confidence == "High":
        reasons.append("strong market quality")
    elif confidence == "Medium":
        reasons.append("decent market quality")
    else:
        reasons.append("low-confidence research candidate")

    if liquidity >= 15000:
        reasons.append("good liquidity")
    if volume >= 25000:
        reasons.append("strong total volume")
    if volume_24h >= 5000:
        reasons.append("active recent trading")
    if catalyst != "General market":
        reasons.append(catalyst.lower())
    if flags:
        reasons.append("risks: " + ", ".join(flags[:3]))

    return "; ".join(reasons)


def score_market(row, bankroll, max_bet_pct):
    question = row.get("question", row.get("title", ""))
    if not focus_match(question, focus):
        return None

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

    # Binary markets should roughly sum to 1. If they do not, pricing may be stale or messy.
    if len(prices) >= 2:
        spread_proxy = abs(sum(prices[:2]) - 1)
    else:
        spread_proxy = 0.12

    vol_score = min(1, math.log10(max(volume, 1)) / 7)
    liq_score = min(1, math.log10(max(liquidity, 1)) / 6)
    price_score = max(0, min(1, 1 - abs(price - 0.5) * 1.35))
    v24_score = min(1, math.log10(max(volume_24h, 1)) / 6) if volume_24h else vol_score * 0.65
    spread_score = 1 - min(1, spread_proxy * 4)

    cat = catalyst_category(question)
    catalyst_bonus = 0.05 if cat != "General market" else 0

    flags = risk_flags(question, price, volume, volume_24h, liquidity, spread_proxy, end_date)

    edge_score = 100 * (
        0.28 * vol_score +
        0.27 * liq_score +
        0.18 * price_score +
        0.14 * v24_score +
        0.08 * spread_score +
        catalyst_bonus
    )

    edge_score -= 6.5 * len(flags)
    edge_score = max(0, min(100, edge_score))

    confidence = confidence_rating(edge_score, len(flags), liquidity, volume, price, volume_24h)
    action = action_label(confidence, edge_score, flags)
    stake = suggested_stake(bankroll, max_bet_pct, confidence, edge_score, flags)

    research_estimate = price + ((edge_score - 55) / 1200)
    research_estimate = max(0.03, min(0.97, research_estimate))
    estimated_edge = research_estimate - price

    why = why_showing(row, confidence, edge_score, flags, cat, liquidity, volume, volume_24h)

    return {
        "Action": action,
        "Confidence": confidence,
        "Edge Score": round(edge_score, 1),
        "Market Price": round(price, 3),
        "Research Estimate": round(research_estimate, 3),
        "Estimated Edge": round(estimated_edge, 3),
        "Suggested Max Bet": stake,
        "Question": question,
        "Outcome to inspect": outcome,
        "Catalyst Type": cat,
        "Liquidity": round(liquidity, 0),
        "Volume": round(volume, 0),
        "24h Volume": round(volume_24h, 0),
        "Risk Flags": ", ".join(flags) if flags else "None",
        "Why Showing": why,
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
            st.warning("No markets passed your filters. Lower the filters or change focus.")
        else:
            if not show_research_only:
                df = df[df["Action"] != "Research Only"]

            if df.empty:
                st.warning("Only research-only markets were found. Turn on 'Show research-only / risky markets' to view them.")
            else:
                action_order = {"Possible Bet": 0, "Watch / Small Bet": 1, "Research Only": 2}
                conf_order = {"High": 0, "Medium": 1, "Low": 2}
                df["_action_sort"] = df["Action"].map(action_order)
                df["_conf_sort"] = df["Confidence"].map(conf_order)

                df = df.sort_values(
                    ["_action_sort", "_conf_sort", "Edge Score", "Liquidity"],
                    ascending=[True, True, False, False]
                ).drop(columns=["_action_sort", "_conf_sort"]).head(top_n)

                possible = (df["Action"] == "Possible Bet").sum()
                watch = (df["Action"] == "Watch / Small Bet").sum()
                avg_score = round(df["Edge Score"].mean(), 1)

                c1, c2, c3 = st.columns(3)
                c1.metric("Possible Bets", int(possible))
                c2.metric("Watch / Small Bet", int(watch))
                c3.metric("Average Edge Score", avg_score)

                st.subheader("Best research candidates")
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        "Open": st.column_config.LinkColumn("Open Market", display_text="Open"),
                    }
                )

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "polymarket_edge_v3_results.csv", "text/csv")

                st.subheader("How to use this")
                st.markdown("""
                **Possible Bet** = worth serious research before entering.  
                **Watch / Small Bet** = maybe, but keep size small.  
                **Research Only** = do not bet blindly.

                **Market Price** is the market-implied probability.  
                **Research Estimate** is a conservative scanner estimate, not a true forecast.  
                **Edge Score** ranks market quality and signal strength.  
                **Confidence** reflects trade quality, not guaranteed outcome probability.

                Before betting, open the market and check:
                1. The exact resolution rules.
                2. The latest news or event catalyst.
                3. The bid/ask spread and order book depth.
                4. Whether you actually have a reason the market is wrong.
                """)
else:
    st.info("Use the sidebar settings, then click **Scan Markets**.")
