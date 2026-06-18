
import json, math, time
from datetime import datetime, timezone
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st

# POLYMARKET V10 LITE — SINGLE FILE
# Upload this as app.py. Paper trading only. No wallet. No real money.

GAMMA_API = "https://gamma-api.polymarket.com"
DEFAULT_BANKROLL = 1000.0

st.set_page_config(page_title="Polymarket V10 Lite", page_icon="🤖", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{padding-top:.75rem;padding-left:.75rem;padding-right:.75rem;max-width:820px}
div.stButton>button{width:100%;min-height:3.1rem;border-radius:16px;font-size:1.05rem;font-weight:800}
.card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.12);border-radius:18px;padding:14px;margin-bottom:13px}
.title{font-size:1rem;font-weight:850;line-height:1.25;margin-top:7px;margin-bottom:7px}
.small{font-size:.83rem;opacity:.82;line-height:1.35}
.pill{display:inline-block;padding:5px 9px;border-radius:999px;margin-right:5px;margin-bottom:5px;font-size:.76rem;font-weight:800}
.green{background:#dcfce7;color:#166534}.yellow{background:#fef3c7;color:#854d0e}.red{background:#fee2e2;color:#991b1b}.blue{background:#dbeafe;color:#1e40af}.gray{background:#e5e7eb;color:#374151}
a{text-decoration:none;font-weight:800}
</style>
""", unsafe_allow_html=True)

def init_state():
    defaults = {
        "cash": DEFAULT_BANKROLL,
        "initial_bankroll": DEFAULT_BANKROLL,
        "trades": [],
        "previous_prices": {},
        "last_candidates": [],
        "scan_count": 0,
        "last_scan": None,
        "snapshots": [],
        "last_skip_reasons": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def safe_get(url, params=None, retries=2):
    err = None
    for _ in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            err = e
            time.sleep(.6)
    raise err

@st.cache_data(ttl=90)
def get_markets(limit):
    data = safe_get(f"{GAMMA_API}/markets", {
        "active": "true", "closed": "false", "limit": int(limit),
        "order": "volume", "ascending": "false"
    })
    return pd.DataFrame(data if isinstance(data, list) else data.get("markets", data.get("data", [])))

def parse_jsonish(v):
    if v is None: return []
    if isinstance(v, list): return v
    if isinstance(v, str):
        try: return json.loads(v)
        except Exception: return [x.strip() for x in v.split(",") if x.strip()]
    return []

def fnum(v, default=0.0):
    try:
        if v is None or v == "": return default
        return float(v)
    except Exception:
        return default

def get_prices(row):
    out = []
    for p in parse_jsonish(row.get("outcomePrices", row.get("outcome_prices", None))):
        try: out.append(float(p))
        except Exception: pass
    return out

def get_outcomes(row):
    return parse_jsonish(row.get("outcomes", None))

def market_url(slug):
    return f"https://polymarket.com/market/{slug}" if slug else "https://polymarket.com/markets"

def news_url(question):
    return "https://www.google.com/search?tbm=nws&q=" + quote_plus(str(question))

def days_until(end_date):
    try:
        if not end_date: return None
        dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
        return (dt - datetime.now(timezone.utc)).days
    except Exception:
        return None

def focus_match(question, focus):
    q = str(question).lower()
    if focus == "All": return True
    buckets = {
        "Sports": ["ufc","nba","nfl","mlb","nhl","soccer","fight","match","championship","wins","score","goal","world cup"],
        "Politics / News": ["trump","biden","iran","israel","china","russia","ukraine","election","president","war","ceasefire","tariff","fed","court"],
        "Business / AI": ["openai","anthropic","cursor","ipo","valuation","apple","tesla","nvidia","google","meta","microsoft","ai","company"],
        "Crypto": ["bitcoin","btc","ethereum","eth","solana","sol","token","coin","airdrop","fdv","meme"],
    }
    return any(w in q for w in buckets.get(focus, []))

def category_for(question):
    q = str(question).lower()
    if any(w in q for w in ["ufc","nba","nfl","mlb","nhl","fight","match","soccer","world cup"]): return "Sports"
    if any(w in q for w in ["trump","biden","iran","israel","china","russia","ukraine","election","war","fed","court","ceasefire"]): return "News"
    if any(w in q for w in ["token","coin","airdrop","fdv","btc","eth","sol","meme"]): return "Crypto"
    if any(w in q for w in ["openai","anthropic","ipo","valuation","nvidia","tesla","company"]): return "Business/AI"
    return "General"

def risk_flags(question, price, volume, volume_24h, liquidity, spread_proxy, end_date):
    q = str(question).lower()
    flags = []
    if liquidity < 2500: flags.append("Low liquidity")
    if volume < 5000: flags.append("Low volume")
    if spread_proxy > .18: flags.append("Wide pricing")
    if price < .06 or price > .94: flags.append("Extreme price")
    if volume_24h > 0 and volume_24h < 1000: flags.append("Weak 24h activity")
    if any(w in q for w in ["fdv","meme","coin","airdrop","token","launch"]): flags.append("Crypto launch risk")
    if any(w in q for w in ["ufc","nba","nfl","mlb","nhl","fight","match"]): flags.append("Sports variance")
    if len(q) > 95: flags.append("Complex wording")
    d = days_until(end_date)
    if d is not None:
        if d <= 2: flags.append("Near expiration")
        elif d > 180: flags.append("Long dated")
    return flags

def category_stats(category):
    trades = pd.DataFrame(st.session_state.trades)
    if trades.empty or "category" not in trades.columns:
        return {"closed":0,"pnl":0.0,"win_rate":0.0,"allowed":True,"multiplier":1.0}
    closed = trades[(trades["status"]=="CLOSED") & (trades["category"]==category)]
    if closed.empty:
        return {"closed":0,"pnl":0.0,"win_rate":0.0,"allowed":True,"multiplier":1.0}
    pnl = float(closed["pnl"].sum())
    wr = float((closed["pnl"] > 0).mean()*100)
    n = len(closed)
    allowed, mult = True, 1.0
    if n >= 6 and pnl < -3: allowed, mult = False, 0.0
    elif n >= 4 and pnl < 0: mult = .60
    elif n >= 6 and pnl > 3 and wr >= 50: mult = 1.20
    elif n >= 10 and pnl > 8 and wr >= 55: mult = 1.35
    return {"closed":n,"pnl":pnl,"win_rate":wr,"allowed":allowed,"multiplier":mult}

def confidence(edge_score, risk_count, liquidity, volume, price):
    if risk_count >= 4: return "Low"
    if edge_score >= 78 and liquidity >= 15000 and volume >= 25000 and .12 <= price <= .88 and risk_count <= 1: return "High"
    if edge_score >= 64 and liquidity >= 5000 and volume >= 10000 and risk_count <= 3: return "Medium"
    return "Low"

def action_label(conf, edge_score, flags):
    hard_avoid = {"Low liquidity","Wide pricing","Complex wording","Crypto launch risk"}
    if any(f in hard_avoid for f in flags): return "Research Only"
    if "Sports variance" in flags:
        return "Watch" if conf in ["High","Medium"] and edge_score >= 64 else "Research Only"
    if conf == "High" and edge_score >= 78 and len(flags) <= 1: return "Possible Bet"
    if conf in ["High","Medium"] and edge_score >= 64: return "Watch"
    return "Research Only"

def stake_size(cash, max_bet_pct, min_stake, conf, edge_score, flags, category):
    if cash <= min_stake or conf == "Low" or len(flags) >= 4: return 0.0
    cat = category_stats(category)
    if not cat["allowed"]: return 0.0
    base = cash * max_bet_pct
    if conf == "High": mult = min(1.0, max(.30, (edge_score - 62) / 26))
    elif conf == "Medium": mult = min(.45, max(.10, (edge_score - 55) / 42))
    else: mult = 0
    mult *= cat["multiplier"]
    for f,p in {"Sports variance":.70,"Near expiration":.70,"Long dated":.70,"Weak 24h activity":.80}.items():
        if f in flags: mult *= p
    raw = base * mult
    return round(min(cash, max(min_stake, raw)), 2) if raw > 0 else 0.0

def build_price_lookup(markets):
    lookup = {}
    for _, row in markets.iterrows():
        slug = row.get("slug", "")
        keybase = slug or row.get("question", row.get("title", ""))
        prices, outcomes = get_prices(row), get_outcomes(row)
        for i, price in enumerate(prices):
            outcome = outcomes[i] if i < len(outcomes) else f"Outcome {i+1}"
            lookup[f"{keybase}||{outcome}"] = float(price)
    return lookup

def score_market(row, settings):
    question = row.get("question", row.get("title", ""))
    if not focus_match(question, settings["focus"]): return None
    slug = row.get("slug", "")
    market_key = slug or question
    end_date = row.get("endDate", row.get("end_date", ""))
    volume = fnum(row.get("volume", row.get("volumeNum", 0)))
    volume_24h = fnum(row.get("volume24hr", row.get("volume24hrClob", 0)))
    liquidity = fnum(row.get("liquidity", row.get("liquidityNum", 0)))
    prices, outcomes = get_prices(row), get_outcomes(row)
    if not prices: return None
    best_i = int(np.nanargmax(prices))
    price = float(prices[best_i])
    outcome = outcomes[best_i] if best_i < len(outcomes) else f"Outcome {best_i+1}"
    if price < settings["min_price"] or price > settings["max_price"]: return None
    if volume < settings["min_volume"] or liquidity < settings["min_liquidity"]: return None
    trade_key = f"{market_key}||{outcome}"
    prev = st.session_state.previous_prices.get(trade_key)
    movement = 0.0 if prev is None else price - float(prev)
    spread_proxy = abs(sum(prices[:2])-1) if len(prices) >= 2 else .12
    flags = risk_flags(question, price, volume, volume_24h, liquidity, spread_proxy, end_date)
    cat = category_for(question)
    vol_score = min(1, math.log10(max(volume,1))/7)
    liq_score = min(1, math.log10(max(liquidity,1))/6)
    price_score = max(0, min(1, 1 - abs(price-.5)*1.35))
    v24_score = min(1, math.log10(max(volume_24h,1))/6) if volume_24h else vol_score*.65
    spread_score = 1 - min(1, spread_proxy*4)
    movement_score = min(1, abs(movement)/.08)
    edge = 100*(.25*vol_score+.25*liq_score+.17*price_score+.13*v24_score+.08*spread_score+.07*movement_score+(.05 if cat!="General" else 0))
    edge = max(0, min(100, edge - 6.5*len(flags)))
    conf = confidence(edge, len(flags), liquidity, volume, price)
    action = action_label(conf, edge, flags)
    stake = stake_size(float(st.session_state.cash), settings["max_bet_pct"], settings["min_stake"], conf, edge, flags, cat)
    reasons = []
    if liquidity >= 15000: reasons.append("good liquidity")
    if volume >= 25000: reasons.append("strong volume")
    if volume_24h >= 5000: reasons.append("recent activity")
    if abs(movement) >= .03: reasons.append(f"moved {movement:+.1%}")
    if cat != "General": reasons.append(f"{cat} category")
    if flags: reasons.append("risks: " + ", ".join(flags[:2]))
    return {
        "Action":action,"Confidence":conf,"Edge Score":round(edge,1),"Market Price":round(price,4),
        "Change":round(movement,4),"Suggested Stake":stake,"Question":question,"Outcome":outcome,
        "Category":cat,"Liquidity":round(liquidity,0),"Volume":round(volume,0),"24h Volume":round(volume_24h,0),
        "Risk Flags":", ".join(flags) if flags else "None","Why":"; ".join(reasons) if reasons else "passed base filters",
        "End Date":end_date,"Open Market":market_url(slug),"News Search":news_url(question),
        "_trade_key":trade_key,"_price":price
    }

def close_trade(trade, exit_price, reason):
    value = float(trade["shares"]) * float(exit_price)
    pnl = value - float(trade["stake"])
    trade.update({"status":"CLOSED","closed_at":datetime.now(timezone.utc).isoformat(),"exit_price":round(float(exit_price),4),
                  "last_price":round(float(exit_price),4),"pnl":round(pnl,2),"unrealized_pnl":0.0,"close_reason":reason})
    st.session_state.cash += value

def update_positions(price_lookup, settings):
    for trade in st.session_state.trades:
        if trade["status"] != "OPEN": continue
        current = price_lookup.get(trade["trade_key"])
        if current is None: continue
        entry, current = float(trade["entry_price"]), float(current)
        trade["last_price"] = round(current,4)
        trade["unrealized_pnl"] = round(float(trade["shares"])*current - float(trade["stake"]), 2)
        pct = (current-entry)/entry if entry else 0
        age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(trade["opened_at"])).total_seconds()/86400
        if pct >= settings["take_profit"]: close_trade(trade, current, f"Take profit +{settings['take_profit']:.0%}")
        elif pct <= -settings["stop_loss"]: close_trade(trade, current, f"Stop loss -{settings['stop_loss']:.0%}")
        elif age_days >= settings["max_hold_days"]: close_trade(trade, current, f"Max hold {settings['max_hold_days']}d")

def already_open(key):
    return any(t["status"]=="OPEN" and t["trade_key"]==key for t in st.session_state.trades)

def open_trade(c):
    stake = min(float(c["Suggested Stake"]), float(st.session_state.cash))
    if stake <= 0 or c["Action"] == "Research Only": return False
    price = float(c["Market Price"])
    trade = {
        "id":len(st.session_state.trades)+1,"status":"OPEN","opened_at":datetime.now(timezone.utc).isoformat(),"closed_at":"",
        "question":c["Question"],"outcome":c["Outcome"],"trade_key":c["_trade_key"],"entry_price":price,"last_price":price,
        "exit_price":None,"stake":round(stake,2),"shares":round(stake/price,6),"unrealized_pnl":0.0,"pnl":0.0,"close_reason":"",
        "action":c["Action"],"confidence":c["Confidence"],"edge_score":c["Edge Score"],"category":c["Category"],
        "open_market":c["Open Market"],"news_search":c["News Search"]
    }
    st.session_state.cash -= stake
    st.session_state.trades.append(trade)
    return True

def auto_enter(candidates, settings):
    opened, skips = 0, []
    open_count = sum(1 for t in st.session_state.trades if t["status"]=="OPEN")
    allowed = ["Possible Bet"] if settings["entry_quality"] == "Possible Bet only" else ["Possible Bet","Watch"]
    for c in candidates:
        if opened >= settings["trades_per_cycle"]: break
        if open_count >= settings["max_open"]:
            skips.append("Max open positions reached"); break
        if c["Action"] not in allowed:
            skips.append(f"Skipped {c['Action']}"); continue
        if already_open(c["_trade_key"]):
            skips.append("Already open"); continue
        if c["Suggested Stake"] <= 0:
            skips.append(f"Stake 0 / blocked: {c.get('Category','Unknown')}"); continue
        if st.session_state.cash < settings["min_stake"]:
            skips.append("Not enough cash"); break
        if open_trade(c):
            opened += 1; open_count += 1
    st.session_state.last_skip_reasons = skips[-5:]
    return opened

def performance():
    trades = pd.DataFrame(st.session_state.trades)
    cash = float(st.session_state.cash)
    if trades.empty:
        equity = cash
        ret = (equity-float(st.session_state.initial_bankroll))/float(st.session_state.initial_bankroll)*100
        return trades,cash,equity,0.0,0.0,0,0,0.0,ret
    open_df, closed_df = trades[trades["status"]=="OPEN"], trades[trades["status"]=="CLOSED"]
    open_value = float((open_df["shares"]*open_df["last_price"]).sum()) if not open_df.empty else 0.0
    realized = float(closed_df["pnl"].sum()) if not closed_df.empty else 0.0
    unrealized = float(open_df["unrealized_pnl"].sum()) if not open_df.empty else 0.0
    equity = cash + open_value
    closed_count = len(closed_df)
    wr = float((closed_df["pnl"] > 0).mean()*100) if closed_count else 0.0
    ret = (equity-float(st.session_state.initial_bankroll))/float(st.session_state.initial_bankroll)*100
    return trades,cash,equity,realized,unrealized,len(open_df),closed_count,wr,ret

def save_snapshot():
    _, cash, equity, realized, unrealized, open_count, closed_count, win_rate, ret = performance()
    st.session_state.snapshots.append({
        "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"equity":round(equity,2),"cash":round(cash,2),
        "return_pct":round(ret,2),"realized_pnl":round(realized,2),"unrealized_pnl":round(unrealized,2),
        "open_positions":int(open_count),"closed_trades":int(closed_count),"win_rate":round(win_rate,1)
    })
    st.session_state.snapshots = st.session_state.snapshots[-300:]

def run_cycle(settings):
    markets = get_markets(settings["market_limit"])
    if markets.empty: return pd.DataFrame(),0,"No markets returned"
    lookup = build_price_lookup(markets)
    update_positions(lookup, settings)
    rows, latest = [], {}
    for _, row in markets.iterrows():
        scored = score_market(row, settings)
        if scored:
            rows.append(scored); latest[scored["_trade_key"]] = scored["_price"]
    if latest: st.session_state.previous_prices = latest
    if not rows: return pd.DataFrame(),0,"No candidates passed filters"
    candidates = pd.DataFrame(rows)
    candidates["_action_sort"] = candidates["Action"].map({"Possible Bet":0,"Watch":1,"Research Only":2})
    candidates["_conf_sort"] = candidates["Confidence"].map({"High":0,"Medium":1,"Low":2})
    candidates = candidates.sort_values(["_action_sort","_conf_sort","Edge Score","Liquidity"], ascending=[True,True,False,False])
    opened = auto_enter(candidates.to_dict("records"), settings) if settings["auto_mode"] else 0
    st.session_state.last_candidates = candidates.to_dict("records")
    st.session_state.scan_count += 1
    st.session_state.last_scan = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_snapshot()
    return candidates, opened, "Cycle complete"

def category_table():
    trades = pd.DataFrame(st.session_state.trades)
    if trades.empty or "category" not in trades.columns: return pd.DataFrame()
    rows = []
    for cat in sorted(trades["category"].dropna().unique()):
        closed = trades[(trades["category"]==cat)&(trades["status"]=="CLOSED")]
        open_ = trades[(trades["category"]==cat)&(trades["status"]=="OPEN")]
        pnl = float(closed["pnl"].sum()) if not closed.empty else 0.0
        wr = float((closed["pnl"] > 0).mean()*100) if not closed.empty else 0.0
        stats = category_stats(cat)
        rows.append({"Category":cat,"Open":len(open_),"Closed":len(closed),"Realized P/L":round(pnl,2),"Win Rate %":round(wr,1),
                     "Status":"Paused" if not stats["allowed"] else "Active","Size Multiplier":round(stats["multiplier"],2)})
    return pd.DataFrame(rows).sort_values(["Status","Realized P/L"], ascending=[True,False])

def pill(action): return "green" if action=="Possible Bet" else ("yellow" if action=="Watch" else "red")
def cpill(conf): return "blue" if conf=="High" else ("yellow" if conf=="Medium" else "gray")

st.title("🤖 Polymarket V10 Lite")
st.caption("Single-file mobile paper trader. Manual run button. No wallet. No real money.")
st.warning("Paper trading only. This version does not run passively in the background.")

trades_df, cash, equity, realized, unrealized, open_count, closed_count, win_rate, ret = performance()
a,b=st.columns(2); a.metric("Equity", f"${equity:.2f}", f"{ret:.2f}%"); b.metric("Cash", f"${cash:.2f}")
c,d=st.columns(2); c.metric("Open", int(open_count)); d.metric("Closed", int(closed_count))
e,f=st.columns(2); e.metric("Realized P/L", f"${realized:.2f}"); f.metric("Win Rate", f"{win_rate:.1f}%")
if st.session_state.last_scan: st.caption(f"Last run: {st.session_state.last_scan}")
st.caption(f"Scans run: {st.session_state.scan_count}")

tab_run, tab_settings, tab_trades, tab_analytics, tab_help = st.tabs(["🚀 Run","⚙️ Settings","📒 Trades","📊 Analytics","🧠 Rules"])

with tab_settings:
    st.subheader("Bot settings")
    auto_mode = st.toggle("Auto-open paper trades", value=True)
    focus = st.selectbox("Focus", ["All","Sports","Politics / News","Business / AI","Crypto"], index=0)
    entry_quality = st.selectbox("Entry quality", ["Possible Bet only","Possible Bet + Watch"], index=1)
    x,y=st.columns(2); max_open=x.slider("Max open positions",1,10,5); trades_per_cycle=y.slider("Trades per cycle",1,5,3)
    x,y=st.columns(2); max_bet_pct=x.slider("Stake %",1.0,15.0,5.0,step=1.0)/100; min_stake=y.slider("Min stake",0.25,10.0,2.0,step=.25)
    x,y=st.columns(2); take_profit=x.slider("Take profit",5,100,25,step=5)/100; stop_loss=y.slider("Stop loss",5,80,25,step=5)/100
    max_hold_days = st.slider("Max hold days",1,30,7)
    with st.expander("Advanced filters"):
        market_limit=st.slider("Markets scanned",25,500,300,step=25)
        min_liquidity=st.number_input("Min liquidity $",min_value=0,value=5000,step=1000)
        min_volume=st.number_input("Min total volume $",min_value=0,value=10000,step=1000)
        min_price=st.slider("Min price",0.01,0.50,0.08,step=0.01)
        max_price=st.slider("Max price",0.50,0.99,0.92,step=0.01)
        show_research=st.checkbox("Show research-only markets",value=False)
        top_n=st.slider("Candidates shown",5,50,12,step=1)
    st.subheader("Reset")
    new_bankroll=st.number_input("New starting bankroll",min_value=10.0,value=float(st.session_state.initial_bankroll),step=10.0)
    if st.button("Reset paper account"):
        st.session_state.cash=float(new_bankroll); st.session_state.initial_bankroll=float(new_bankroll)
        st.session_state.trades=[]; st.session_state.previous_prices={}; st.session_state.last_candidates=[]
        st.session_state.scan_count=0; st.session_state.last_scan=None; st.session_state.snapshots=[]
        st.success("Reset complete")

settings={"auto_mode":auto_mode,"focus":focus,"entry_quality":entry_quality,"max_open":max_open,"trades_per_cycle":trades_per_cycle,
          "max_bet_pct":max_bet_pct,"min_stake":min_stake,"take_profit":take_profit,"stop_loss":stop_loss,"max_hold_days":max_hold_days,
          "market_limit":market_limit,"min_liquidity":min_liquidity,"min_volume":min_volume,"min_price":min_price,"max_price":max_price}

with tab_run:
    st.subheader("Run bot")
    if st.button("🚀 Run Auto Cycle Now", type="primary"):
        with st.spinner("Scanning markets and updating paper trades..."):
            candidates, opened, msg = run_cycle(settings)
        if candidates.empty: st.warning(msg)
        else:
            st.success(f"{msg}. Opened {opened} paper trade(s).")
            if st.session_state.last_skip_reasons: st.caption("Recent skip reasons: " + " | ".join(st.session_state.last_skip_reasons))
    records=st.session_state.last_candidates
    if not records: st.info("Tap **Run Auto Cycle Now** to start.")
    else:
        df=pd.DataFrame(records)
        if not show_research: df=df[df["Action"]!="Research Only"]
        df=df.head(top_n)
        st.subheader("Top candidates")
        for _, r in df.iterrows():
            st.markdown(f"""
<div class="card">
<span class="pill {pill(r['Action'])}">{r['Action']}</span>
<span class="pill {cpill(r['Confidence'])}">{r['Confidence']}</span>
<span class="pill gray">Score {r['Edge Score']}</span>
<div class="title">{r['Question']}</div>
<div class="small"><b>Outcome:</b> {r['Outcome']}</div>
<div class="small"><b>Price:</b> {r['Market Price']} | <b>Stake:</b> ${r['Suggested Stake']} | <b>Change:</b> {r['Change']}</div>
<div class="small"><b>Category:</b> {r['Category']}</div>
<div class="small"><b>Why:</b> {r['Why']}</div>
<div class="small"><b>Risk:</b> {r['Risk Flags']}</div><br>
<a href="{r['Open Market']}" target="_blank">Open Market</a> &nbsp; | &nbsp; <a href="{r['News Search']}" target="_blank">News</a>
</div>""", unsafe_allow_html=True)

with tab_trades:
    st.subheader("Trade ledger")
    trades_df, *_ = performance()
    if trades_df.empty: st.info("No trades yet.")
    else:
        st.dataframe(trades_df, use_container_width=True)
        st.download_button("Download trades CSV", trades_df.to_csv(index=False).encode("utf-8"), "v10_lite_trades.csv", "text/csv")

with tab_analytics:
    st.subheader("Analytics")
    cat_df=category_table()
    if cat_df.empty: st.info("No category stats yet.")
    else:
        st.markdown("### Category performance"); st.dataframe(cat_df, use_container_width=True)
        paused=cat_df[cat_df["Status"]=="Paused"]
        if not paused.empty: st.warning("Paused categories: "+", ".join(paused["Category"].tolist()))
    snaps=pd.DataFrame(st.session_state.snapshots)
    if snaps.empty: st.info("No equity snapshots yet.")
    else:
        st.markdown("### Equity over time"); st.line_chart(snaps.set_index("time")["equity"]); st.dataframe(snaps.tail(20), use_container_width=True)

with tab_help:
    st.subheader("How to use V10 Lite")
    st.markdown("""
This is the easy single-file version.

It scans Polymarket, opens/closes paper trades, tracks category performance, and can pause losing categories after enough data.

It does **not** run passively in the background. You still need to tap **Run Auto Cycle Now**.

Recommended paper test:
- Starting bankroll: **$1,000**
- Stake: **5%**
- Trades per cycle: **3**
- Max open positions: **5**
- Take profit: **25%**
- Stop loss: **25%**
- Test: **30–60 days**
""")
