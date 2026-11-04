#!/usr/bin/env python3
"""
lucid_alerts.py - stuurt een melding naar je telefoon (via de gratis ntfy-app) zodra een van de 3 strategieen een signaal geeft.

Strategieen (5-minuten bars, 1 micro-contract, stop 1,5x ATR, doel 1,5x het risico):
  1. MNQ  Turtle soup               (mislukte uitbraak van het 20-bar hoog/laag)
  2. MNQ  VWAP-terugkeer            (prijs 2 ATR van het dag-VWAP)
  3. MGC  Break & retest weekniveau (uitbraak van hoog/laag van vorige week + terugtest)

Koersen komen van Yahoo Finance (NQ=F en GC=F, 1-minuutdata, enkele minuten vertraagd). Naast de definitieve melding stuurt het script een VOORSIGNAAL zodra de lopende bar zo zou sluiten dat er een signaal komt. Dit is een HYPOTHESE uit een backtest op CFD-data,
geen advies. Het script plaatst GEEN orders; jij zet de order zelf met de stop en het doel uit de melding.

Gebruik:
    pip install pandas numpy
    python3 lucid_alerts.py --topic JOUW-GEHEIME-NAAM          # live draaien
    python3 lucid_alerts.py --topic JOUW-GEHEIME-NAAM --test   # stuurt een testmelding en toont recente signalen
Stel in de ntfy-app (iOS/Android) een abonnement in op dezelfde naam.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
import numpy as np, pandas as pd

TZ = "Europe/Amsterdam"
E0, E1, FLAT = 540, 1290, 1350          # entry 09:00-21:30, alles dicht 22:30 (minuten sinds middernacht, Amsterdam)
STOP_ATR, RR, MAX_TRADES = 1.5, 1.5, 3

# Hoog-impact nieuws (tijden New York); in de ~45 minuten rond deze momenten komt er een waarschuwing en geen signaal.
NEWS = [("2026-11-06", 8, 30), ("2026-11-10", 8, 30), ("2026-11-25", 8, 30), ("2026-12-04", 8, 30), ("2026-12-09", 14, 0),
        ("2026-12-10", 8, 30), ("2026-12-23", 8, 30), ("2027-01-08", 8, 30), ("2027-01-13", 8, 30), ("2027-01-27", 14, 0)]

MARKETS = {"MNQ": dict(symbol="NQ=F", epic="US100", pointval=2.0), "MGC": dict(symbol="GC=F", epic="GOLD", pointval=10.0)}

# ---------- indicatoren
def rma(s, n): return s.ewm(alpha=1 / n, adjust=False).mean()
def atr(b, n=14):
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    return rma(tr, n)
def ev(cond): return cond & ~cond.shift(fill_value=False)
def sg(long, short):
    s = pd.Series(0, index=long.index)
    s[long.fillna(False)] = 1; s[short.fillna(False)] = -1
    return s

def prep(df):
    b = df.resample("5min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    b["date"] = b.index.date
    b["t"] = b.index.hour * 60 + b.index.minute
    b["atr"] = atr(b)
    hlc3 = (b.high + b.low + b.close) / 3
    b["vwap"] = hlc3.groupby(b.date).transform(lambda s: s.expanding().mean())
    return b

# ---------- strategieen (identiek aan de backtest)
def turtle_soup(b):
    lo, hi = b.low.rolling(20).min().shift(), b.high.rolling(20).max().shift()
    return sg(ev((b.low < lo) & (b.close > lo)), ev((b.high > hi) & (b.close < hi)))

def vwap_reversion(b):
    d = (b.close - b.vwap) / b.atr
    return sg(ev(d < -2), ev(d > 2))

def levels_prev_week(b):
    iso = b.index.isocalendar()
    wk = (iso.year.astype(int) * 100 + iso.week.astype(int)).values
    hi = b.groupby(wk).high.max(); lo = b.groupby(wk).low.min()
    order = list(hi.index)
    prev = {order[i]: (hi.iloc[i - 1], lo.iloc[i - 1]) for i in range(1, len(order))}
    H = np.array([prev.get(w, (np.nan, np.nan))[0] for w in wk]); L = np.array([prev.get(w, (np.nan, np.nan))[1] for w in wk])
    return H, L

def break_retest(b, hi_lvl, lo_lvl, window=12, tol=0.15, fail=0.5):
    c, h, l, a, t = (b[k].values for k in ("close", "high", "low", "atr", "t")); d = b["date"].values
    hi, lo = np.asarray(hi_lvl, float), np.asarray(lo_lvl, float)
    n = len(b); sig = np.zeros(n, int); act = None
    for i in range(1, n):
        if act is not None:
            side, lvl, i0, d0 = act
            if d[i] != d0 or i - i0 > window: act = None
            elif side == 1:
                if c[i] < lvl - fail * a[i]: act = None
                elif l[i] <= lvl + tol * a[i] and c[i] > lvl:
                    if E0 <= t[i] < E1: sig[i] = 1
                    act = None
            else:
                if c[i] > lvl + fail * a[i]: act = None
                elif h[i] >= lvl - tol * a[i] and c[i] < lvl:
                    if E0 <= t[i] < E1: sig[i] = -1
                    act = None
        if act is None and not np.isnan(a[i]):
            if not np.isnan(hi[i]) and c[i] > hi[i] and c[i - 1] <= hi[i]: act = (1, hi[i], i, d[i])
            elif not np.isnan(lo[i]) and c[i] < lo[i] and c[i - 1] >= lo[i]: act = (-1, lo[i], i, d[i])
    return pd.Series(sig, index=b.index)

STRATS = [("MNQ Turtle soup", "MNQ", turtle_soup),
          ("MNQ VWAP-terugkeer", "MNQ", vwap_reversion),
          ("MGC Break&retest vorige week", "MGC", lambda b: break_retest(b, *levels_prev_week(b)))]

# ---------- data
def _yahoo(symbol, interval, rng):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.replace('=', '%3D')}?interval={interval}&range={rng}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        j = json.load(r)
    res = j["chart"]["result"][0]; q = res["indicators"]["quote"][0]
    return pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
                        index=pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(TZ)).dropna()

def fetch_yahoo(symbol):
    """5-min historie (1 maand, nodig voor weekniveaus) + 1-min data voor de laatste dagen (minder vertraging).
    De laatste, nog niet afgesloten 5-min bar blijft erin; scan() behandelt die apart."""
    d5 = _yahoo(symbol, "5m", "1mo")
    try:
        d1 = _yahoo(symbol, "1m", "5d")
        d5 = pd.concat([d5[d5.index < d1.index[0]], d1])
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 1-min data niet beschikbaar ({e}); gebruik 5-min")
    return d5


# ---------- Capital.com (gratis demo-account, realtime CFD-koersen); alleen actief als de 3 CAPITAL_* variabelen gezet zijn
class Capital:
    base = "https://demo-api-capital.backend-capital.com/api/v1"
    def __init__(s):
        s.key, s.user, s.pw = [(os.environ.get(k) or "").strip() for k in ("CAPITAL_API_KEY", "CAPITAL_EMAIL", "CAPITAL_PASSWORD")]
        s.hdr = None; s.hist = {}; s.blocked_until = 0
    def enabled(s): return bool(s.key and s.user and s.pw)
    def _req(s, method, path, body=None):
        h = dict(s.hdr or {"X-CAP-API-KEY": s.key}); h["Content-Type"] = "application/json"
        req = urllib.request.Request(s.base + path, data=json.dumps(body).encode() if body else None, headers=h, method=method)
        try:
            return urllib.request.urlopen(req, timeout=20)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300]
            raise RuntimeError(f"HTTP {e.code} op {method} {path.split('?')[0]}: {body}") from None
    def login(s):
        s.hdr = None
        if time.time() < s.blocked_until: raise RuntimeError("inloggen gepauzeerd na eerdere fout (10 min)")
        s.blocked_until = time.time() + 600      # alleen vrijgeven als inloggen lukt; voorkomt 429 en blokkade
        r = s._req("POST", "/session", {"identifier": s.user, "password": s.pw, "encryptedPassword": False})
        s.hdr = {"X-CAP-API-KEY": s.key, "CST": r.headers["CST"], "X-SECURITY-TOKEN": r.headers["X-SECURITY-TOKEN"]}
        s.blocked_until = 0
    def _prices(s, epic, res, **q):
        qs = "&".join(f"{k}={v}" for k, v in q.items())
        for attempt in (0, 1):
            try:
                if s.hdr is None: s.login()
                j = json.load(s._req("GET", f"/prices/{epic}?resolution={res}&{qs}"))
                break
            except RuntimeError as e:
                if "HTTP 401" in str(e) and attempt == 0 and s.hdr is not None: s.hdr = None; continue
                raise
        rows = []
        for c in j["prices"]:
            mid = lambda k: (c[k]["bid"] + c[k]["ask"]) / 2
            rows.append((c["snapshotTimeUTC"], mid("openPrice"), mid("highPrice"), mid("lowPrice"), mid("closePrice")))
        d = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"])
        d.index = pd.to_datetime(d.pop("t"), utc=True).dt.tz_convert(TZ)
        return d
    def fetch(s, epic):
        if epic not in s.hist:                      # eenmalig ~30 dagen 5-min historie (voor weekniveaus)
            end = pd.Timestamp.now("UTC").tz_localize(None); parts = []
            for i in range(10):
                to = end - pd.Timedelta(days=3 * i); fr = to - pd.Timedelta(days=3)
                parts.append(s._prices(epic, "MINUTE_5", max=1000, **{"from": fr.strftime("%Y-%m-%dT%H:%M:%S"), "to": to.strftime("%Y-%m-%dT%H:%M:%S")}))
            h = pd.concat(parts); s.hist[epic] = h[~h.index.duplicated()].sort_index()
        h = s.hist[epic]
        m1 = s._prices(epic, "MINUTE", max=60)       # laatste uur op 1 minuut, samengesteld tot 5-min bars (incl. lopende bar)
        r = m1.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        h = pd.concat([h[h.index < r.index[0]], r]); s.hist[epic] = h[h.index >= h.index[-1] - pd.Timedelta(days=32)]
        return s.hist[epic]
CAP = Capital()

def fetch(symbol):
    if CAP.enabled():
        try:
            epic = next(m["epic"] for m in MARKETS.values() if m["symbol"] == symbol)
            return CAP.fetch(epic)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Capital.com mislukt ({e}); terugval op Yahoo")
    return fetch_yahoo(symbol)

def near_news(ts):
    for d, h, m in NEWS:
        t = pd.Timestamp(f"{d} {h:02d}:{m:02d}", tz="America/New_York").tz_convert(TZ)
        if t - pd.Timedelta(minutes=30) <= ts <= t + pd.Timedelta(minutes=60): return True
    return False

def alert_text(name, mkt, side, entry, a):
    pv = MARKETS[mkt]["pointval"]; sd = STOP_ATR * a
    stop, tgt = entry - side * sd, entry + side * sd * RR
    return (f"{'LONG' if side == 1 else 'SHORT'} {name}",
            f"1 {mkt}-contract | marktorder | STOP {sd:.1f} punten {'onder' if side == 1 else 'boven'} je instap | "
            f"DOEL {sd * RR:.1f} punten {'boven' if side == 1 else 'onder'} je instap | risico ca. ${sd * pv:.0f} | winst ca. ${sd * pv * RR:.0f} | "
            f"(koers hier: entry ca. {entry:.2f}, stop {stop:.2f}, doel {tgt:.2f}; jouw platform wijkt een paar punten af) | alles dicht 22:30")

def push(topic, title, msg, priority="high"):
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode("utf-8"),
                                 headers={"Title": title.encode("ascii", "ignore").decode(), "Priority": priority, "Tags": "chart_with_upwards_trend"})
    urllib.request.urlopen(req, timeout=20).read()

def scan(sent, topic, dry=False, replay_bars=0):
    found = []; cache = {}
    now = pd.Timestamp.now(tz=TZ)
    for name, mkt, fn in STRATS:
        try:
            if mkt not in cache: cache[mkt] = prep(fetch(MARKETS[mkt]["symbol"]))
        except (urllib.error.URLError, KeyError, TimeoutError, ValueError, IndexError) as e:
            print(f"[{time.strftime('%H:%M:%S')}] data {mkt} mislukt: {e}"); continue
        b = cache[mkt]; sig = fn(b)
        partial = b.index[-1] + pd.Timedelta(minutes=5) > now          # laatste bar nog niet afgesloten?
        last_done = len(b) - 2 if partial else len(b) - 1
        idx = range(last_done - max(1, replay_bars) + 1, last_done + 1) if replay_bars else [last_done]
        for i in idx:
            s_ = int(sig.iloc[i]); ts = b.index[i]
            if s_ == 0 or not (E0 <= b.t.iloc[i] < E1) or np.isnan(b.atr.iloc[i]): continue
            key = f"{name}|{ts.isoformat()}"; day_key = f"{name}|{b.date.iloc[i]}"
            if replay_bars == 0:
                if key in sent: continue
                if (now - ts) > pd.Timedelta(minutes=45): continue          # te oud, niet meer instappen
                if sent.get(day_key, 0) >= MAX_TRADES: continue
                if near_news(ts): sent[key] = 1; continue
            title, msg = alert_text(name, mkt, s_, b.close.iloc[i], b.atr.iloc[i])
            found.append((ts, title, msg))
            if replay_bars == 0:
                sent[key] = 1; sent[day_key] = sent.get(day_key, 0) + 1
                if not dry: push(topic, title, msg)
        # voorsignaal: de bar die nu loopt zou bij sluiten op de huidige prijs een signaal geven
        if partial and replay_bars == 0:
            i = len(b) - 1; s_ = int(sig.iloc[i]); ts = b.index[i]
            pkey = f"pre|{name}|{ts.isoformat()}"
            if s_ != 0 and E0 <= b.t.iloc[i] < E1 and not np.isnan(b.atr.iloc[i]) and pkey not in sent and not near_news(ts) \
               and f"{name}|{ts.isoformat()}" not in sent and sent.get(f"{name}|{b.date.iloc[i]}", 0) < MAX_TRADES:
                sent[pkey] = 1
                close_at = (ts + pd.Timedelta(minutes=5)).strftime("%H:%M")
                title, msg = alert_text(name, mkt, s_, b.close.iloc[i], b.atr.iloc[i])
                found.append((ts, "VOORSIGNAAL " + title, msg))
                if not dry:
                    push(topic, "VOORSIGNAAL " + title,
                         f"Maak je klaar. Het signaal is pas definitief als de 5-minuten-bar van {close_at} zo sluit; je krijgt dan nog een melding. Nu: " + msg,
                         priority="default")
    return found

def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="geheime ntfy-kanaalnaam, kies zelf een lange willekeurige naam")
    ap.add_argument("--test", action="store_true", help="stuur een testmelding en toon de signalen van de laatste ~2 dagen")
    ap.add_argument("--once", action="store_true", help="een keer controleren en stoppen (voor GitHub Actions)")
    ap.add_argument("--loop", type=int, default=0, help="blijf zoveel seconden elke --every seconden controleren (voor GitHub Actions)")
    ap.add_argument("--every", type=int, default=20, help="seconden tussen controles (standaard 20)")
    a = ap.parse_args()
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts_sent.json")
    sent = json.load(open(state_file)) if os.path.exists(state_file) else {}
    if a.test:
        push(a.topic, "Test lucid_alerts", "Als je dit leest, werken de meldingen.", "default")
        print("Testmelding gestuurd. Signalen in de laatste 2 dagen (alleen ter controle, geen meldingen):")
        for ts, title, msg in sorted(scan(sent, a.topic, dry=True, replay_bars=576)):
            print(" ", ts.strftime("%a %d-%m %H:%M"), title, "|", msg)
        return
    if a.once or a.loop:
        t_end = time.time() + a.loop
        while True:
            for ts, title, msg in scan(sent, a.topic):
                print(f"{title} | {msg}", flush=True)
            json.dump(sent, open(state_file, "w"))
            if a.once or time.time() + a.every >= t_end: return
            time.sleep(a.every)
    print("lucid_alerts draait. Stoppen met Ctrl+C. Laat dit venster open en je computer aan.")
    while True:
        for ts, title, msg in scan(sent, a.topic):
            print(f"[{time.strftime('%H:%M:%S')}] {title} | {msg}")
        json.dump(sent, open(state_file, "w"))
        time.sleep(a.every)

if __name__ == "__main__":
    main()
