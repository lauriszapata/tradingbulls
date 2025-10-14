# Limita el rango de fechas para la API de Binance
def clamp_to_api_range(start_ms, end_ms, max_days=30):
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    if end_ms > now_ms:
        end_ms = now_ms
    min_start = end_ms - max_days * 24 * 60 * 60 * 1000
    if start_ms < min_start:
        start_ms = min_start
    return start_ms, end_ms
# --- FUNCIONES DE FEATURES Y ETIQUETADO ---
def add_macro_pair(df_sym, df_eth_indic):
    aux = df_sym[["timestamp"]].merge(
        df_eth_indic[["timestamp","close","ATR","EMA7","EMA25"]]
        .rename(columns={"close":"eth_close","ATR":"eth_ATR","EMA7":"eth_EMA7","EMA25":"eth_EMA25"}),
        on="timestamp", how="left"
    ).ffill()
    out = df_sym.copy()
    out["ret_sym_5m"]  = out["close"].pct_change(fill_method=None)
    out["ret_sym_1h"]  = out["close"].pct_change(12, fill_method=None)
    out["ret_eth_5m"]  = aux["eth_close"].pct_change(fill_method=None)
    out["ret_eth_1h"]  = aux["eth_close"].pct_change(12, fill_method=None)
    out["eth_atr_pct"] = (aux["eth_ATR"] / aux["eth_close"] * 100).replace([np.inf,-np.inf], np.nan)
    out["sym_ema_slope"] = out["EMA7"] - out["EMA25"]
    out["eth_ema_slope"] = aux["eth_EMA7"] - aux["eth_EMA25"]
    return out

def build_features(df):
    X = df[FEATURE_COLS].shift(1)
    return X.replace([np.inf,-np.inf], np.nan).ffill().bfill()

def label_hits(df, k_tp, k_sl, H):
    close, high, low = df["close"].values, df["high"].values, df["low"].values
    atrp = df["ATR_pct"].values / 100.0
    labels = np.full(len(df), np.nan)
    for i in range(len(df)-H-1):
        entry = close[i+1]
        is_long = df["trend_long"].iloc[i+1] == 1
        tp, sl = k_tp*atrp[i+1], k_sl*atrp[i+1]
        hit = None
        for j in range(i+2, i+2+H):
            if j >= len(df):
                break
            if is_long:
                if low[j]  <= entry*(1-sl):
                    hit = 0
                    break
                if high[j] >= entry*(1+tp):
                    hit = 1
                    break
            else:
                if high[j] >= entry*(1+sl):
                    hit = 0
                    break
                if low[j]  <= entry*(1-tp):
                    hit = 1
                    break
        labels[i+1] = np.nan if hit is None else hit
    return pd.Series(labels, index=df.index)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tommy Bulls - Duo 5m Backtest [PROGRESS + CONTEXT CACHE]
(see header in prior attempt for full description)
"""
# (Shortened header to avoid issues)

import os, time, math, argparse, sys, requests
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

EXCHANGE_ID = "binanceusdm"
TIMEFRAME = "5m"
TRAIN_START, TRAIN_END = "2025-06-01 00:00:00", "2025-07-31 23:59:59"
TEST_MONTHS = ["2025-09"]

TAKER_FEE, SLIPPAGE = 0.0005, 0.0002
MARGIN_USD_DEFAULT, LEVERAGE_DEFAULT = 100.0, 5.0

GRID_K_TP  = [1.75, 2.0]
GRID_K_SL  = [1.25, 1.5]
GRID_PMIN  = [0.55, 0.60, 0.65]
GRID_H     = [8, 10, 12]
ADX_MIN_GRID       = [15, 18, 20]
DIST_VWAP_MAX_GRID = [0.5, 1.0]
PF_MIN_SELECT     = 1.3
SHARPE_MIN_SELECT = 1.0
MAX_DD_PCT_SELECT = 0.20

BINANCE_FAPI = "https://fapi.binance.com"

FEATURE_COLS = [
    "close","volume","EMA7","EMA25","EMA99","RSI14","ADX14","ATR_pct",
    "BB_width","BB_%B","VWAP","dist_VWAP_pct","trend_long","trend_short",
    "fundingRate","fundingRate_z","openInterest","oi_change_pct","glsr","glsr_z","tlsr","tlsr_z",
    "ret_sym_5m","ret_sym_1h","ret_eth_5m","ret_eth_1h","eth_atr_pct","sym_ema_slope","eth_ema_slope",
    "dow","hour","sess_asia","sess_eu","sess_us",
    "vr_low","vr_mid","vr_high"
]

def utc_ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)

# Convierte una fecha en string a milisegundos UTC

def get_exchange():
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    ex.timeout = 20000
    ex.enableRateLimit = True
    return ex

# Inicializa el objeto de exchange CCXT para Binance USDM

def http_get_with_retries(url, params, retries=5, backoff=1.8, timeout=20):
    last_err=None
    for a in range(1, retries+1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.ok: return r.json()
            last_err = requests.exceptions.HTTPError(f"{r.status_code} {r.text}")
        except requests.exceptions.RequestException as e:
            last_err = e
        time.sleep(backoff ** a)
    raise last_err if last_err else RuntimeError("HTTP error")

# Realiza una petición HTTP con reintentos exponenciales

def fetch_ohlcv_all(ex, sym, tf, s_ms, e_ms, lim=1000):
    out, ms = [], s_ms
    pbar = tqdm(total=1, desc=f"Fetch {sym} {tf}", leave=False)
    last = s_ms
    while True:
        tries = 0
        while True:
            try:
                batch = ex.fetch_ohlcv(sym, timeframe=tf, since=ms, limit=lim)
                break
            except Exception:
                tries += 1
                if tries >= 5: raise
                time.sleep(1.5 * tries)
        if not batch: break
        out += batch
        last = batch[-1][0]
        ms = last + 1
        if last >= e_ms: break
        span = max(e_ms - s_ms, 1)
        pbar.n = min(1.0, (last - s_ms) / span)
        pbar.refresh()
        time.sleep(0.45)
    pbar.n = 1.0; pbar.refresh(); pbar.close()
    df = pd.DataFrame(out, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp")

# Descarga todas las velas OHLCV en el rango dado, con control de reintentos y barra de progreso

def binance_get(url, params): return http_get_with_retries(url, params, 5, 1.8, 20)

# Wrapper para peticiones HTTP a la API de Binance con reintentos

def cache_path(cache_dir, sym, name, start_ms, end_ms):
    return os.path.join(cache_dir, f"{name}_{sym}_{start_ms}_{end_ms}.parquet")

# Genera la ruta de archivo para caché de contexto


def save_cache(df, path):
    try:
        if df is None: return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)

    except Exception:
        pass

# Guarda un DataFrame en caché (formato parquet)

def load_cache(path):
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        return None
    return None

# Carga un DataFrame desde caché si existe

def fetch_funding(symbol, start, end):
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    out, p = [], {"symbol": symbol, "limit": 1000, "startTime": start, "endTime": end}
    total = end - start
    last_time = start
    while True:
        try:
            d = binance_get(url, p)
        except Exception: break
        if not d: break
        out += d
        if len(d) < 1000: break
        last_time = d[-1]["fundingTime"]
        p["startTime"] = last_time + 1
        percent = min(100, 100 * (last_time - start) / total)
        print(f"Funding {symbol}: {percent:.1f}% completado")
        time.sleep(0.2)
    print(f"Funding {symbol}: 100% completado")
    if not out: return pd.DataFrame(columns=["timestamp","fundingRate"])
    df = pd.DataFrame(out)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["timestamp","fundingRate"]]

# Descarga la tasa de fondeo histórica para el símbolo

def fetch_open_interest_1h(symbol, start, end):

    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    step = 24 * 60 * 60 * 1000  # 1 día
    out, ms = [], start
    total = end - start
    while ms <= end:
        e = min(end, ms + step - 1)
        p = {"symbol": symbol, "period": "1h", "startTime": ms, "endTime": e, "limit": 500}
        print(f"Solicitando OI {symbol}: desde {ms} hasta {e}")
        tries = 0
        max_tries = 2
        while tries < max_tries:
            percent = min(100, 100 * (ms - start) / total)
            print(f"OpenInterest {symbol}: {percent:.1f}% completado (intento {tries+1}/{max_tries})")
            try:
                import requests
                r = requests.get(url, params=p, timeout=40)
                print(f"Status: {r.status_code}")
                if r.ok:
                    d = r.json()
                    print(f"Recibidos {len(d)} registros")
                    if d: out += d
                else:
                    print(f"Error HTTP: {r.status_code} {r.text}")
                ms = e + 1; time.sleep(2)
                break
            except Exception as ex:
                tries += 1
                print(f"Excepción: {ex}")
                time.sleep(5.0)
        else:
            ms = e + 1
    print(f"OpenInterest {symbol}: 100% completado")
    if not out: return pd.DataFrame(columns=["timestamp","openInterest"])
    df = pd.DataFrame(out)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    val = "sumOpenInterestValue" if "sumOpenInterestValue" in df.columns else "sumOpenInterest"
    df["openInterest"] = pd.to_numeric(df[val], errors="coerce")
    return df[["timestamp","openInterest"]]

# Descarga el interés abierto histórico (1h) para el símbolo

def fetch_long_short_1h(symbol, start, end):
    frames=[]
    for ep in ["globalLongShortAccountRatio","topLongShortAccountRatio"]:
        url=f"{BINANCE_FAPI}/futures/data/{ep}"
        step=7*24*60*60*1000; ms=start; out=[]
        while ms<=end:
            e=min(end,ms+step-1)
            p={"symbol":symbol,"period":"1h","startTime":ms,"endTime":e,"limit":500}
            try:
                d=binance_get(url,p)
                if d: out+=d
                ms=e+1; time.sleep(0.25)
            except Exception:
                ms=e+1
        if not out: continue
        df=pd.DataFrame(out)
        df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms",utc=True)
        df["longShortRatio"]=pd.to_numeric(df["longShortRatio"],errors="coerce")
        tag="glsr" if ep.startswith("global") else "tlsr"
        df.rename(columns={"longShortRatio":tag},inplace=True)
        frames.append(df[["timestamp",tag]])
    if not frames: return pd.DataFrame(columns=["timestamp","glsr","tlsr"])
    m=None

# Calcula indicadores técnicos y agrega columnas al DataFrame
    for f in frames:
        m=f if m is None else pd.merge_asof(
            m.sort_values("timestamp"),
            f.sort_values("timestamp"),
            on="timestamp",

# Calcula el z-score móvil de una serie
            direction="nearest",
            tolerance=pd.Timedelta("65min"))
    return m.sort_values("timestamp")

# Descarga los ratios long/short (global y top) históricos (1h) para el símbolo


def get_or_fetch_context(sym_code, start_ms, end_ms, use_context=True, cache_dir=None):
    # Debug: imprime las fechas legibles de los milisegundos
    from datetime import datetime
    def ms_to_str(ts):
        if ts > 1e12:
            ts /= 1000
        return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[DEBUG] start_ms: {start_ms} ({ms_to_str(start_ms)}), end_ms: {end_ms} ({ms_to_str(end_ms)})")
    if not use_context:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    os.makedirs(cache_dir, exist_ok=True)
    # Ajusta los rangos para la API (por defecto 30 días)
    start_ms, end_ms = clamp_to_api_range(start_ms, end_ms, max_days=30)
    print(f"[DEBUG] Rango ajustado para API: start_ms={start_ms}, end_ms={end_ms}, símbolo={sym_code}")
    fpath = cache_path(cache_dir, sym_code, "funding", start_ms, end_ms)
    opath = cache_path(cache_dir, sym_code, "open_interest", start_ms, end_ms)
    lpath = cache_path(cache_dir, sym_code, "longshort", start_ms, end_ms)
    funding = load_cache(fpath)
    oi      = load_cache(opath)
    ls      = load_cache(lpath)
    if funding is None:
        funding = fetch_funding(sym_code, start_ms, end_ms); save_cache(funding, fpath)
    if oi is None:
        # Para OI, usa siempre los últimos 30 días desde ahora
        now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        oi_start = now_ms - 30 * 24 * 60 * 60 * 1000
        oi_end = now_ms
        print(f"[DEBUG] OI usando rango real: {oi_start} ({datetime.utcfromtimestamp(oi_start/1000).strftime('%Y-%m-%d %H:%M:%S')}), {oi_end} ({datetime.utcfromtimestamp(oi_end/1000).strftime('%Y-%m-%d %H:%M:%S')})")
        oi = fetch_open_interest_1h(sym_code, oi_start, oi_end); save_cache(oi, opath)
    if ls is None:
        print(f"[DEBUG] Llamando a fetch_long_short_1h con start_ms={start_ms}, end_ms={end_ms}, símbolo={sym_code}")
        ls = fetch_long_short_1h(sym_code, start_ms, end_ms); save_cache(ls, lpath)
    return funding if funding is not None else pd.DataFrame(), \
           oi if oi is not None else pd.DataFrame(), \
           ls if ls is not None else pd.DataFrame()

# Obtiene o descarga el contexto de mercado (funding, OI, LS) usando caché

def add_session_features(df, tz='UTC'):
    ts = df["timestamp"]
    dt = ts.dt.tz_localize('UTC') if ts.dt.tz is None else ts
    dt = dt.dt.tz_convert(tz)
    df["dow"]  = dt.dt.weekday
    df["hour"] = dt.dt.hour

# Realiza el merge de contexto y aplica fallback si falta información
    df["sess_asia"] = ((df["hour"]>=0)  & (df["hour"]<8)).astype(int)
    df["sess_eu"]   = ((df["hour"]>=7)  & (df["hour"]<15)).astype(int)
    df["sess_us"]   = ((df["hour"]>=12) & (df["hour"]<20)).astype(int)
    return df

def add_vol_regime(df):
    p20, p80 = df["ATR_pct"].quantile([0.2, 0.8])
    df["vr_low"]  = (df["ATR_pct"] <= p20).astype(int)
    df["vr_high"] = (df["ATR_pct"] >= p80).astype(int)
    df["vr_mid"]  = 1 - df[["vr_low","vr_high"]].max(axis=1)
    return df

def add_indicators(df):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["EMA7"]  = ta.ema(df["close"], 7)
    df["EMA25"] = ta.ema(df["close"], 25)

    df["EMA99"] = ta.ema(df["close"], 99)
    df["RSI14"] = ta.rsi(df["close"], 14)
    adx = ta.adx(df["high"], df["low"], df["close"], 14)
    df["ADX14"] = adx["ADX_14"]
    df["ATR"]   = ta.atr(df["high"], df["low"], df["close"], 14)

    df["ATR_pct"] = (df["ATR"] / df["close"]).clip(lower=0) * 100
    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None and not bb.empty:
        mid = [c for c in bb.columns if "BBM" in c][0]
        up  = [c for c in bb.columns if "BBU" in c][0]
        low = [c for c in bb.columns if "BBL" in c][0]
        df["BB_bbm"], df["BB_bbh"], df["BB_bbl"] = bb[mid], bb[up], bb[low]
        df["BB_width"] = ((df["BB_bbh"] - df["BB_bbl"]) / df["BB_bbm"]) * 100
        df["BB_%B"]    = (df["close"] - df["BB_bbl"]) / (df["BB_bbh"] - df["BB_bbl"])
    else:
        df["BB_bbm"] = df["BB_bbh"] = df["BB_bbl"] = df["BB_width"] = df["BB_%B"] = np.nan
    di = df.set_index("timestamp")
    vwap = ta.vwap(di["high"], di["low"], di["close"], di["volume"])
    df["VWAP"] = vwap.reindex(di.index).values
    df["dist_VWAP_pct"] = ((df["close"] - df["VWAP"]) / df["VWAP"]) * 100
    df["trend_long"]  = (df["EMA7"] > df["EMA25"]).astype(int)
    df["trend_short"] = (df["EMA7"] < df["EMA25"]).astype(int)
    df = add_session_features(df, tz='UTC')
    df = add_vol_regime(df)

    return df


def _zscore(s, win=96, minp=10):
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - m) / (sd + 1e-9)


def ensure_context_fallback(df5, funding, oi, ls):
    out = df5.copy().sort_values("timestamp")
    if funding is None or funding.empty:
        out["fundingRate"] = 0.0
    else:
        out = pd.merge_asof(out, funding.sort_values("timestamp"),
                            on="timestamp", direction="backward",
                            tolerance=pd.Timedelta("8h"))
        out["fundingRate"] = out["fundingRate"].fillna(0.0)
    if ls is None or ls.empty:
        tilt = (out["EMA7"] - out["EMA25"]) / (out["EMA25"].abs() + 1e-9)
        glsr = 1.0 + _zscore(tilt).clip(-2, 2) * 0.05
        out["glsr"] = glsr.fillna(np.nan)
        out["tlsr"] = out["glsr"]
    else:
        out = pd.merge_asof(out, oi.sort_values("timestamp"),
                            on="timestamp", direction="backward",
                            tolerance=pd.Timedelta("4h"))
        if "openInterest" not in out.columns:
            out["openInterest"] = np.nan
        if "tlsr" not in out.columns:
            out["tlsr"] = np.nan
        if "glsr" not in out.columns:
            out["glsr"] = np.nan
    # Advertencia si hay muchos NaN en glsr/tlsr
    nan_ratio = out["glsr"].isna().mean()
    if nan_ratio > 0.2:
        print(f"[ADVERTENCIA] Más del 20% de los valores de glsr son NaN. Revisa el rango de contexto disponible.")
    else:
        out = pd.merge_asof(out, ls.sort_values("timestamp"),
                            on="timestamp", direction="nearest",
                            tolerance=pd.Timedelta("65min"))
        out["glsr"] = out["glsr"].fillna(1.0)
        out["tlsr"] = out["tlsr"].fillna(1.0)
    out["fundingRate_z"] = _zscore(out["fundingRate"])
    if "openInterest" not in out.columns:
        out["openInterest"] = np.nan
    out["oi_change_pct"] = out["openInterest"].pct_change(fill_method=None).fillna(0) * 100
    out["glsr_z"] = _zscore(out["glsr"])
    out["tlsr_z"] = _zscore(out["tlsr"])
    return out

# Descarga los ratios long/short (global y top) históricos (1h) para el símbolo
def fetch_long_short_1h(symbol, start, end):
    # Solo descarga globalLongShortAccountRatio, rango limitado a 7 días para debug
    url = f"{BINANCE_FAPI}/futures/data/globalLongShortAccountRatio"
    step = 7 * 24 * 60 * 60 * 1000
    ms = start
    total = end - start
    all_data = []
    while ms <= end:
        e = min(end, ms + step - 1)
        p = {"symbol": symbol, "period": "1h", "startTime": ms, "endTime": e, "limit": 500}
        print(f"Solicitando LS {symbol}: desde {ms} hasta {e}")
        try:
            import requests
            r = requests.get(url, params=p, timeout=30)
            print(f"Status: {r.status_code}")
            if r.ok:
                data = r.json()
                print(f"Recibidos {len(data)} registros")
                all_data += data
            else:
                print(f"Error HTTP: {r.status_code} {r.text}")
        except Exception as ex:
            print(f"Excepción: {ex}")
        ms = e + 1
        time.sleep(2)
    print(f"Total registros descargados: {len(all_data)}")
    if all_data:
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["glsr"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
        df["tlsr"] = df["glsr"]
        return df[["timestamp", "glsr", "tlsr"]]
    return pd.DataFrame(columns=["timestamp", "glsr", "tlsr"])
                # ...existing code...

    def label_hits(df, k_tp, k_sl, H):
        close, high, low = df["close"].values, df["high"].values, df["low"].values
        atrp = df["ATR_pct"].values / 100.0
        labels = np.full(len(df), np.nan)
        for i in range(len(df)-H-1):
            entry = close[i+1]
            is_long = df["trend_long"].iloc[i+1] == 1
            tp, sl = k_tp*atrp[i+1], k_sl*atrp[i+1]
            hit = None
            for j in range(i+2, i+2+H):
                if j >= len(df):
                    break
                if is_long:
                    if low[j]  <= entry*(1-sl):
                        hit = 0
                        break
                    if high[j] >= entry*(1+tp):
                        hit = 1
                        break
                else:
                    if high[j] >= entry*(1+sl):
                        hit = 0
                        break
                    if low[j]  <= entry*(1-tp):
                        hit = 1
                        break
            labels[i+1] = np.nan if hit is None else hit
        return pd.Series(labels, index=df.index)

def ev_threshold(tp, sl, fee=TAKER_FEE, slip=SLIPPAGE):
    return (sl + 2*(fee+slip)) / (tp + sl)

def simulate(df, preds, k_tp, k_sl, H, p_min, ADX_MIN, DIST_VWAP_MAX,
             margin_usd=MARGIN_USD_DEFAULT, leverage=LEVERAGE_DEFAULT,
             avoid_vr_low=True, sessions_eu_us_only=True,
             cooldown_bars=4, use_half_time_exit=True, use_breakeven=False):
    trades = []
    c,h,l = df["close"].values, df["high"].values, df["low"].values
    atr   = df["ATR_pct"].values / 100.0
    long  = df["trend_long"].values
    ts    = df["timestamp"].values
    last_trade_i = -10**9
    for i in range(len(df)-H-1):
        if avoid_vr_low and df["vr_low"].iloc[i+1] == 1: continue
        if sessions_eu_us_only and not (df["sess_eu"].iloc[i+1] == 1 or df["sess_us"].iloc[i+1] == 1): continue
        if (i+1 - last_trade_i) < cooldown_bars: continue
        p = preds[i+1]
        if np.isnan(p): continue
        if df["ADX14"].iloc[i+1] < ADX_MIN: continue
        if abs(df["dist_VWAP_pct"].iloc[i+1]) > DIST_VWAP_MAX: continue
        p_gate = p_min
        if df["vr_high"].iloc[i+1] == 1 or df["ADX14"].iloc[i+1] >= 18: p_gate += 0.03
        if df["vr_mid"].iloc[i+1] == 1 and abs(df["dist_VWAP_pct"].iloc[i+1]) <= 0.3: p_gate -= 0.02
        p_gate = float(np.clip(p_gate, 0.45, 0.85))
        tp, sl = k_tp*atr[i+1], k_sl*atr[i+1]
        if p < max(p_gate, ev_threshold(tp, sl)): continue
        is_long = long[i+1] == 1
        e = c[i+1]
        tpv, slv = ((e*(1+tp), e*(1-sl)) if is_long else (e*(1-tp), e*(1+sl)))
        ex_price, ex_reason = None, None
        for j in range(i+2, i+2+H):
            if j>=len(df): break
            if is_long:
                if l[j] <= slv: ex_price=slv; ex_reason="SL"; break
                if h[j] >= tpv: ex_price=tpv; ex_reason="TP"; break
            else:
                if h[j] >= slv: ex_price=slv; ex_reason="SL"; break
                if l[j] <= tpv: ex_price=tpv; ex_reason="TP"; break
            if use_half_time_exit and j == (i+1 + H//2):
                if is_long and (h[j] < e*(1+0.3*atr[i+1])):
                    ex_price = c[j]; ex_reason="Tmid"; break
                if (not is_long) and (l[j] > e*(1-0.3*atr[i+1])):
                    ex_price = c[j]; ex_reason="Tmid"; break
        if ex_price is None:
            ex_reason="Time"; ex_price=c[min(i+1+H, len(df)-1)]
        qty   = (margin_usd * leverage) / e
        gross = (ex_price-e)*qty if is_long else (e-ex_price)*qty
        costs = (margin_usd * leverage) * (TAKER_FEE + SLIPPAGE) * 2
        pnl   = gross - costs
        trades.append({
            "timestamp_entry": pd.to_datetime(ts[i+1]),
            "side": "LONG" if is_long else "SHORT",
            "price_entry": float(e),
            "price_exit": float(ex_price),
            "p": float(p),
            "pnl_usd": float(pnl),
            "exit": ex_reason
        })
        last_trade_i = i+1
    if not trades:
        return trades, {"trades":0,"pnl":0,"dd":0,"dd_pct":0,"pf":0,"sh":0,"hr":0,"exp":0}
    pnl_s = pd.Series([t["pnl_usd"] for t in trades])
    eq_s  = pnl_s.cumsum()
    peak  = float(eq_s.cummax().max()) if len(eq_s) else 1.0
    dd    = float((eq_s - eq_s.cummax()).min())
    dd_pct = abs(dd) / max(1.0, peak if peak!=0 else 1.0)
    wins = pnl_s[pnl_s>0].sum()
    loss = -pnl_s[pnl_s<0].sum()
    pf = wins / loss if loss>0 else float("inf")
    s = np.array(pnl_s)
    sh = (s.mean()/s.std(ddof=1)*math.sqrt(288)) if s.std(ddof=1)>0 else 0.0
    return trades, {"trades": len(trades), "pnl": float(pnl_s.sum()), "dd": dd, "dd_pct": dd_pct, "pf": float(pf), "sh": float(sh), "hr": float((pnl_s>0).mean()), "exp": float(pnl_s.mean())}

def ccxt_pair(symbol_code):
    base = symbol_code.replace("USDT", "")
    return f"{base}/USDT:USDT"

def prepare_data(ex, sym_code, train_start, train_end, test_start, test_end, use_context=True, cache_dir=None):
    sym_pair = ccxt_pair(sym_code)
    print(f"   ⛓️  {sym_code}: descargando OHLCV 5m (train/test)...")
    tr_raw = fetch_ohlcv_all(ex, sym_pair, TIMEFRAME, utc_ms(train_start), utc_ms(train_end))
    te_raw = fetch_ohlcv_all(ex, sym_pair, TIMEFRAME, utc_ms(test_start),  utc_ms(test_end))
    if tr_raw.empty or te_raw.empty: raise RuntimeError(f"No candles for {sym_code}.")
    print("   🔧 Indicadores (train/test)...")
    tr = add_indicators(tr_raw); te = add_indicators(te_raw)
    eth_pair = ccxt_pair("ETHUSDT")
    print("   ⛓️  ETH para macro (train→test)...")
    eth = fetch_ohlcv_all(ex, eth_pair, TIMEFRAME, utc_ms(train_start), utc_ms(test_end))
    if eth.empty:
        eth_indic = pd.DataFrame({"timestamp": pd.concat([tr["timestamp"], te["timestamp"]]).drop_duplicates().sort_values()})
        eth_indic["close"]=np.nan; eth_indic["ATR"]=np.nan; eth_indic["EMA7"]=np.nan; eth_indic["EMA25"]=np.nan
    else:
        eth_indic = add_indicators(eth)
    # Ajusta los milisegundos para que nunca sean mayores al momento actual
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    s = min(utc_ms(train_start), now_ms)
    e = min(utc_ms(test_end), now_ms)
    print("   📥 Funding/OI/LS (con caché)...")
    funding, oi, ls = get_or_fetch_context(sym_code, s, e, use_context=use_context, cache_dir=cache_dir)
    print("   🔗 Merge contexto + fallbacks...")
    tr = ensure_context_fallback(tr, funding, oi, ls)
    te = ensure_context_fallback(te, funding, oi, ls)
    print("   📈 Macro pair (ETH)...")
    tr = add_macro_pair(tr, eth_indic)
    te = add_macro_pair(te, eth_indic)
    print("   🧱 Features + escalado...")
    Xtr_full = build_features(tr)
    Xte_full = build_features(te)
    scaler = StandardScaler()
    Xtr_mat = scaler.fit_transform(Xtr_full.fillna(0))
    Xte_mat = scaler.transform(Xte_full.fillna(0))
    Xtr = pd.DataFrame(Xtr_mat, columns=Xtr_full.columns, index=Xtr_full.index)
    Xte = pd.DataFrame(Xte_mat, columns=Xtr_full.columns, index=Xte_full.index)
    return tr, te, Xtr, Xte

def train_model(X, y):
    m = GradientBoostingClassifier(random_state=42)
    m.fit(X, y)
    return m

def kfold_auc(X, y, k=5):
    kf = KFold(k, shuffle=False)
    aucs = []
    for tr, va in kf.split(X):
        m = GradientBoostingClassifier(random_state=42)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:,1]
        try: aucs.append(roc_auc_score(y.iloc[va], p))
        except: pass
    return np.mean(aucs) if aucs else 0.5

def run_for_month(ex, sym_code, month, avoid_vr_low=True, sessions_eu_us_only=True, cooldown_bars=4, use_half_time_exit=True, use_breakeven=False, margin_usd=100.0, leverage=5.0, use_context=True, cache_dir=None):
    days = pd.Period(month).days_in_month
    test_start = f"{month}-01 00:00:00"
    test_end   = f"{month}-{days} 23:59:59"
    tr, te, Xtr, Xte = prepare_data(ex, sym_code, TRAIN_START, TRAIN_END, test_start, test_end, use_context=use_context, cache_dir=cache_dir)
    results, best_packs = [], []

    total_combos = len(GRID_H)*len(GRID_K_TP)*len(GRID_K_SL)*len(GRID_PMIN)*len(ADX_MIN_GRID)*len(DIST_VWAP_MAX_GRID)
    pbar = tqdm(total=total_combos, desc=f"Grid {sym_code} {month}", leave=False)

    for H in GRID_H:
        y = label_hits(tr, 1.25, 1.0, H)
        idx = y.dropna().index
        if len(idx) < 200:
            pbar.update(len(GRID_K_TP)*len(GRID_K_SL)*len(GRID_PMIN)*len(ADX_MIN_GRID)*len(DIST_VWAP_MAX_GRID))
            continue
        model = train_model(Xtr.loc[idx, FEATURE_COLS], y.loc[idx].astype(int))
        p_test = pd.Series(model.predict_proba(Xte.loc[:,FEATURE_COLS])[:,1], index=te.index)
        for k_tp in GRID_K_TP:
          for k_sl in GRID_K_SL:
            for pmin in GRID_PMIN:
              for adx_min in ADX_MIN_GRID:
                for dvw in DIST_VWAP_MAX_GRID:
                  trades, mx = simulate(
                      te, p_test.values, k_tp, k_sl, H, pmin,
                      ADX_MIN=adx_min, DIST_VWAP_MAX=dvw,
                      margin_usd=margin_usd, leverage=leverage,
                      avoid_vr_low=avoid_vr_low, sessions_eu_us_only=sessions_eu_us_only,
                      cooldown_bars=cooldown_bars, use_half_time_exit=use_half_time_exit, use_breakeven=use_breakeven
                  )
                  results.append({"H":H,"k_tp":k_tp,"k_sl":k_sl,"p_min":pmin,"ADX_min":adx_min,"dist_VWAP_max":dvw,**mx})
                  best_packs.append((mx["pnl"], {"H":H,"k_tp":k_tp,"k_sl":k_sl,"p_min":pmin,"ADX_min":adx_min,"dist_VWAP_max":dvw,"trades":trades,"metrics":mx}))
                  pbar.update(1)
    pbar.close()

    if not results:
        return pd.Series({"trades":0,"pnl":0.0}), []
    res_df = pd.DataFrame(results).sort_values("pnl", ascending=False).reset_index(drop=True)
    sel = res_df[(res_df["pf"] >= PF_MIN_SELECT) & (res_df["sh"] >= SHARPE_MIN_SELECT) & (res_df["dd_pct"] <= MAX_DD_PCT_SELECT)]
    best_row = sel.iloc[0] if not sel.empty else res_df.iloc[0]
    top1_pnl, top1_pack = sorted(best_packs, key=lambda x: x[0], reverse=True)[0]
    return best_row, top1_pack["trades"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
    ap.add_argument("--margin-usd", type=float, default=MARGIN_USD_DEFAULT)
    ap.add_argument("--leverage", type=float, default=5.0)
    ap.add_argument("--cooldown", type=int, default=4)
    ap.add_argument("--half-time-exit", action="store_true", default=True)
    ap.add_argument("--no-half-time-exit", action="store_true")
    ap.add_argument("--breakeven", action="store_true", default=False)
    ap.add_argument("--allow-vr-low", action="store_true")
    ap.add_argument("--all-sessions", action="store_true")
    ap.add_argument("--use-context", action="store_true", default=True)
    ap.add_argument("--context-cache-dir", type=str, default=os.path.expanduser("~/Desktop/tb_cache"))
    ap.add_argument("--outdir", type=str, default=os.path.expanduser("~/Desktop/tb_results"))
    args = ap.parse_args()

    out = os.path.expanduser(args.outdir)
    os.makedirs(out, exist_ok=True)
    cache_dir = os.path.expanduser(args.context_cache_dir)

    ex = get_exchange()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    months = TEST_MONTHS[:]


    total_tasks = len(symbols) * len(months)
    rows = []
    all_trades = {sym: [] for sym in symbols}
    start_time = time.time()
    completed_tasks = 0

    for sym in symbols:
        for m in months:
            print(f"\n========== [{sym}] Evaluating {m} (5m) ==========")
            task_start = time.time()
            best, trades = run_for_month(
                ex, sym, m,
                avoid_vr_low=not args.allow_vr_low,
                sessions_eu_us_only=not args.all_sessions,
                cooldown_bars=args.cooldown,
                use_half_time_exit=(not args.no_half_time_exit),
                use_breakeven=args.breakeven,
                margin_usd=args.margin_usd, leverage=args.leverage,
                use_context=args.use_context, cache_dir=cache_dir
            )
            rows.append({"symbol":sym,"month":m,**best.to_dict(),"margin_usd":args.margin_usd,"leverage":args.leverage})
            all_trades[sym].append(pd.DataFrame(trades))
            completed_tasks += 1
            elapsed = time.time() - start_time
            avg_per_task = elapsed / completed_tasks
            remaining = avg_per_task * (total_tasks - completed_tasks)
            percent = (completed_tasks / total_tasks) * 100
            print(f"Progreso: {completed_tasks}/{total_tasks} ({percent:.1f}%) | Tiempo estimado restante: {remaining/60:.1f} min")

    summary = pd.DataFrame(rows)
    combo = summary.groupby("month", as_index=False)[["pnl"]].sum().rename(columns={"pnl":"pnl_usd"})
    combo["trades"] = summary.groupby("month")["trades"].sum().values
    combo["pnl_usd_cum"] = combo["pnl_usd"].cumsum()

    combo.to_csv(os.path.join(out, "summary_5m_duo_AugSep_2025.csv"), index=False)
    summary.to_csv(os.path.join(out, "summary_5m_duo_AugSep_2025_by_symbol.csv"), index=False)

    for sym in symbols:
        sym_trades = pd.concat(all_trades[sym], ignore_index=True) if all_trades[sym] else pd.DataFrame(columns=["timestamp_entry","pnl_usd"])
        if not sym_trades.empty and "timestamp_entry" in sym_trades.columns:
            sym_trades["timestamp_entry"] = pd.to_datetime(sym_trades["timestamp_entry"])
            sym_trades["date"] = sym_trades["timestamp_entry"].dt.date
            daily = sym_trades.groupby("date", as_index=False)["pnl_usd"].agg(trades="count", pnl_total="sum")
            daily["positives"] = sym_trades.groupby("date")["pnl_usd"].apply(lambda s: (s>0).sum()).values
            daily["negatives"] = sym_trades.groupby("date")["pnl_usd"].apply(lambda s: (s<0).sum()).values
            daily["pnl_avg_per_trade"] = daily["pnl_total"]/daily["trades"]
            daily.sort_values("date").to_csv(os.path.join(out, f"daily_5m_{sym}_AugSep_2025.csv"), index=False)
            sym_trades.sort_values("timestamp_entry").to_csv(os.path.join(out, f"trades_5m_{sym}_AugSep_top1.csv"), index=False)

    print("\nSaved compact outputs to:", out)
    print("Context cached in:", cache_dir)

if __name__ == "__main__":
    main()
