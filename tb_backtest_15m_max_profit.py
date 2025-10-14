#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tommy Bulls - 15m Maximum Profit Strategy
Objetivo: Maximizar PNL neto con valor esperado positivo
Entrenar: Julio-Agosto | Probar: Septiembre
"""

import os, time, math, argparse
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
EXCHANGE_ID = "binanceusdm"
TIMEFRAME = "15m"
TRAIN_START = "2025-07-01 00:00:00"
TRAIN_END = "2025-08-31 23:59:59"
TEST_START = "2025-09-01 00:00:00"
TEST_END = "2025-09-30 23:59:59"

# Parámetros de trading
MARGIN_USD = 100.0
LEVERAGE = 5.0
TAKER_FEE = 0.0005
SLIPPAGE = 0.0002

# Grid de optimización para máximo PNL
GRID_K_TP_FULL = [1.0, 1.25, 1.5, 1.75, 2.0]
GRID_K_SL_FULL = [0.8, 1.0, 1.25, 1.5]
GRID_P_MIN_FULL = [0.55, 0.60, 0.65, 0.70, 0.75]
GRID_H_FULL = [4, 5, 6, 8, 10]  # Horizonte en velas de 15m (1-2.5h)
GRID_ADX_MIN_FULL = [15, 18, 20, 22]
GRID_DIST_VWAP_FULL = [0.5, 1.0, 1.5]  # %

# Filtros de contexto (on/off) para modo completo
GRID_REQUIRE_POS_FUNDING_FULL = [False, True]
GRID_REQUIRE_GLSR_ABOVE1_FULL = [False, True]

# Filtros reducidos para modo rápido
GRID_REQUIRE_POS_FUNDING_QUICK = [False]
GRID_REQUIRE_GLSR_ABOVE1_QUICK = [False]

# Valores reducidos para modo rápido
GRID_K_TP_QUICK = [1.25, 1.5]
GRID_K_SL_QUICK = [1.0, 1.25]
GRID_P_MIN_QUICK = [0.55, 0.6, 0.65]
GRID_H_QUICK = [4, 6, 8]
GRID_ADX_MIN_QUICK = [15, 18, 20]
GRID_DIST_VWAP_QUICK = [1.0, 1.5]

# Restricciones de selección
MAX_DD_PCT = 0.20  # 20%
MIN_PF = 1.3
MIN_SHARPE = 1.0

# Features para el modelo
FEATURE_COLS = [
    "close", "volume", "EMA7", "EMA25", "EMA99", "RSI14", "ADX14", "ATR_pct",
    "BB_width", "BB_%B", "VWAP", "dist_VWAP_pct", 
    "fundingRate", "fundingRate_z", "openInterest", "oi_change_pct",
    "glsr", "glsr_z", "tlsr", "tlsr_z",
    "ret_sym_15m", "ret_sym_1h", "ret_btc_15m", "ret_btc_1h",
    "btc_atr_pct", "sym_ema_slope", "btc_ema_slope",
    "dow", "hour", "sess_asia", "sess_eu", "sess_us",
    "vr_low", "vr_mid", "vr_high", "trend_long", "trend_short"
]

# ==================== FUNCIONES AUXILIARES ====================

def utc_ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)

def get_exchange():
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    ex.timeout = 20000
    # Cargar mercados para poder validar símbolos (Futures requieren sufijo :USDT)
    try:
        ex.load_markets()
    except Exception as e:
        print(f"[WARN] No se pudo cargar markets inicialmente: {e}")
    return ex

def fetch_ohlcv_all(ex, symbol, tf, start_ms, end_ms, limit=1000):
    """Descarga OHLCV completo"""
    out, ms = [], start_ms
    error_consec = 0
    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=ms, limit=limit)
            error_consec = 0
            if not batch:
                break
            out += batch
            ms = batch[-1][0] + 1
            if batch[-1][0] >= end_ms:
                break
            time.sleep(0.35)
        except Exception as e:
            print(f"[WARN] Error fetching OHLCV: {e}")
            # Si símbolo inválido, abortar inmediatamente
            if 'BadSymbol' in str(e):
                break
            error_consec += 1
            if error_consec > 5:
                print("[WARN] Demasiados errores consecutivos en descarga OHLCV, abortando bucle.")
                break
            time.sleep(min(5, 1 + error_consec))
            continue
    
    df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

def fetch_ohlcv_all_cached(ex, symbol, tf, start_ms, end_ms, cache_dir="data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"ohlcv_{symbol.replace('/','')}_{tf}_{start_ms}_{end_ms}.parquet"
    fpath = os.path.join(cache_dir, fname)
    if os.path.exists(fpath):
        try:
            df = pd.read_parquet(fpath)
            # Validar rango
            if df['timestamp'].min() <= pd.to_datetime(start_ms, unit='ms', utc=True) and df['timestamp'].max() >= pd.to_datetime(end_ms, unit='ms', utc=True):
                return df
        except Exception:
            pass
    df = fetch_ohlcv_all(ex, symbol, tf, start_ms, end_ms)
    try:
        df.to_parquet(fpath, index=False)
    except Exception:
        pass
    return df

def add_indicators(df):
    """Calcula indicadores técnicos"""
    d = df.copy()
    if "timestamp" in d.columns:
        d = d.sort_values("timestamp").reset_index(drop=True)
    
    # EMAs
    d["EMA7"] = ta.ema(d["close"], length=7)
    d["EMA25"] = ta.ema(d["close"], length=25)
    d["EMA99"] = ta.ema(d["close"], length=99)
    
    # Osciladores
    d["RSI14"] = ta.rsi(d["close"], length=14)
    adx = ta.adx(d["high"], d["low"], d["close"], length=14)
    d["ADX14"] = adx["ADX_14"] if "ADX_14" in adx.columns else np.nan
    
    # Volatilidad
    atr = ta.atr(d["high"], d["low"], d["close"], length=14)
    d["ATR"] = atr
    d["ATR_pct"] = (atr / d["close"] * 100).replace([np.inf, -np.inf], np.nan)
    
    # Bollinger
    bb = ta.bbands(d["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        # Detectar nombres de columnas (pueden variar según versión)
        cols = bb.columns.tolist()
        mid_col = [c for c in cols if 'BBM' in c][0] if any('BBM' in c for c in cols) else cols[0]
        up_col = [c for c in cols if 'BBU' in c][0] if any('BBU' in c for c in cols) else cols[1]
        low_col = [c for c in cols if 'BBL' in c][0] if any('BBL' in c for c in cols) else cols[2]
        d["BB_width"] = (bb[up_col] - bb[low_col]) / bb[mid_col] * 100
        d["BB_%B"] = (d["close"] - bb[low_col]) / (bb[up_col] - bb[low_col])
    else:
        d["BB_width"] = np.nan
        d["BB_%B"] = np.nan
    
    # VWAP (usar DatetimeIndex para evitar warnings de la librería)
    try:
        di = d.set_index("timestamp") if "timestamp" in d.columns else d
        vwap_series = ta.vwap(di["high"], di["low"], di["close"], di["volume"])  # requiere DatetimeIndex
        # Alinear por posición tras haber ordenado por timestamp
        d["VWAP"] = np.array(vwap_series)
    except Exception:
        # Fallback simple
        d["VWAP"] = ta.vwap(d["high"], d["low"], d["close"], d["volume"])
    d["dist_VWAP_pct"] = ((d["close"] - d["VWAP"]) / d["VWAP"] * 100).replace([np.inf, -np.inf], np.nan)
    
    # Tendencia
    d["trend_long"] = ((d["EMA7"] > d["EMA25"]) & (d["close"] > d["EMA7"])).astype(int)
    d["trend_short"] = ((d["EMA7"] < d["EMA25"]) & (d["close"] < d["EMA7"])).astype(int)
    
    # Sesiones y tiempo
    d["dow"] = d["timestamp"].dt.dayofweek
    d["hour"] = d["timestamp"].dt.hour
    d["sess_asia"] = ((d["hour"] >= 0) & (d["hour"] < 8)).astype(int)
    d["sess_eu"] = ((d["hour"] >= 8) & (d["hour"] < 16)).astype(int)
    d["sess_us"] = ((d["hour"] >= 16) & (d["hour"] < 24)).astype(int)
    
    # Volatilidad realizada
    atr_pctl = d["ATR_pct"].rolling(96, min_periods=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    d["vr_low"] = (atr_pctl < 0.33).astype(int)
    d["vr_mid"] = ((atr_pctl >= 0.33) & (atr_pctl < 0.67)).astype(int)
    d["vr_high"] = (atr_pctl >= 0.67).astype(int)
    
    return d.ffill().bfill()

def add_macro_context(df_sym, df_btc):
    """Agrega variables macro de BTC"""
    out = df_sym.copy()
    
    # Merge BTC indicators
    btc_cols = df_btc[["timestamp", "close", "ATR", "EMA7", "EMA25"]].rename(
        columns={"close": "btc_close", "ATR": "btc_ATR", "EMA7": "btc_EMA7", "EMA25": "btc_EMA25"}
    )
    out = out.merge(btc_cols, on="timestamp", how="left").ffill()
    
    # Retornos
    out["ret_sym_15m"] = out["close"].pct_change(fill_method=None)
    out["ret_sym_1h"] = out["close"].pct_change(4, fill_method=None)  # 4 velas de 15m = 1h
    out["ret_btc_15m"] = out["btc_close"].pct_change(fill_method=None)
    out["ret_btc_1h"] = out["btc_close"].pct_change(4, fill_method=None)
    
    # BTC ATR y pendiente
    out["btc_atr_pct"] = (out["btc_ATR"] / out["btc_close"] * 100).replace([np.inf, -np.inf], np.nan)
    out["sym_ema_slope"] = out["EMA7"] - out["EMA25"]
    out["btc_ema_slope"] = out["btc_EMA7"] - out["btc_EMA25"]
    
    return out

def add_futures_context(df, funding, oi, ls):
    """Agrega funding, OI, LS"""
    out = df.copy().sort_values("timestamp")
    
    # Funding
    if funding is None or funding.empty:
        out["fundingRate"] = 0.0
    else:
        out = pd.merge_asof(out, funding.sort_values("timestamp"),
                            on="timestamp", direction="backward",
                            tolerance=pd.Timedelta("8h"))
        out["fundingRate"] = out["fundingRate"].fillna(0.0)
    
    # Open Interest
    if oi is None or oi.empty:
        out["openInterest"] = np.nan
    else:
        out = pd.merge_asof(out, oi.sort_values("timestamp"),
                            on="timestamp", direction="backward",
                            tolerance=pd.Timedelta("4h"))
        if "openInterest" not in out.columns:
            out["openInterest"] = np.nan
    
    # Long/Short Ratio
    if ls is None or ls.empty:
        out["glsr"] = 1.0
        out["tlsr"] = 1.0
    else:
        out = pd.merge_asof(out, ls.sort_values("timestamp"),
                            on="timestamp", direction="nearest",
                            tolerance=pd.Timedelta("65min"))
        if "glsr" not in out.columns:
            out["glsr"] = np.nan
        if "tlsr" not in out.columns:
            out["tlsr"] = np.nan
        out["glsr"] = out["glsr"].fillna(1.0)
        out["tlsr"] = out["tlsr"].fillna(1.0)
    
    # Z-scores
    def _zscore(s, win=96, minp=10):
        m = s.rolling(win, min_periods=minp).mean()
        sd = s.rolling(win, min_periods=minp).std()
        return (s - m) / (sd + 1e-9)
    
    out["fundingRate_z"] = _zscore(out["fundingRate"])
    out["oi_change_pct"] = out["openInterest"].pct_change(fill_method=None).fillna(0) * 100
    out["glsr_z"] = _zscore(out["glsr"])
    out["tlsr_z"] = _zscore(out["tlsr"])
    
    return out

def label_hits(df, k_tp, k_sl, H):
    """Etiqueta hits de TP vs SL"""
    close, high, low = df["close"].values, df["high"].values, df["low"].values
    atrp = df["ATR_pct"].values / 100.0
    labels = np.full(len(df), np.nan)
    
    for i in range(len(df) - H - 1):
        entry = close[i + 1]
        is_long = df["trend_long"].iloc[i + 1] == 1
        tp, sl = k_tp * atrp[i + 1], k_sl * atrp[i + 1]
        hit = None
        
        for j in range(i + 2, i + 2 + H):
            if j >= len(df):
                break
            if is_long:
                if low[j] <= entry * (1 - sl):
                    hit = 0
                    break
                if high[j] >= entry * (1 + tp):
                    hit = 1
                    break
            else:
                if high[j] >= entry * (1 + sl):
                    hit = 0
                    break
                if low[j] <= entry * (1 - tp):
                    hit = 1
                    break
        
        labels[i + 1] = np.nan if hit is None else hit
    
    return pd.Series(labels, index=df.index)

def build_features(df):
    """Construye matriz de features"""
    X = df[FEATURE_COLS].shift(1)
    # Reemplazar inf y llenar NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    # Imputar NaNs: forward fill primero, luego backward, luego 0
    X = X.ffill().bfill().fillna(0)
    return X

def simulate_trades(df, prob, k_tp, k_sl, H, p_min, adx_min, dist_vwap_max, require_pos_funding=False, require_glsr_above1=False):
    """Simula trades y calcula PNL"""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = df["ATR_pct"].values / 100.0
    adx = df["ADX14"].values
    dvwap = df["dist_VWAP_pct"].abs().values
    funding_arr = df.get("fundingRate", pd.Series([0]*len(df))).values
    glsr_arr = df.get("glsr", pd.Series([1]*len(df))).values
    ts = df["timestamp"].values
    
    trades = []
    last_i = -999
    
    for i in range(len(df) - H - 1):
        if i <= last_i + 4:  # Cooldown 4 velas
            continue
        
        p = prob[i]
        if p < p_min or adx[i + 1] < adx_min or dvwap[i + 1] > dist_vwap_max:
            continue
        if require_pos_funding and funding_arr[i + 1] <= 0:
            continue
        if require_glsr_above1 and glsr_arr[i + 1] <= 1.0:
            continue
        
        is_long = df["trend_long"].iloc[i + 1] == 1
        if not is_long and df["trend_short"].iloc[i + 1] != 1:
            continue
        
        entry = c[i + 1]
        atr_entry = atr[i + 1]
        tp_val = k_tp * atr_entry
        sl_val = k_sl * atr_entry
        
        # Valor esperado
        ev = p * tp_val - (1 - p) * sl_val - (TAKER_FEE + SLIPPAGE) * 2
        if ev <= 0:
            continue
        
        # Simular salida
        exit_price = None
        exit_reason = None
        
        for j in range(i + 2, i + 2 + H):
            if j >= len(df):
                break
            if is_long:
                if l[j] <= entry * (1 - sl_val):
                    exit_price = entry * (1 - sl_val)
                    exit_reason = "SL"
                    break
                if h[j] >= entry * (1 + tp_val):
                    exit_price = entry * (1 + tp_val)
                    exit_reason = "TP"
                    break
            else:
                if h[j] >= entry * (1 + sl_val):
                    exit_price = entry * (1 + sl_val)
                    exit_reason = "SL"
                    break
                if l[j] <= entry * (1 - tp_val):
                    exit_price = entry * (1 - tp_val)
                    exit_reason = "TP"
                    break
        
        if exit_price is None:
            exit_price = c[min(i + 1 + H, len(df) - 1)]
            exit_reason = "Time"
        
        # Calcular PNL
        qty = (MARGIN_USD * LEVERAGE) / entry
        gross = (exit_price - entry) * qty if is_long else (entry - exit_price) * qty
        costs = (MARGIN_USD * LEVERAGE) * (TAKER_FEE + SLIPPAGE) * 2
        pnl = gross - costs
        
        trades.append({
            "timestamp": pd.to_datetime(ts[i + 1]),
            "side": "LONG" if is_long else "SHORT",
            "entry": float(entry),
            "exit": float(exit_price),
            "prob": float(p),
            "ev": float(ev),
            "pnl": float(pnl),
            "reason": exit_reason,
            "k_tp": k_tp,
            "k_sl": k_sl,
            "H": H,
            "atr_pct_entry": float(atr_entry*100),
            "tp_pct": float(tp_val*100),
            "sl_pct": float(sl_val*100),
            "tp_price": float(entry * (1 + tp_val if is_long else 1 - tp_val)),
            "sl_price": float(entry * (1 - sl_val if is_long else 1 + sl_val)),
            "horizon_end_ts": pd.to_datetime(ts[min(i + 1 + H, len(df) - 1)])
        })
        last_i = i + 1
    
    if not trades:
        return [], {"trades": 0, "pnl": 0, "dd": 0, "dd_pct": 0, "pf": 0, "sh": 0, "hr": 0, "exp": 0}
    
    pnl_s = pd.Series([t["pnl"] for t in trades])
    eq_s = pnl_s.cumsum()
    peak = float(eq_s.cummax().max()) if len(eq_s) else 1.0
    dd = float((eq_s - eq_s.cummax()).min())
    dd_pct = abs(dd) / max(1.0, peak if peak != 0 else 1.0)
    wins = pnl_s[pnl_s > 0].sum()
    loss = -pnl_s[pnl_s < 0].sum()
    pf = wins / loss if loss > 0 else float("inf")
    s = np.array(pnl_s)
    sh = (s.mean() / s.std(ddof=1) * math.sqrt(96)) if s.std(ddof=1) > 0 else 0.0  # 96 velas/día en 15m
    
    return trades, {
        "trades": len(trades),
        "pnl": float(pnl_s.sum()),
        "dd": dd,
        "dd_pct": dd_pct,
        "pf": float(pf),
        "sh": float(sh),
        "hr": float((pnl_s > 0).mean()),
        "exp": float(pnl_s.mean())
    }

print("🚀 Tommy Bulls 15m - Maximum Profit Strategy")
print("="*60)
print(f"Entrenamiento: {TRAIN_START} → {TRAIN_END}")
print(f"Prueba: {TEST_START} → {TEST_END}")
print(f"Margen: ${MARGIN_USD} | Apalancamiento: {LEVERAGE}x")
print("="*60)

############################
# CONTEXTO FUTURES (Funding, OI, Long/Short)
############################
import requests

BINANCE_FAPI = "https://fapi.binance.com"

def http_get_with_retries(url, params, max_retries=5, backoff=1.5, timeout=25):
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(backoff * (i+1)); continue
            r.raise_for_status(); return r.json()
        except Exception as ex:
            if i == max_retries-1: raise
            time.sleep(backoff * (i+1))

def fetch_funding(symbol, start_ms, end_ms):
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    out = []
    p = {"symbol": symbol, "limit": 1000, "startTime": start_ms, "endTime": end_ms}
    while True:
        d = http_get_with_retries(url, p, 5, 1.8, 25)
        if not d: break
        out += d
        if len(d) < 1000: break
        p["startTime"] = d[-1]["fundingTime"] + 1
        time.sleep(0.2)
    if not out: return pd.DataFrame(columns=["timestamp","fundingRate"])
    df = pd.DataFrame(out)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["timestamp","fundingRate"]]

def fetch_open_interest_1h(symbol, start_ms, end_ms):
    # Notar: Binance limita rango histórico práctico; aceptamos NaNs fuera de ventana real descargada
    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    step = 24*60*60*1000
    out = []; ms = start_ms
    while ms <= end_ms:
        e = min(end_ms, ms + step - 1)
        p = {"symbol": symbol, "period": "1h", "startTime": ms, "endTime": e, "limit": 500}
        try:
            d = http_get_with_retries(url, p, 4, 1.5, 25)
            if d: out += d
        except Exception:
            pass
        ms = e + 1; time.sleep(0.25)
    if not out: return pd.DataFrame(columns=["timestamp","openInterest"])
    df = pd.DataFrame(out)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    val = "sumOpenInterestValue" if "sumOpenInterestValue" in df.columns else "sumOpenInterest"
    df["openInterest"] = pd.to_numeric(df[val], errors="coerce")
    return df[["timestamp","openInterest"]]

def fetch_long_short_1h(symbol, start_ms, end_ms):
    frames=[]
    for ep in ["globalLongShortAccountRatio","topLongShortAccountRatio"]:
        url=f"{BINANCE_FAPI}/futures/data/{ep}"
        step=7*24*60*60*1000; ms=start_ms; out=[]
        while ms<=end_ms:
            e=min(end_ms, ms+step-1)
            p={"symbol":symbol, "period":"1h", "startTime":ms, "endTime":e, "limit":500}
            try:
                d=http_get_with_retries(url,p,4,1.5,25)
                if d: out+=d
            except Exception:
                pass
            ms=e+1; time.sleep(0.25)
        if not out: continue
        df=pd.DataFrame(out)
        df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms",utc=True)
        df["longShortRatio"]=pd.to_numeric(df["longShortRatio"],errors="coerce")
        tag="glsr" if ep.startswith("global") else "tlsr"
        df.rename(columns={"longShortRatio":tag},inplace=True)
        frames.append(df[["timestamp",tag]])
    if not frames: return pd.DataFrame(columns=["timestamp","glsr","tlsr"])
    m=None
    for f in frames:
        m = f if m is None else pd.merge_asof(m.sort_values("timestamp"), f.sort_values("timestamp"), on="timestamp", direction="nearest", tolerance=pd.Timedelta("65min"))
    return m.sort_values("timestamp")

def get_or_fetch_context(symbol, start_ms, end_ms, cache_dir="context_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    def cp(name): return os.path.join(cache_dir, f"{name}_{symbol}_{start_ms}_{end_ms}.parquet")
    def load(p):
        try:
            return pd.read_parquet(p) if os.path.exists(p) else None
        except Exception:
            return None
    def save(df,p):
        try:
            if df is not None and not df.empty: df.to_parquet(p, index=False)
        except Exception: pass
    fpath, opath, lpath = cp("funding"), cp("oi"), cp("ls")
    funding = load(fpath)
    if funding is None:
        funding = fetch_funding(symbol, start_ms, end_ms); save(funding, fpath)
    oi = load(opath)
    if oi is None:
        # Intentar solo último rango de 30 días desde ahora para OI (limitación API)
        now_ms = int(pd.Timestamp.utcnow().timestamp()*1000)
        oi = fetch_open_interest_1h(symbol, now_ms - 30*24*60*60*1000, now_ms); save(oi, opath)
    ls = load(lpath)
    if ls is None:
        ls = fetch_long_short_1h(symbol, start_ms, end_ms); save(ls, lpath)
    return funding, oi, ls

def merge_context(df, funding, oi, ls):
    out = df.copy().sort_values("timestamp")
    # Funding
    if funding is not None and not funding.empty:
        out = pd.merge_asof(out, funding.sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta("8h"))
    if "fundingRate" not in out.columns: out["fundingRate"] = 0.0
    out["fundingRate"] = out["fundingRate"].fillna(0.0)
    # OI
    if oi is not None and not oi.empty:
        out = pd.merge_asof(out, oi.sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta("4h"))
    if "openInterest" not in out.columns: out["openInterest"] = np.nan
    # LS
    if ls is not None and not ls.empty:
        out = pd.merge_asof(out, ls.sort_values("timestamp"), on="timestamp", direction="nearest", tolerance=pd.Timedelta("65min"))
    if "glsr" not in out.columns: out["glsr"] = 1.0
    if "tlsr" not in out.columns: out["tlsr"] = 1.0
    # Z
    def _z(s, win=96, minp=20):
        m = s.rolling(win, min_periods=minp).mean(); sd = s.rolling(win, min_periods=minp).std()
        return (s - m)/(sd + 1e-9)
    out["fundingRate_z"] = _z(out["fundingRate"])    
    out["oi_change_pct"] = out["openInterest"].pct_change(fill_method=None).fillna(0)*100
    out["glsr_z"] = _z(out["glsr"])
    out["tlsr_z"] = _z(out["tlsr"])
    return out

############################
# ENTRENAMIENTO Y OPTIMIZACIÓN
############################

def train_model(X_tr, y_tr, sample_weight=None):
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    return clf

def _build_class_weights(y_tr):
    counts = y_tr.value_counts()
    total = len(y_tr)
    weight_map = {cls: total/(len(counts)*cnt) for cls,cnt in counts.items()}
    return y_tr.map(weight_map).values

def _platt_calibration_build(raw_hold, y_hold):
    if len(np.unique(y_hold)) < 2:
        return None
    lr = LogisticRegression(max_iter=1000)
    lr.fit(raw_hold.reshape(-1,1), y_hold)
    return lr

def evaluate_config(df_train, df_test, k_tp, k_sl, H, p_min, adx_min, dist_vwap_max, require_pos_funding, require_glsr_above1, return_prob=False, calibrate=True, class_weighting=True):
    # Etiquetas en train
    y = label_hits(df_train, k_tp, k_sl, H)
    y_train = y.dropna()
    X_train = build_features(df_train.loc[y_train.index])
    mask = ~y_train.isna()
    X_tr = X_train[mask]
    y_tr = y_train[mask].astype(int)
    if y_tr.nunique() < 2:
        return None  # No se puede entrenar
    scaler = StandardScaler(); X_trs = scaler.fit_transform(X_tr)
    sample_weight = _build_class_weights(y_tr) if class_weighting else None
    model = train_model(X_trs, y_tr, sample_weight=sample_weight)
    # Predicciones en test
    X_te = build_features(df_test)
    X_tes = scaler.transform(X_te)
    prob = model.predict_proba(X_tes)[:,1]
    # Calibración (Platt) usando hold-out interno
    if calibrate and len(X_trs) > 50:
        split = int(len(X_trs)*0.8)
        X_fit, X_hold = X_trs[:split], X_trs[split:]
        y_fit, y_hold = y_tr.iloc[:split], y_tr.iloc[split:]
        if len(np.unique(y_hold)) == 2:
            base_model = GradientBoostingClassifier(random_state=42)
            base_model.fit(X_fit, y_fit)
            raw_hold = base_model.predict_proba(X_hold)[:,1]
            lr_cal = _platt_calibration_build(raw_hold, y_hold.values)
            if lr_cal is not None:
                raw_test = base_model.predict_proba(X_tes)[:,1]
                prob = lr_cal.predict_proba(raw_test.reshape(-1,1))[:,1]
    trades, stats = simulate_trades(df_test, prob, k_tp, k_sl, H, p_min, adx_min, dist_vwap_max, require_pos_funding=require_pos_funding, require_glsr_above1=require_glsr_above1)
    stats.update({
        "k_tp":k_tp, "k_sl":k_sl, "H":H, "p_min":p_min, "adx_min":adx_min, "dist_vwap_max":dist_vwap_max,
        "require_pos_funding":require_pos_funding, "require_glsr_above1": require_glsr_above1,
        "trades_log":trades
    })
    # AUC test (solo donde existe label)
    try:
        y_test_label = label_hits(df_test, k_tp, k_sl, H)
        mask_te = ~y_test_label.isna()
        if mask_te.sum()>10 and len(np.unique(y_test_label[mask_te]))==2:
            stats["auc"] = roc_auc_score(y_test_label[mask_te].astype(int), prob[mask_te])
        else:
            stats["auc"] = np.nan
    except Exception:
        stats["auc"] = np.nan
    if return_prob:
        stats["prob_array"] = prob
    return stats

def select_best(results):
    # Filtrar por restricciones
    viable = [r for r in results if r and r["trades"]>0 and r["dd_pct"] <= MAX_DD_PCT and r["pf"] >= MIN_PF and r["sh"] >= MIN_SHARPE]
    if not viable:
        return None
    # Maximizar PNL
    viable.sort(key=lambda r: (r["pnl"]), reverse=True)
    return viable[0]

def normalize_symbol(ex, symbol):
    # Para binanceusdm los perpetuos suelen ser BASE/USDT:USDT
    if symbol in ex.markets: return symbol
    base = symbol.split('/')[0]
    candidates = [f"{base}/USDT:USDT", f"{base}/USDT"]
    for c in candidates:
        if c in ex.markets: return c
    # Buscar cualquier mercado que empiece por base y termine en :USDT
    for m in ex.markets:
        if m.startswith(base) and m.endswith(":USDT"): return m
    return symbol

def run(symbol="ETH/USDT", quick=False, calibrate=True, class_weighting=True, no_context=False):
    ex = get_exchange()
    symbol = normalize_symbol(ex, symbol)
    print(f"[INFO] Usando símbolo normalizado: {symbol}")
    print(f"[INFO] Descargando OHLCV {symbol} 15m...")
    start_ms = utc_ms(TRAIN_START)
    end_ms = utc_ms(TEST_END)
    df_sym = fetch_ohlcv_all_cached(ex, symbol, TIMEFRAME, start_ms, end_ms)
    print(f"[OK] Velas {len(df_sym)}")
    print("[INFO] Descargando OHLCV BTC/USDT para contexto macro...")
    btc_symbol = normalize_symbol(ex, "BTC/USDT")
    df_sym = add_indicators(df_sym)
    try:
        df_btc = fetch_ohlcv_all_cached(ex, btc_symbol, TIMEFRAME, start_ms, end_ms)
        df_btc = add_indicators(df_btc)
        df_sym = add_macro_context(df_sym, df_btc)
    except Exception as e:
        print(f"[WARN] No se pudo obtener BTC contexto ({e}). Continuando sin macro BTC.")
        # Crear columnas macro vacías controladas
        for col in ["ret_btc_15m","ret_btc_1h","btc_atr_pct","btc_ema_slope"]:
            df_sym[col] = 0.0
    # Contexto futures
    funding = oi = ls = None
    if not no_context:
        print("[INFO] Descargando contexto futures...")
        # Para APIs de contexto usar código base sin sufijo :USDT
        symbol_code = symbol.replace("/","").replace(":USDT","")
        funding, oi, ls = get_or_fetch_context(symbol_code, start_ms, end_ms)
    else:
        print("[INFO] Contexto futures desactivado (--no-context)")
    df_sym = add_futures_context(df_sym, funding, oi, ls)
    # Split train/test
    df_train = df_sym[(df_sym["timestamp"]>=pd.Timestamp(TRAIN_START, tz="UTC")) & (df_sym["timestamp"]<=pd.Timestamp(TRAIN_END, tz="UTC"))].reset_index(drop=True)
    df_test  = df_sym[(df_sym["timestamp"]>=pd.Timestamp(TEST_START, tz="UTC")) & (df_sym["timestamp"]<=pd.Timestamp(TEST_END, tz="UTC"))].reset_index(drop=True)
    print(f"[INFO] Train: {len(df_train)} filas | Test: {len(df_test)} filas")
    # Optimización
    results = []
    if quick:
        GRID_K_TP = GRID_K_TP_QUICK
        GRID_K_SL = GRID_K_SL_QUICK
        GRID_P_MIN = GRID_P_MIN_QUICK
        GRID_H = GRID_H_QUICK
        GRID_ADX_MIN = GRID_ADX_MIN_QUICK
        GRID_DIST_VWAP = GRID_DIST_VWAP_QUICK
        GRID_REQ_FUND = GRID_REQUIRE_POS_FUNDING_QUICK
        GRID_REQ_GLSR = GRID_REQUIRE_GLSR_ABOVE1_QUICK
    else:
        GRID_K_TP = GRID_K_TP_FULL
        GRID_K_SL = GRID_K_SL_FULL
        GRID_P_MIN = GRID_P_MIN_FULL
        GRID_H = GRID_H_FULL
        GRID_ADX_MIN = GRID_ADX_MIN_FULL
        GRID_DIST_VWAP = GRID_DIST_VWAP_FULL
        GRID_REQ_FUND = GRID_REQUIRE_POS_FUNDING_FULL
        GRID_REQ_GLSR = GRID_REQUIRE_GLSR_ABOVE1_FULL

    total_cfg = (len(GRID_K_TP)*len(GRID_K_SL)*len(GRID_H)*len(GRID_P_MIN)*
                 len(GRID_ADX_MIN)*len(GRID_DIST_VWAP)*len(GRID_REQ_FUND)*len(GRID_REQ_GLSR))
    cfg_i = 0
    for k_tp in GRID_K_TP:
        for k_sl in GRID_K_SL:
            for H in GRID_H:
                for p_min in GRID_P_MIN:
                    for adx_min in GRID_ADX_MIN:
                        for dist_v in GRID_DIST_VWAP:
                            for req_f in GRID_REQ_FUND:
                                for req_g in GRID_REQ_GLSR:
                                    cfg_i += 1
                                    print(f"[GRID] ({cfg_i}/{total_cfg}) k_tp={k_tp} k_sl={k_sl} H={H} p_min={p_min} adx_min={adx_min} distVWAP<={dist_v} posFund={req_f} glsr>1={req_g} cal={calibrate} cw={class_weighting}")
                                    stats = evaluate_config(df_train, df_test, k_tp, k_sl, H, p_min, adx_min, dist_v, req_f, req_g, calibrate=calibrate, class_weighting=class_weighting)
                                    if stats:
                                        results.append(stats)
    print(f"[INFO] Configuraciones evaluadas: {len(results)}")
    best = select_best(results)
    if not best:
        print("[WARN] Ninguna configuración cumple restricciones. Mostrando top-3 por PNL sin filtros.")
        results_sorted = sorted([r for r in results if r], key=lambda r: r['pnl'], reverse=True)
        for r in results_sorted[:3]:
            print(r)
        if results_sorted:
            best = results_sorted[0]
            print("[INFO] Continuando con la mejor por PNL para generar exports (sin cumplir restricciones).")
        else:
            # Aun así exportar grid vacío para traza
            pd.DataFrame(results).drop(columns=["trades_log"], errors="ignore").to_csv(f"grid_results_{symbol.replace('/','')}.csv", index=False)
            print(f"[OK] Resultados grid guardados en grid_results_{symbol.replace('/','')}.csv")
            return
    print("\n===== MEJOR CONFIGURACIÓN =====")
    # Métricas extendidas
    trades_list = best["trades_log"]
    pnl_list = [t["pnl"] for t in trades_list]
    wins_n = sum(1 for x in pnl_list if x>0)
    loss_n = sum(1 for x in pnl_list if x<=0)
    win_rate = wins_n / max(1, (wins_n+loss_n))
    avg_pnl = np.mean(pnl_list) if pnl_list else 0
    med_pnl = np.median(pnl_list) if pnl_list else 0
    # Máx racha pérdidas
    max_consec_loss = 0; cur = 0
    for x in pnl_list:
        if x<=0: cur+=1; max_consec_loss=max(max_consec_loss,cur)
        else: cur=0
    for k in ["k_tp","k_sl","H","p_min","adx_min","dist_vwap_max","trades","pnl","dd","dd_pct","pf","sh","hr","exp","auc"]:
        print(f"{k}: {best[k]}")
    print(f"win_rate: {win_rate:.3f}")
    print(f"avg_pnl: {avg_pnl:.2f}")
    print(f"median_pnl: {med_pnl:.2f}")
    print(f"max_consec_loss: {max_consec_loss}")
    # Preparar identificador de archivo seguro
    sym_id = symbol.replace('/', '').replace(':', '')
    # Guardar trades
    trades_df = pd.DataFrame(best["trades_log"]) if best["trades_log"] else pd.DataFrame()
    if not trades_df.empty:
        trades_df.to_csv(f"trades_15m_{sym_id}.csv", index=False)
        print(f"[OK] Trades guardados en trades_15m_{sym_id}.csv")
        # Equity curve export
        eq = pd.Series([t['pnl'] for t in best['trades_log']]).cumsum()
        eq_df = pd.DataFrame({"trade_index": range(1,len(eq)+1), "equity": eq})
        eq_df.to_csv(f"equity_curve_{sym_id}.csv", index=False)
        print(f"[OK] Equity curve guardada en equity_curve_{sym_id}.csv")
    # Re-entrenar mejor config para exportar probabilidades
    print("[INFO] Recalculando probabilidades para archivo de predicciones...")
    re_stats = evaluate_config(df_train, df_test, best['k_tp'], best['k_sl'], best['H'], best['p_min'], best['adx_min'], best['dist_vwap_max'], best.get('require_pos_funding', False), best.get('require_glsr_above1', False), return_prob=True, calibrate=calibrate, class_weighting=class_weighting)
    probs = re_stats.get("prob_array", [])
    preds_df = df_test[["timestamp","close","trend_long","trend_short","ADX14","ATR_pct","dist_VWAP_pct","fundingRate","glsr","tlsr","openInterest","oi_change_pct"]].copy()
    preds_df["prob"] = probs
    # EV estimado por fila
    atr_frac = preds_df["ATR_pct"] / 100.0
    preds_df["ev"] = preds_df["prob"] * (best['k_tp']*atr_frac) - (1 - preds_df["prob"]) * (best['k_sl']*atr_frac) - (TAKER_FEE+SLIPPAGE)*2
    preds_df["entry_signal"] = (
        (preds_df["prob"] >= best['p_min']) &
        (preds_df["ADX14"] >= best['adx_min']) &
        (preds_df["dist_VWAP_pct"].abs() <= best['dist_vwap_max']) &
        (preds_df["ev"] > 0) &
        ((preds_df["trend_long"]==1) | (preds_df["trend_short"]==1))
    ).astype(int)
    preds_df.to_csv(f"predictions_15m_{sym_id}.csv", index=False)
    print(f"[OK] Predicciones guardadas en predictions_15m_{sym_id}.csv")
    # Añadir label real para mejor análisis (usando misma función de etiquetado)
    print("[INFO] Calculando labels reales sobre test para archivo de predicciones...")
    labels_real = label_hits(df_test, best['k_tp'], best['k_sl'], best['H'])
    preds_df_with_label = preds_df.merge(df_test[['timestamp']].assign(label=labels_real.values), on='timestamp', how='left')
    preds_df_with_label.to_csv(f"predictions_15m_{sym_id}_with_labels.csv", index=False)
    print(f"[OK] Predicciones con labels guardadas en predictions_15m_{sym_id}_with_labels.csv")
    # Guardar resumen extendido
    summary = best.copy(); summary.pop('trades_log', None)
    summary.update({"win_rate":win_rate,"avg_pnl":avg_pnl,"median_pnl":med_pnl,"max_consec_loss":max_consec_loss, "auc":best.get("auc")})
    pd.DataFrame([summary]).to_csv(f"best_summary_{sym_id}.csv", index=False)
    print(f"[OK] Resumen extendido guardado en best_summary_{sym_id}.csv")
    # Guardar resumen
    pd.DataFrame(results).drop(columns=["trades_log"], errors="ignore").sort_values("pnl", ascending=False).to_csv(f"grid_results_{sym_id}.csv", index=False)
    print(f"[OK] Resultados grid guardados en grid_results_{sym_id}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETH/USDT", help="Símbolo objetivo Futures")
    parser.add_argument("--quick", action="store_true", help="Modo rápido (grid reducido)")
    parser.add_argument("--no-calibrate", action="store_true", help="Desactiva calibración Platt interna")
    parser.add_argument("--no-class-weighting", action="store_true", help="Desactiva weighting de clases")
    parser.add_argument("--no-context", action="store_true", help="Desactiva descarga y merge de contexto futures (funding, OI, LS)")
    args = parser.parse_args()
    run(args.symbol, quick=args.quick, calibrate=not args.no_calibrate, class_weighting=not args.no_class_weighting, no_context=args.no_context)

print("\n[INFO] Implementación completada. Ejecuta el script para iniciar la optimización.")
