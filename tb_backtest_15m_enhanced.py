#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tommy Bulls - 15m Enhanced Maximum Profit Strategy
Versión optimizada para maximizar PNL con mejores indicadores y filtros adaptativos
"""

import os, time, math, argparse
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
# import xgboost as xgb  # Comentado por si no está instalado
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN OPTIMIZADA ====================
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

# Grid optimizado más agresivo para encontrar oportunidades
GRID_K_TP_ENHANCED = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]  # Más variedad en TP
GRID_K_SL_ENHANCED = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]       # Más variedad en SL
GRID_P_MIN_ENHANCED = [0.51, 0.53, 0.55, 0.58, 0.60, 0.65, 0.70]  # Menos restrictivo
GRID_H_ENHANCED = [3, 4, 5, 6, 8, 10, 12, 15]              # Más horizontes
GRID_ADX_MIN_ENHANCED = [10, 12, 15, 18, 20, 25]           # Menos restrictivo en ADX
GRID_DIST_VWAP_ENHANCED = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]  # Más permisivo con distancia VWAP

# Filtros de contexto más permisivos
GRID_REQUIRE_POS_FUNDING_ENHANCED = [False, True]
GRID_REQUIRE_GLSR_ABOVE1_ENHANCED = [False, True]
GRID_VOLATILITY_FILTER_ENHANCED = [False, True]  # Nuevo filtro de volatilidad

# Restricciones relajadas para encontrar estrategias viables
MAX_DD_PCT = 0.35  # 35% (más permisivo)
MIN_PF = 1.1      # 1.1 (más permisivo) 
MIN_SHARPE = 0.5  # 0.5 (más permisivo)
MIN_TRADES = 5    # Mínimo 5 trades para validez estadística

# Features expandidas con más indicadores técnicos
FEATURE_COLS_ENHANCED = [
    # Precio y volumen básico
    "close", "volume", "high", "low", "open",
    
    # EMAs múltiples para capturar diferentes tendencias
    "EMA7", "EMA15", "EMA25", "EMA50", "EMA99", "EMA200",
    
    # Osciladores técnicos
    "RSI14", "RSI21", "ADX14", "ATR_pct", "CCI14", "MFI14",
    
    # Bandas de Bollinger expandidas
    "BB_width", "BB_%B", "BB_squeeze", 
    
    # VWAP y variaciones
    "VWAP", "dist_VWAP_pct", "vwap_slope",
    
    # Indicadores de momentum
    "MACD_signal", "MACD_hist", "ROC_10", "Williams_R",
    
    # Patrones de velas
    "doji", "hammer", "engulfing", "inside_bar",
    
    # Variables de contexto futures
    "fundingRate", "fundingRate_z", "fundingRate_momentum",
    "openInterest", "oi_change_pct", "oi_momentum",
    "glsr", "glsr_z", "glsr_momentum", 
    "tlsr", "tlsr_z", "tlsr_momentum",
    
    # Retornos multi-timeframe
    "ret_sym_15m", "ret_sym_1h", "ret_sym_4h", "ret_sym_1d",
    "ret_btc_15m", "ret_btc_1h", "ret_btc_4h", 
    
    # Volatilidad realizada multi-timeframe
    "vol_15m", "vol_1h", "vol_4h", "vol_ratio",
    "btc_atr_pct", "corr_btc_15m", "beta_btc",
    
    # Pendientes de EMAs para detectar momentum
    "sym_ema_slope", "btc_ema_slope", "ema_divergence",
    
    # Variables temporales expandidas
    "dow", "hour", "minute", "is_weekend",
    "sess_asia", "sess_eu", "sess_us", "sess_overlap",
    
    # Volatilidad realizada categorizada
    "vr_low", "vr_mid", "vr_high", "vr_extreme",
    
    # Señales de tendencia mejoradas
    "trend_long", "trend_short", "trend_strength", "trend_change",
    
    # Indicadores de microestructura 
    "bid_ask_spread_proxy", "order_flow_proxy", "pressure_buy", "pressure_sell"
]

# ==================== FUNCIONES AUXILIARES MEJORADAS ====================

def utc_ms(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)

def get_exchange():
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    ex.timeout = 30000  # Incrementar timeout
    try:
        ex.load_markets()
    except Exception as e:
        print(f"[WARN] No se pudo cargar markets inicialmente: {e}")
    return ex

def fetch_ohlcv_all_cached(ex, symbol, tf, start_ms, end_ms, cache_dir="data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"ohlcv_{symbol.replace('/','').replace(':','')}_{tf}_{start_ms}_{end_ms}.parquet"
    fpath = os.path.join(cache_dir, fname)
    if os.path.exists(fpath):
        try:
            df = pd.read_parquet(fpath)
            if df['timestamp'].min() <= pd.to_datetime(start_ms, unit='ms', utc=True) and df['timestamp'].max() >= pd.to_datetime(end_ms, unit='ms', utc=True):
                return df
        except Exception:
            pass
    
    # Fetch con retry mejorado
    out, ms = [], start_ms
    error_consec = 0
    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=ms, limit=1000)
            error_consec = 0
            if not batch:
                break
            out += batch
            ms = batch[-1][0] + 1
            if batch[-1][0] >= end_ms:
                break
            time.sleep(0.2)  # Más conservativo
        except Exception as e:
            print(f"[WARN] Error fetching OHLCV: {e}")
            if 'BadSymbol' in str(e) or 'Invalid symbol' in str(e):
                break
            error_consec += 1
            if error_consec > 5:
                print("[WARN] Demasiados errores consecutivos, abortando.")
                break
            time.sleep(min(10, 2 ** error_consec))
            continue
    
    df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    
    try:
        df.to_parquet(fpath, index=False)
    except Exception:
        pass
    return df

def add_enhanced_indicators(df):
    """Calcula indicadores técnicos expandidos"""
    d = df.copy()
    if "timestamp" in d.columns:
        d = d.sort_values("timestamp").reset_index(drop=True)
    
    # EMAs múltiples
    for period in [7, 15, 25, 50, 99, 200]:
        d[f"EMA{period}"] = ta.ema(d["close"], length=period)
    
    # Osciladores expandidos
    d["RSI14"] = ta.rsi(d["close"], length=14)
    d["RSI21"] = ta.rsi(d["close"], length=21)
    d["CCI14"] = ta.cci(d["high"], d["low"], d["close"], length=14)
    d["MFI14"] = ta.mfi(d["high"], d["low"], d["close"], d["volume"], length=14)
    d["Williams_R"] = ta.willr(d["high"], d["low"], d["close"], length=14)
    d["ROC_10"] = ta.roc(d["close"], length=10)
    
    # ADX
    adx = ta.adx(d["high"], d["low"], d["close"], length=14)
    d["ADX14"] = adx["ADX_14"] if "ADX_14" in adx.columns else np.nan
    
    # MACD
    macd = ta.macd(d["close"])
    if macd is not None and not macd.empty:
        cols = macd.columns.tolist()
        d["MACD_signal"] = macd[cols[1]] if len(cols) > 1 else np.nan
        d["MACD_hist"] = macd[cols[2]] if len(cols) > 2 else np.nan
    else:
        d["MACD_signal"] = np.nan
        d["MACD_hist"] = np.nan
    
    # ATR mejorado
    atr = ta.atr(d["high"], d["low"], d["close"], length=14)
    d["ATR"] = atr
    d["ATR_pct"] = (atr / d["close"] * 100).replace([np.inf, -np.inf], np.nan)
    
    # Bollinger Bands expandidas
    bb = ta.bbands(d["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        cols = bb.columns.tolist()
        mid_col = [c for c in cols if 'BBM' in c][0] if any('BBM' in c for c in cols) else cols[0]
        up_col = [c for c in cols if 'BBU' in c][0] if any('BBU' in c for c in cols) else cols[1]  
        low_col = [c for c in cols if 'BBL' in c][0] if any('BBL' in c for c in cols) else cols[2]
        d["BB_width"] = (bb[up_col] - bb[low_col]) / bb[mid_col] * 100
        d["BB_%B"] = (d["close"] - bb[low_col]) / (bb[up_col] - bb[low_col])
        # BB Squeeze (cuando las bandas se contraen)
        d["BB_squeeze"] = (d["BB_width"] < d["BB_width"].rolling(20).mean() * 0.8).astype(int)
    else:
        d["BB_width"] = np.nan
        d["BB_%B"] = np.nan
        d["BB_squeeze"] = 0
    
    # VWAP mejorado
    try:
        di = d.set_index("timestamp") if "timestamp" in d.columns else d
        vwap_series = ta.vwap(di["high"], di["low"], di["close"], di["volume"])
        d["VWAP"] = np.array(vwap_series)
    except Exception:
        d["VWAP"] = ta.vwap(d["high"], d["low"], d["close"], d["volume"])
    
    d["dist_VWAP_pct"] = ((d["close"] - d["VWAP"]) / d["VWAP"] * 100).replace([np.inf, -np.inf], np.nan)
    d["vwap_slope"] = d["VWAP"].pct_change(4)  # Pendiente VWAP
    
    # Patrones de velas básicos
    d["doji"] = (abs(d["close"] - d["open"]) / (d["high"] - d["low"] + 1e-10) < 0.1).astype(int)
    d["hammer"] = ((d["close"] > d["open"]) & ((d["close"] - d["low"]) > 2 * (d["high"] - d["close"])) & ((d["open"] - d["low"]) > 2 * (d["high"] - d["close"]))).astype(int)
    d["engulfing"] = ((d["close"] > d["open"]) & (d["close"].shift(1) < d["open"].shift(1)) & (d["close"] > d["open"].shift(1)) & (d["open"] < d["close"].shift(1))).astype(int)
    d["inside_bar"] = ((d["high"] < d["high"].shift(1)) & (d["low"] > d["low"].shift(1))).astype(int)
    
    # Tendencias mejoradas
    d["trend_long"] = ((d["EMA7"] > d["EMA25"]) & (d["close"] > d["EMA7"])).astype(int)
    d["trend_short"] = ((d["EMA7"] < d["EMA25"]) & (d["close"] < d["EMA7"])).astype(int)
    d["trend_strength"] = abs(d["EMA7"] - d["EMA25"]) / d["close"] * 100
    d["trend_change"] = ((d["trend_long"] != d["trend_long"].shift(1)) | (d["trend_short"] != d["trend_short"].shift(1))).astype(int)
    
    # Variables temporales expandidas
    d["dow"] = d["timestamp"].dt.dayofweek
    d["hour"] = d["timestamp"].dt.hour
    d["minute"] = d["timestamp"].dt.minute
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["sess_asia"] = ((d["hour"] >= 0) & (d["hour"] < 8)).astype(int)
    d["sess_eu"] = ((d["hour"] >= 8) & (d["hour"] < 16)).astype(int)
    d["sess_us"] = ((d["hour"] >= 16) & (d["hour"] < 24)).astype(int)
    d["sess_overlap"] = (((d["hour"] >= 7) & (d["hour"] < 9)) | ((d["hour"] >= 15) & (d["hour"] < 17))).astype(int)
    
    # Volatilidad realizada expandida
    atr_pctl = d["ATR_pct"].rolling(96, min_periods=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    d["vr_low"] = (atr_pctl < 0.25).astype(int)
    d["vr_mid"] = ((atr_pctl >= 0.25) & (atr_pctl < 0.75)).astype(int)
    d["vr_high"] = (atr_pctl >= 0.75).astype(int)
    d["vr_extreme"] = (atr_pctl >= 0.9).astype(int)
    
    # Retornos multi-timeframe
    d["ret_sym_15m"] = d["close"].pct_change(fill_method=None)
    d["ret_sym_1h"] = d["close"].pct_change(4, fill_method=None)  # 4 velas = 1h
    d["ret_sym_4h"] = d["close"].pct_change(16, fill_method=None)  # 16 velas = 4h
    d["ret_sym_1d"] = d["close"].pct_change(96, fill_method=None)  # 96 velas = 1d
    
    # Volatilidad multi-timeframe
    d["vol_15m"] = d["ret_sym_15m"].rolling(4).std()
    d["vol_1h"] = d["ret_sym_1h"].rolling(4).std()
    d["vol_4h"] = d["ret_sym_4h"].rolling(6).std()
    d["vol_ratio"] = d["vol_15m"] / (d["vol_1h"] + 1e-10)
    
    # Pendientes EMAs
    d["sym_ema_slope"] = (d["EMA7"] - d["EMA25"]) / d["close"] * 100
    d["ema_divergence"] = d["EMA7"].pct_change(4) - d["EMA25"].pct_change(4)
    
    # Proxies de microestructura (aproximaciones basadas en OHLCV)
    d["bid_ask_spread_proxy"] = (d["high"] - d["low"]) / d["close"] * 100
    d["order_flow_proxy"] = (d["close"] - d["open"]) / (d["high"] - d["low"] + 1e-10)
    d["pressure_buy"] = ((d["close"] > (d["high"] + d["low"]) / 2) & (d["volume"] > d["volume"].rolling(10).mean())).astype(int)
    d["pressure_sell"] = ((d["close"] < (d["high"] + d["low"]) / 2) & (d["volume"] > d["volume"].rolling(10).mean())).astype(int)
    
    return d.ffill().bfill()

def add_enhanced_macro_context(df_sym, df_btc):
    """Agrega contexto macro expandido de BTC"""
    out = df_sym.copy()
    
    # Merge BTC indicators
    btc_cols = df_btc[["timestamp", "close", "ATR", "EMA7", "EMA25", "EMA50"]].rename(
        columns={"close": "btc_close", "ATR": "btc_ATR", "EMA7": "btc_EMA7", 
                "EMA25": "btc_EMA25", "EMA50": "btc_EMA50"}
    )
    out = out.merge(btc_cols, on="timestamp", how="left").ffill()
    
    # Retornos BTC multi-timeframe
    out["ret_btc_15m"] = out["btc_close"].pct_change(fill_method=None)
    out["ret_btc_1h"] = out["btc_close"].pct_change(4, fill_method=None)
    out["ret_btc_4h"] = out["btc_close"].pct_change(16, fill_method=None)
    
    # BTC ATR y correlaciones
    out["btc_atr_pct"] = (out["btc_ATR"] / out["btc_close"] * 100).replace([np.inf, -np.inf], np.nan)
    out["btc_ema_slope"] = (out["btc_EMA7"] - out["btc_EMA25"]) / out["btc_close"] * 100
    
    # Correlación móvil con BTC
    out["corr_btc_15m"] = out["ret_sym_15m"].rolling(48).corr(out["ret_btc_15m"])
    
    # Beta con BTC (coeficiente de sensibilidad)
    def rolling_beta(ret_asset, ret_market, window=48):
        cov = ret_asset.rolling(window).cov(ret_market)
        var_market = ret_market.rolling(window).var()
        return cov / (var_market + 1e-10)
    
    out["beta_btc"] = rolling_beta(out["ret_sym_15m"], out["ret_btc_15m"])
    
    return out

def add_enhanced_futures_context(df, funding, oi, ls):
    """Agrega contexto de futuros mejorado con momentum"""
    out = df.copy().sort_values("timestamp")
    
    # Funding Rate con momentum
    if funding is None or funding.empty:
        out["fundingRate"] = 0.0
        out["fundingRate_momentum"] = 0.0
    else:
        out = pd.merge_asof(out, funding.sort_values("timestamp"),
                            on="timestamp", direction="backward",
                            tolerance=pd.Timedelta("8h"))
        out["fundingRate"] = out["fundingRate"].fillna(0.0)
        out["fundingRate_momentum"] = out["fundingRate"].pct_change(3).fillna(0.0)  # Momentum de 3 periodos
    
    # Open Interest con momentum
    if oi is None or oi.empty:
        out["openInterest"] = np.nan
        out["oi_momentum"] = 0.0
    else:
        out = pd.merge_asof(out, oi.sort_values("timestamp"),
                            on="timestamp", direction="backward",
                            tolerance=pd.Timedelta("4h"))
        if "openInterest" not in out.columns:
            out["openInterest"] = np.nan
        out["oi_momentum"] = out["openInterest"].pct_change(4).fillna(0.0)
    
    # Long/Short Ratio con momentum
    if ls is None or ls.empty:
        out["glsr"] = 1.0
        out["tlsr"] = 1.0
        out["glsr_momentum"] = 0.0
        out["tlsr_momentum"] = 0.0
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
        out["glsr_momentum"] = out["glsr"].pct_change(2).fillna(0.0)
        out["tlsr_momentum"] = out["tlsr"].pct_change(2).fillna(0.0)
    
    # Z-scores mejorados
    def _zscore_robust(s, win=96, minp=20):
        m = s.rolling(win, min_periods=minp).median()  # Usar mediana para robustez
        mad = (s - m).abs().rolling(win, min_periods=minp).median()  # MAD en lugar de std
        return (s - m) / (mad * 1.4826 + 1e-9)  # Factor de conversión MAD a std
    
    out["fundingRate_z"] = _zscore_robust(out["fundingRate"])
    out["oi_change_pct"] = out["openInterest"].pct_change(fill_method=None).fillna(0) * 100
    out["glsr_z"] = _zscore_robust(out["glsr"])
    out["tlsr_z"] = _zscore_robust(out["tlsr"])
    
    return out

# Continúa en la siguiente parte...