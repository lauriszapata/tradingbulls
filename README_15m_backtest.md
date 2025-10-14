# Tommy Bulls - 15m Maximum Profit Strategy

## Descripción
Estrategia de backtesting optimizada para futuros perpetuos (15m timeframe) en Binance USDM.  
**Objetivo:** Maximizar PNL neto con valor esperado positivo, controlando drawdown, profit factor y Sharpe ratio.

## Características principales
- **Timeframe:** 15 minutos
- **Datos:** Julio-Agosto 2025 (entrenamiento) | Septiembre 2025 (prueba out-of-sample)
- **Modelo:** Gradient Boosting con calibración Platt opcional y weighting de clases
- **Filtros:** ADX, distancia a VWAP, probabilidad mínima, EV > 0, funding rate, long/short ratio
- **Métricas clave:** PNL, Profit Factor ≥1.3, Sharpe ≥1.0, Max DD% ≤20%, AUC test
- **Contexto:** Funding rate, Open Interest, Long/Short ratios (global/top), macro BTC

## Instalación
```bash
pip install -r requirements.txt
```

## Uso

### Modo rápido (grid reducido ~32 configuraciones)
```bash
python3 tb_backtest_15m_max_profit.py --symbol ETH/USDT --quick
```

### Modo completo (grid extenso ~1280 configuraciones)
```bash
python3 tb_backtest_15m_max_profit.py --symbol ETH/USDT
```

### Desactivar calibración Platt
```bash
python3 tb_backtest_15m_max_profit.py --symbol ETH/USDT --quick --no-calibrate
```

### Desactivar class weighting
```bash
python3 tb_backtest_15m_max_profit.py --symbol ETH/USDT --quick --no-class-weighting
```

## Salidas generadas

### Archivos principales
- **trades_15m_<symbol>.csv**: Todos los trades ejecutados con detalles (TP/SL precios, prob, EV, k_tp/k_sl, horizonte)
- **equity_curve_<symbol>.csv**: Curva de equity acumulada trade a trade
- **grid_results_<symbol>.csv**: Resultados de todas las configuraciones evaluadas (ordenadas por PNL)
- **best_summary_<symbol>.csv**: Resumen detallado de la mejor configuración (incluye parámetros, métricas, filtros activos)
- **predictions_15m_<symbol>.csv**: Probabilidades, EV y señales de entrada para cada vela del test
- **predictions_15m_<symbol>_with_labels.csv**: Igual que anterior + labels reales (TP=1, SL=0, NaN=timeout)

### Estructura de datos cacheada
- **data_cache/**: OHLCV descargado (ETH, BTC) en parquet para reutilización
- **context_cache/**: Funding, Open Interest, Long/Short ratios en parquet

## Parámetros de la grid

### Modo rápido
- k_tp: [1.25, 1.5]
- k_sl: [1.0, 1.25]
- H (horizonte velas): [4, 6] → 1-1.5h
- p_min (prob mín): [0.6, 0.65]
- ADX_min: [18, 20]
- dist_VWAP_max (%): [1.0]
- Filtros contexto: require_pos_funding=[False], require_glsr_above1=[False]

### Modo completo
- k_tp: [1.0, 1.25, 1.5, 1.75, 2.0]
- k_sl: [0.8, 1.0, 1.25, 1.5]
- H: [4, 5, 6, 8, 10] → 1-2.5h
- p_min: [0.55, 0.60, 0.65, 0.70, 0.75]
- ADX_min: [15, 18, 20, 22]
- dist_VWAP_max: [0.5, 1.0, 1.5]
- require_pos_funding: [False, True]
- require_glsr_above1: [False, True]

## Restricciones de selección
La mejor configuración debe cumplir:
- **Max Drawdown %** ≤ 20%
- **Profit Factor** ≥ 1.3
- **Sharpe Ratio** ≥ 1.0
- Entre las configuraciones viables, se elige la de **mayor PNL neto**

## Features del modelo (lag=1)
### Precio y volumen
- close, volume, EMA7, EMA25, EMA99, RSI14, ADX14, ATR_pct

### Bollinger y VWAP
- BB_width, BB_%B, VWAP, dist_VWAP_pct

### Contexto futures
- fundingRate, fundingRate_z, openInterest, oi_change_pct
- glsr, glsr_z, tlsr, tlsr_z

### Retornos (15m y 1h)
- ret_sym_15m, ret_sym_1h, ret_btc_15m, ret_btc_1h

### Pendientes y macro
- sym_ema_slope, btc_ema_slope, btc_atr_pct

### Temporal y sesión
- dow (día semana), hour, sess_asia, sess_eu, sess_us
- vr_low, vr_mid, vr_high (régimen volatilidad)
- trend_long, trend_short

## Calibración y weighting
- **Calibración Platt (default ON):** Divide train en 80% fit + 20% hold para ajustar probabilidades via logistic regression.
- **Class weighting (default ON):** Ponderación inversa a la frecuencia de clases TP/SL para balancear.
- Ambos mejoran la estimación de p_TP y el cálculo de EV gating.

## Logs y métricas extendidas
Al finalizar muestra:
- k_tp, k_sl, H, p_min, adx_min, dist_vwap_max
- trades, pnl, dd, dd_pct, pf (profit factor), sh (Sharpe), hr (hit rate), exp (expectativa)
- **auc** (AUC-ROC test)
- win_rate, avg_pnl, median_pnl, max_consec_loss

## Notas técnicas
- **Símbolos:** Detecta automáticamente formato perpetual (ej: ETH/USDT → ETH/USDT:USDT)
- **Horizonte:** Usa first-touch TP/SL dentro de H velas; si no toca → salida por tiempo
- **EV gating:** Solo entra si EV = p_TP*TP - (1-p_TP)*SL - fees > 0
- **Cooldown:** 4 velas entre trades consecutivos
- **OI limitado:** Binance API solo sirve últimos ~30 días; fuera de ese rango → NaN
- **BTC fallback:** Si BTC falla descarga, continua sin features macro (0-relleno)

## Próximas mejoras sugeridas
1. Optimización Bayesiana (reducir grid time)
2. Trailing stop parcial dentro del horizonte
3. Filtro adicional: AUC mínimo en viable configs
4. SHAP/Permutation importance para features top
5. Penalización dinámica k_sl en volatilidad alta (vr_high)
6. Estrategia multipar (diversificación)

## Autor
Tommy Bulls Backtest Framework  
Versión: 1.0 (14 Oct 2025)
