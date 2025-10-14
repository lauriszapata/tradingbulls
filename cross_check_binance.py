import pandas as pd
import ccxt
import os

RESULTS_DIR = os.path.expanduser('~/Desktop/tb_results')
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TIMEFRAME = '5m'

def find_dataset_csv(results_dir, symbol):
    files = [f for f in os.listdir(results_dir) if f.startswith(f'dataset_5m_{symbol}') and f.endswith('.csv')]
    if not files:
        print(f'No se encontró dataset para {symbol}')
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    return os.path.join(results_dir, files[0])

def fetch_binance_ohlcv(symbol, start, end, limit=1000):
    binance = ccxt.binanceusdm()
    binance.load_markets()
    ohlcv = []
    since = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    while since < end_ms:
        batch = binance.fetch_ohlcv(symbol, TIMEFRAME, since, limit)
        if not batch:
            break
        ohlcv.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < limit:
            break
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def random_check(symbol, date_str, time_str):
    csv_path = find_dataset_csv(RESULTS_DIR, symbol)
    if not csv_path:
        return
    df_local = pd.read_csv(csv_path)
    if "timestamp" in df_local.columns:
        df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], utc=True)
    else:
        print(f"No hay columna 'timestamp' en {csv_path}")
        return
    ts = pd.Timestamp(f"{date_str} {time_str}", tz='UTC')
    local_row = df_local[df_local["timestamp"]==ts]
    df_binance = fetch_binance_ohlcv(symbol.replace('USDT','/USDT:USDT'), ts, ts + pd.Timedelta('5min'))
    bin_row = df_binance[df_binance["timestamp"]==ts]
    print(f"\nChequeo aleatorio para {symbol} en {ts}")
    if not bin_row.empty and not local_row.empty:
        print(f"LOCAL: open={local_row['open'].values[0]}, high={local_row['high'].values[0]}, low={local_row['low'].values[0]}, close={local_row['close'].values[0]}, vol={local_row['volume'].values[0]}")
        print(f"BINANCE: open={bin_row['open'].values[0]}, high={bin_row['high'].values[0]}, low={bin_row['low'].values[0]}, close={bin_row['close'].values[0]}, vol={bin_row['volume'].values[0]}")
        diff_close = abs(local_row['close'].values[0] - bin_row['close'].values[0])
        if diff_close < 0.01:
            print("✓ Datos coinciden correctamente")
        else:
            print(f"⚠️  Diferencia en close: {diff_close}")
    else:
        print("✗ No se encontró la vela en uno de los archivos")

if __name__ == "__main__":
    random_check('BTCUSDT', '2025-06-15', '10:00:00')
    random_check('BTCUSDT', '2025-07-20', '14:30:00')
    random_check('BTCUSDT', '2025-08-25', '08:30:00')
    random_check('ETHUSDT', '2025-09-01', '12:00:00')
