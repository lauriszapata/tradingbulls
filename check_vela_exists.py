import pandas as pd
import os

RESULTS_DIR = os.path.expanduser('~/Desktop/tb_results')
SYMBOLS = ['BTCUSDT', 'ETHUSDT']

for symbol in SYMBOLS:
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith(f'dataset_5m_{symbol}') and f.endswith('.csv')]
    if not files:
        print(f'No se encontró dataset para {symbol}')
        continue
    files.sort(key=lambda x: os.path.getmtime(os.path.join(RESULTS_DIR, x)), reverse=True)
    csv_path = os.path.join(RESULTS_DIR, files[0])
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"\n{symbol}: Rango de fechas en dataset local:")
        print('Mínima:', df['timestamp'].min())
        print('Máxima:', df['timestamp'].max())
        # Verificar velas en puntos clave del rango
        puntos = [
            pd.Timestamp('2025-06-15 10:00:00', tz='UTC'),
            pd.Timestamp('2025-07-20 14:30:00', tz='UTC'),
            pd.Timestamp('2025-08-25 08:30:00', tz='UTC'),
            pd.Timestamp('2025-09-01 12:00:00', tz='UTC')
        ]
        for ts in puntos:
            vela = df[df['timestamp'] == ts]
            if not vela.empty:
                print(f"✓ Vela {ts} existe: close={vela['close'].values[0]}")
            else:
                print(f"✗ Vela {ts} NO existe")
    else:
        print(f"No hay columna 'timestamp' en {csv_path}")
