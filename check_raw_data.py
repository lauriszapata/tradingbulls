import pandas as pd
import os

RESULTS_DIR = os.path.expanduser('~/Desktop/tb_results')

def find_latest_csv(results_dir):
    files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not files:
        print('No se encontraron archivos CSV en', results_dir)
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    return os.path.join(results_dir, files[0])

def check_raw_data(filepath):
    print(f'Analizando archivo: {filepath}')
    df = pd.read_csv(filepath)
    print('\nColumnas y tipos:')
    print(df.dtypes)
    print(f'\nFilas: {len(df)}')
    if "timestamp" in df.columns:
        print('Fecha mínima:', pd.to_datetime(df["timestamp"]).min())
        print('Fecha máxima:', pd.to_datetime(df["timestamp"]).max())
    print('\nPorcentaje de NaN por columna:')
    print(df.isna().mean().round(3) * 100)
    print('\nDuplicados:', df.duplicated().sum())
    for col in ["close", "open", "high", "low"]:
        if col in df.columns:
            print(f'{col}: min={df[col].min()}, max={df[col].max()}')
    print('\nPrimeras filas:')
    print(df.head())
    print('\nÚltimas filas:')
    print(df.tail())

if __name__ == "__main__":
    csv_path = find_latest_csv(RESULTS_DIR)
    if csv_path:
        check_raw_data(csv_path)
    else:
        print('No se encontró archivo de datos crudos para analizar.')
