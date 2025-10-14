import pandas as pd
import numpy as np
import os

RESULTS_DIR = os.path.expanduser('~/Desktop/tb_results')
SYMBOLS = ['BTCUSDT', 'ETHUSDT']

def find_dataset_csv(results_dir, symbol):
    files = [f for f in os.listdir(results_dir) if f.startswith(f'dataset_5m_{symbol}') and f.endswith('.csv')]
    if not files:
        print(f'No se encontró dataset para {symbol}')
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    return os.path.join(results_dir, files[0])

def check_indicators(symbol):
    csv_path = find_dataset_csv(RESULTS_DIR, symbol)
    if not csv_path:
        return
    
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"Verificación de indicadores para {symbol}")
    print(f"{'='*60}")
    
    # 1. Verificar columnas presentes
    expected_cols = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'EMA7', 'EMA25', 'EMA99', 'RSI14', 'ADX14', 'ATR_pct',
        'BB_width', 'BB_%B', 'VWAP', 'dist_VWAP_pct',
        'fundingRate', 'openInterest', 'glsr', 'tlsr',
        'trend_long', 'trend_short'
    ]
    
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        print(f"❌ Columnas faltantes: {missing}")
    else:
        print("✅ Todas las columnas esperadas están presentes")
    
    # 2. Verificar porcentaje de NaN en indicadores clave
    print("\n📊 Porcentaje de NaN por indicador:")
    key_indicators = ['EMA7', 'EMA25', 'RSI14', 'ADX14', 'ATR_pct', 'VWAP', 
                      'fundingRate', 'openInterest', 'glsr', 'tlsr']
    for col in key_indicators:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            status = "✅" if nan_pct < 5 else ("⚠️" if nan_pct < 20 else "❌")
            print(f"{status} {col}: {nan_pct:.2f}% NaN")
    
    # 3. Verificar rangos lógicos de indicadores
    print("\n🔍 Verificación de rangos lógicos:")
    if 'RSI14' in df.columns:
        rsi_valid = df['RSI14'].between(0, 100).all()
        print(f"{'✅' if rsi_valid else '❌'} RSI14 en rango [0, 100]: {rsi_valid}")
        if not rsi_valid:
            print(f"   Valores fuera de rango: min={df['RSI14'].min()}, max={df['RSI14'].max()}")
    
    if 'ATR_pct' in df.columns:
        atr_valid = (df['ATR_pct'] >= 0).all()
        print(f"{'✅' if atr_valid else '❌'} ATR_pct >= 0: {atr_valid}")
    
    if 'EMA7' in df.columns and 'EMA25' in df.columns:
        # Verificar que EMAs sean razonables respecto al precio
        ema7_check = df['EMA7'].between(df['close'] * 0.5, df['close'] * 1.5).mean()
        print(f"{'✅' if ema7_check > 0.95 else '⚠️'} EMA7 cerca del precio: {ema7_check*100:.1f}% de las velas")
    
    # 4. Verificar cálculo de VWAP
    if 'VWAP' in df.columns and 'close' in df.columns:
        # VWAP debería estar cerca del precio típico
        vwap_diff = abs(df['VWAP'] - df['close']).mean() / df['close'].mean()
        print(f"{'✅' if vwap_diff < 0.05 else '⚠️'} VWAP desviación promedio del close: {vwap_diff*100:.2f}%")
    
    # 5. Verificar indicadores de contexto
    print("\n📡 Indicadores de contexto:")
    if 'fundingRate' in df.columns:
        funding_range = df['fundingRate'].describe()
        print(f"  fundingRate: mean={funding_range['mean']:.6f}, std={funding_range['std']:.6f}")
    
    if 'openInterest' in df.columns:
        oi_non_zero = (df['openInterest'] != 0).sum()
        oi_total = len(df)
        print(f"  openInterest: {oi_non_zero}/{oi_total} valores no-cero ({oi_non_zero/oi_total*100:.1f}%)")
    
    if 'glsr' in df.columns:
        glsr_range = df['glsr'].describe()
        print(f"  glsr: mean={glsr_range['mean']:.4f}, std={glsr_range['std']:.4f}")
    
    # 6. Muestra de datos para inspección visual
    print("\n📋 Muestra de datos (primera fila con todos los indicadores):")
    if len(df) > 100:
        sample_row = df.iloc[100]
        for col in ['timestamp', 'close', 'EMA7', 'RSI14', 'ADX14', 'ATR_pct', 
                    'VWAP', 'fundingRate', 'openInterest', 'glsr']:
            if col in df.columns:
                print(f"  {col}: {sample_row[col]}")

if __name__ == "__main__":
    for symbol in SYMBOLS:
        check_indicators(symbol)
