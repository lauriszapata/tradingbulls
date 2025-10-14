import requests
import time
import pandas as pd

BINANCE_FAPI = "https://fapi.binance.com"

# Prueba de descarga de LS con logging detallado

def test_ls_download(symbol, start, end):
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
        print(df.head())
    else:
        print("No se descargaron datos.")

if __name__ == "__main__":
    # Rango de prueba: últimos 7 días
    import time as t
    end = int(t.time() * 1000)
    start = end - 7 * 24 * 60 * 60 * 1000
    test_ls_download("BTCUSDT", start, end)
