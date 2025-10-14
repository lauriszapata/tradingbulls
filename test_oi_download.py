import requests
import time
import pandas as pd

BINANCE_FAPI = "https://fapi.binance.com"

def test_oi_download(symbol, start, end):
    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    step = 24 * 60 * 60 * 1000  # 1 día
    ms = start
    total = end - start
    all_data = []
    while ms <= end:
        e = min(end, ms + step - 1)
        p = {"symbol": symbol, "period": "1h", "startTime": ms, "endTime": e, "limit": 500}
        print(f"Solicitando OI {symbol}: desde {ms} hasta {e}")
        try:
            r = requests.get(url, params=p, timeout=40)
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
    import time as t
    end = int(t.time() * 1000)
    start = end - 30 * 24 * 60 * 60 * 1000  # 30 días
    test_oi_download("BTCUSDT", start, end)
