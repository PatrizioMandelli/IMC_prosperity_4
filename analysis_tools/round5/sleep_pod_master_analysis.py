import pandas as pd
import numpy as np
from pathlib import Path
import glob
from statsmodels.tsa.stattools import adfuller

# Configurazione prodotti e oracoli identificati come ottimali
RELATIONS = {
    'SLEEP_POD_POLYESTER': 'UV_VISOR_AMBER',
    'SLEEP_POD_SUEDE': 'MICROCHIP_SQUARE',
    'SLEEP_POD_NYLON': 'MICROCHIP_TRIANGLE',
    'SLEEP_POD_LAMB_WOOL': 'TRANSLATOR_ECLIPSE_CHARCOAL',
    'SLEEP_POD_COTTON': 'SLEEP_POD_POLYESTER'
}

def load_data(data_dir: Path):
    print(f"📡 Caricamento dati da {data_dir}...")
    files = sorted(glob.glob(str(data_dir / "prices_round_5_day_*.csv")))
    df_list = [pd.read_csv(f, sep=";").rename(columns=lambda x: x.strip().lower()) for f in files]
    return pd.concat(df_list, ignore_index=True).sort_values(by=['day', 'timestamp'])

def main():
    data_path = Path("data/round5")
    df = load_data(data_path)
    pivot = df.pivot_table(index=['day', 'timestamp'], columns='product', values='mid_price').ffill().dropna()
    
    print("\n" + "="*80)
    print(" SLEEP POD MASTER ANALYSIS - PRODUCTION PARAMETERS")
    print("="*80)
    
    results = []
    for pod, oracle in RELATIONS.items():
        y, x = pivot[pod], pivot[oracle]
        beta, alpha = np.polyfit(x, y, 1)
        spread = y - (alpha + beta * x)
        adf_p = adfuller(spread)[1]
        
        print(f"\nTarget: {pod:25} | Oracle: {oracle}")
        print(f"  > Alpha: {alpha:10.2f}")
        print(f"  > Beta:  {beta:10.4f}")
        print(f"  > StdErr: {spread.std():9.2f}")
        print(f"  > ADF p-value: {adf_p:.4f} ({'STATIONARY' if adf_p < 0.05 else 'NON-STATIONARY'})")
        
        results.append({'pod': pod, 'alpha': alpha, 'beta': beta})

    print("\n" + "="*80)
    print(" TRADER INITIALIZATION CODE (COPY-PASTE)")
    print("="*80)
    print("self.params = {")
    for res in results:
        print(f"    '{res['pod']}': {{'alpha': {res['alpha']:.2f}, 'beta': {res['beta']:.4f}}},")
    print("}")

if __name__ == "__main__":
    main()
