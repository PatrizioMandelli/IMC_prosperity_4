import pandas as pd
import glob

files = sorted(glob.glob("data/round1/prices_round_1_day_*.csv"))
print(files)
df = pd.concat([pd.read_csv(f, sep=";") for f in files], ignore_index=True)
df['mid_price'] = pd.to_numeric(df['mid_price'], errors='coerce')

for product in df['product'].unique():
    stds = []
    for day in df['day'].unique():
        mid = df[(df['product'] == product) & (df['day'] == day)]['mid_price'].dropna()
        diff = mid.diff().dropna()
        # escludi outlier oltre il 99° percentile
        diff_clean = diff[diff.abs() < diff.abs().quantile(0.99)]
        stds.append(diff_clean.std())
    print(f"{product}: {pd.Series(stds).mean():.4f}")

#INTARIAN_PEPPER_ROOT: 2.9359
#ASH_COATED_OSMIUM: 3.5191
