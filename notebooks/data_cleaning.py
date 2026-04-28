"""
Clean data for OLS / Decision Tree / XGBoost / NN.

Decisions documented from EDA:
- Drop pct_female: pct_male + pct_female = 1 (perfect collinearity)
- Drop pct_black: VIF=362 in full set; iterative VIF prune removed it
- Drop ZIP 20006: 3 of top-3 Cook's D observations (D > 1.0); downtown DC commercial
                  zone with extreme rent_to_income ~0.65 vs sample median 0.20
- Two output frames:
    df_demo: also drops median_income & median_rent (target is their ratio →
             demographic-structure model, answers the research question directly)
    df_full: keeps income & rent (sensitivity / full ISLP benchmark)
"""
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path('data/raw_data/dmv_rent_income_dataset.csv')
df = pd.read_csv(INPUT)
df['zip'] = df['zip'].astype(str).str.zfill(5)

print(f'Input: {df.shape}')

# ---- Shared cleaning steps ----
DROP_ZIPS = ['20006']
COLLIN_DROPS = ['pct_female', 'pct_black']

df_clean = df[~df['zip'].isin(DROP_ZIPS)].copy()
df_clean = df_clean.drop(columns=COLLIN_DROPS)
print(f'After ZIP 20006 + collinear drops: {df_clean.shape}')
print(f'  ZIPs removed: {DROP_ZIPS} → {len(df_clean["zip"].unique())} ZIPs remain')
print(f'  Cols dropped: {COLLIN_DROPS}')

# ---- Two output variants ----
df_demo = df_clean.drop(columns=['median_income', 'median_rent']).copy()
df_full = df_clean.copy()

print(f'\ndf_demo (no income/rent): {df_demo.shape}, cols = {list(df_demo.columns)}')
print(f'df_full (with income/rent): {df_full.shape}, cols = {list(df_full.columns)}')

# ---- Sanity checks ----
assert df_demo.isna().sum().sum() == 0, 'df_demo has NaN'
assert df_full.isna().sum().sum() == 0, 'df_full has NaN'
assert (df_demo['rent_to_income'] > 0).all(), 'df_demo has nonpositive target'
assert df_demo['zip'].nunique() == 167, f"Expected 167 ZIPs, got {df_demo['zip'].nunique()}"

# ---- Save ----
df_demo.to_csv('data/processed_data/dmv_clean_demographics.csv', index=False)
df_full.to_csv('data/processed_data/dmv_clean_full.csv', index=False)
# ---- Predictor lists for Phase B ----
TARGET = 'rent_to_income'
ID_COLS = ['zip', 'year']

predictors_demo = [c for c in df_demo.columns if c not in ID_COLS + [TARGET]]
predictors_full = [c for c in df_full.columns if c not in ID_COLS + [TARGET]]
print(f'\nPredictors (demo, n={len(predictors_demo)}): {predictors_demo}')
print(f'Predictors (full, n={len(predictors_full)}): {predictors_full}')