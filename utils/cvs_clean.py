import os
import sys
import pandas as pd

f_path = "/home/haoyu/projects/siglip2pt/data/20251114_pms_oos_dlc.csv"
df = pd.read_csv(f_path)

save_path = "/home/haoyu/projects/siglip2pt/data/20251114_pms_cleaned.csv"

df['SKUName'] = (
    df['SKUName']
    .astype(str)
    .str.replace("_", " ")
    .str.replace("’", "'")
    .str.replace("‘", "'")
    .str.replace("×", "x")
    .str.replace("脳", "x")
    .str.replace("B******", "BITCHIN'")
    .str.replace("L'Or茅al", "L'Oréal")
    .str.replace("Bacard铆", "Bacardí")
    .str.strip()
)

# BarCode column as int, if not, not fill anything
def clean_barcode(x):
    try:
        return str(int(x))
    except:
        return x

df['BarCode'] = df['BarCode'].apply(clean_barcode)

df.to_csv(save_path, index=False)

# keep SkuId,SKUName
df_cleaned = df[['SkuId', 'SKUName']]
# rename SkuId to ProductId
df_cleaned = df_cleaned.rename(columns={'SkuId': 'ProductId'})
# sort by SKUName
df_cleaned = df_cleaned.sort_values(by='SKUName').reset_index(drop=True)
df_cleaned.to_csv("/home/haoyu/projects/siglip2pt/data/20251114_pms_skuname.csv", index=False)
