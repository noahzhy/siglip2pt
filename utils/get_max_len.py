# load csv from /home/haoyu/projects/siglip2pt/data/20251114_pms_oos_dlc.csv
import os
import sys
import pandas as pd

topN = 20

f_path = "/home/haoyu/projects/siglip2pt/data/20251114_pms_skuname.csv"
df = pd.read_csv(f_path)

skunames = df['SKUName'].astype(str).tolist()

# sort by length descending
sorted_names = sorted(skunames, key=len, reverse=True)

# take top 5
topX = sorted_names[:topN]

# print names and lengths
for i, name in enumerate(topX, 1):
    print(f"{i}. length {len(name)} | {name}")

# # count if SKUName and url0 both same
# dup_groups = df.groupby(['SKUName', 'url0']).size().reset_index(name='count')

# # keep only duplicates (count > 1)
# dup_groups = dup_groups[dup_groups['count'] > 1]

# print(dup_groups)

# total_dup = len(dup_groups)
# print("Total duplicate groups:", total_dup)

# # get all SKUName include _ symbol
# sku_with_underscore = df[df['SKUName'].str.contains('_', na=False)]
# total_underscore = len(sku_with_underscore)
# print("Total SKUName with underscore:", total_underscore)
# for index, row in sku_with_underscore.iterrows():
#     print(f"SKUName: {row['SKUName']}, url0: {row['url0']}")


# # count same SKUName include colour symbol
# # all lowercase
# sku_with_colour = df[df['SKUName'].str.lower().str.contains('colour', na=False)]
# total_colour = len(sku_with_colour)
# print("Total SKUName with colour:", total_colour)
# for index, row in sku_with_colour.iterrows():
#     print(f"SKUName: {row['SKUName']}, url0: {row['url0']}")

