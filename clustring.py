import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

CSV_PATH = "Data/Exp01_Map-matched&DTW.csv"
OUTPUT_DIR = "PlotsImg"
LAT_COL = "lat"
LON_COL = "lon"
CONFIRMED_COL = "Confirmed"
LEFT_COL = "Left"
RIGHT_COL = "Right"

df = pd.read_csv(CSV_PATH)

# Select relevant columns
cols = ['lat', 'lon', 'Left', 'Right', 'Confirmed']

# 1. Overall Spearman correlation
spearman_corr_overall = df[cols].corr(method='spearman')

plt.figure(figsize=(8,6))
sns.heatmap(spearman_corr_overall, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Spearman Correlation (Overall)")
plt.savefig("PlotsImg/spearman_corr_overall.png", dpi=300)
plt.show()

# 2. Spearman correlation when Confirmed > 0
df_confirmed = df[df['Confirmed'] > 0]
spearman_corr_confirmed = df_confirmed[cols].corr(method='spearman')

plt.figure(figsize=(8,6))
sns.heatmap(spearman_corr_confirmed, annot=True, cmap='viridis', fmt=".2f")
plt.title("Spearman Correlation (Confirmed > 0)")
plt.savefig("PlotsImg/spearman_corr_confirmed.png", dpi=300)
plt.show()

# 3. Optional: Pairplot to see scatter trends
sns.pairplot(df, vars=['lat','lon','Left','Right'], hue='Confirmed', palette='coolwarm', plot_kws={'alpha':0.6})
plt.savefig("PlotsImg/pairplot_confirmed.png", dpi=300)
plt.show()
