
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/ML/NO2_2023.csv", parse_dates=["time"])

df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
df["is_night"] = df["hour"].between(0, 6).astype(int)
df["is_peak_hour"] = df["hour"].between(8, 20).astype(int)

df['normalized_price'] = df.groupby('date')["Price kWh"].transform(zscore)

cols_for_corr = [
    "hour", "month", "weekday", "is_weekend", "is_night", "is_peak_hour",
    "Price kWh", "normalized_price"
]

corr_matrix = df[cols_for_corr].corr()

mask = abs(corr_matrix) <= 0.05

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            mask=mask, linewidths=0.5, square=True, cbar_kws={"shrink": .8})
plt.show()



hourly_avg = df.groupby("hour")["normalized_price"].mean()
correlation = df[["hour", "normalized_price"]].corr().iloc[0,1]
sns.lineplot(data=hourly_avg)
plt.xlabel("Hour")
plt.ylabel("Average normalized price")
plt.grid(True)
plt.show()