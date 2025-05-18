import pandas as pd

csv_filename = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/ML/Norway.csv"  # Remplacez par le chemin du fichier CSV
df_cleaned = pd.read_csv(csv_filename)

df_cleaned.columns = ['time', 'Price kWh']

df_cleaned['time'] = pd.to_datetime(df_cleaned['time'])
df_cleaned['Price kWh'] = df_cleaned['Price kWh']/1000

print(df_cleaned.head())
df_cleaned.to_csv("C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/ML/Norway.csv", index=False)
