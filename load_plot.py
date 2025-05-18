import pandas as pd
import matplotlib.pyplot as plt

file = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/Data_creation/load_profil.csv"
df = pd.read_csv(file, parse_dates=['time'], index_col='time')

df['Power consumption'] = df['Power consumption']/1000
summer_day = df.loc['2023-07-21']
winter_day = df.loc['2023-01-01']

hourly_average = df.groupby(df.index.hour)['Power consumption'].mean()

plt.plot(summer_day.index.hour, summer_day['Power consumption'], label="Summer Day", color='orange', marker='o')
plt.plot(winter_day.index.hour, winter_day['Power consumption'], label="Winter Day", color='blue', marker='o')
plt.plot(hourly_average.index, hourly_average, label="Hourly Average", color='green', linestyle='--')
plt.title("Energy Power Consumption Hourly Trend (Summer Day vs Winter Day)")
plt.xlabel("Hours of the day")
plt.ylabel("Power Consumption (kWh)")
plt.xticks(range(24))
plt.grid(True)
plt.legend()
plt.show()

df['month'] = df.index.month
monthly_cumulative_consumption = df.groupby('month')['Power consumption'].sum()

monthly_cumulative_consumption.plot(kind='bar', color='skyblue')
plt.title("Cumulative Consumption by Month")
plt.xlabel("Month")
plt.ylabel("Cumulative Consumption (kWh)")
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(axis='y')
plt.show()



total_consumption = df['Power consumption'].sum()
daily_average_consumption = df.resample('D')['Power consumption'].sum().mean()

print(f"Total Consumption for the Year (kWh): {total_consumption}")
print(f"Average Daily Consumption (kWh): {daily_average_consumption}")
print(max(df['Power consumption']))