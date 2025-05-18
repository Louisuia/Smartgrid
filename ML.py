import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_filename1 = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/ML/NO2_2023.csv"
csv_filename2 = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/ML/NO2_2022.csv"
csv_filename3 = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/ML/NO2_2021.csv"
csv_filename4 = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/ML/NO2_2020.csv"
df_test = pd.read_csv(csv_filename1)
df_train2022 = pd.read_csv(csv_filename2)
df_train2021 = pd.read_csv(csv_filename3)
df_train2020 = pd.read_csv(csv_filename4)

def data_prep(df):
    df['Price kWh t-1'] = df['Price kWh'].shift(1)
    df['Price kWh t-2'] = df['Price kWh'].shift(2)
    df['Price kWh t-3'] = df['Price kWh'].shift(3)
    df['Price kWh t+1'] = df['Price kWh'].shift(-1)
    df['Price kWh t+2'] = df['Price kWh'].shift(-2)
    df['Price kWh t+3'] = df['Price kWh'].shift(-3)
    
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date 
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['month'] = pd.to_datetime(df['time']).dt.month
    df['weekday'] = pd.to_datetime(df['time']).dt.weekday

    df = df[['time', 'hour','date', 'month', 'weekday','Price kWh', 'Price kWh t-1', 'Price kWh t-2', 'Price kWh t-3','Price kWh t+1','Price kWh t+2']]
    df = df.dropna()


# y = df_test
# data_prep(y)
# y = y.dropna()
# y = y[['time', 'hour','date', 'month', 'weekday','Price kWh', 'Price kWh t-1', 'Price kWh t-2', 'Price kWh t-3','Price kWh t+1','Price kWh t+2','Price kWh t+3']]
# y.to_csv(csv_filename1, index=False)


################################ ML ################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def ml1h(df_train,df_test):

    features = ['hour','weekday','month','Price kWh t-1','Price kWh t-2','Price kWh']

    X_train = df_train[features]
    y_train = df_train['Price kWh t+1']

    X_test = df_test[features]
    y_test = df_test['Price kWh t+1']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    df_test['Predicted t+1'] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE) : {mse}")
    print(f"Mean Absolute Error (MAE) : {mae}")
    print(f"R² Score : {r2}")

def ml2h(df_train,df_test):

    features = ['hour','weekday','month','Price kWh t-1','Price kWh t-2','Price kWh']

    X_train = df_train[features]
    y_train = df_train['Price kWh t+2']

    X_test = df_test[features]
    y_test = df_test['Price kWh t+2']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    df_test['Predicted t+2'] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE) : {mse}")
    print(f"Mean Absolute Error (MAE) : {mae}")
    print(f"R² Score : {r2}")

def ml3h(df_train,df_test):

    features = ['hour','weekday','month','Price kWh t-1','Price kWh t-2','Price kWh']

    X_train = df_train[features]
    y_train = df_train['Price kWh t+3']

    X_test = df_test[features]
    y_test = df_test['Price kWh t+3']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    df_test['Predicted t+3'] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE) : {mse}")
    print(f"Mean Absolute Error (MAE) : {mae}")
    print(f"R² Score : {r2}")

print("ml1h")
ml1h(pd.concat([df_train2022,df_train2021,df_train2020]),df_test)
print("ml2h")
ml2h(pd.concat([df_train2022,df_train2021,df_train2020]),df_test)
print("ml3h")
ml3h(pd.concat([df_train2022,df_train2021,df_train2020]),df_test)

# df_test.to_csv(csv_filename1, index=False)
# print(df_test.head())

def add_opt_charge(df):
    df['opt_charge'] = False
    i = 0
    while i < len(df) - 1:
        if df.iloc[i]['hour'] >= 19 or df.iloc[i]['hour'] < 7:
            if df.iloc[i]['Predicted t+3'] > df.iloc[i]['Price kWh']:
                df.at[i, 'opt_charge'] = True
                i += 12
                continue
        i += 1

    return df

# add_opt_charge(df_test)
# df_test.to_csv(csv_filename1, index=False)

def plot_5_days_prices(df_test, start_day="2023-03-01"):

    df_test["time"] = pd.to_datetime(df_test["time"])
    start_date = pd.to_datetime(start_day)
    end_date = start_date + pd.Timedelta(days=8)
    daily_data = df_test[(df_test["time"] >= start_date) & (df_test["time"] < end_date)]
    if daily_data.empty:
        print("Aucune donnée pour cette période.")
        return

    daily_data = daily_data.sort_values(by="time")
    # daily_data["Price Moving Avg 5j"] = daily_data["Price kWh"].rolling(120).mean()
    # daily_data["Price Moving Avg 24h"] = daily_data["Price kWh"].rolling(24).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_data["time"], daily_data["Price kWh"], label="Real price", marker="o", color="blue")
    # plt.plot(daily_data["time"].shift(-1), daily_data["Predicted t+1"], label="Predicted t+1", linestyle="--", color="red")
    # plt.plot(daily_data["time"], daily_data["Predicted t+2"], label="Predicted t+2", linestyle="--", color="orange")
    plt.plot(daily_data["time"].shift(-3), daily_data["Predicted t+3"], label="Predicted t+3", linestyle="--", color="orange")
    # plt.plot(daily_data["time"], daily_data["Price Moving Avg 5j"], label="Moyenne Mobile 5j", linestyle="--", color="orange")
    # plt.plot(daily_data["time"], daily_data["Price Moving Avg 24h"], label="Moyenne Mobile 24h", linestyle="--", color="red")
    
    for day in pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq="D"):
        for hour_range in [(19, 23, 59), (0, 7, 0)]:
            start = pd.Timestamp(f"{day.date()} {hour_range[0]}:00:00")
            end = pd.Timestamp(f"{day.date()} {hour_range[1]}:{hour_range[2]}:59")
            plt.axvspan(start, end, color="green", alpha=0.2)

    
    charge_hours = daily_data[daily_data["opt_charge"] == True]["time"]
    for charge_time in charge_hours:
        start_charge = charge_time
        end_charge = charge_time + pd.Timedelta(hours=2)
        charge_period = daily_data[(daily_data["time"] >= start_charge) & (daily_data["time"] < end_charge)]
        
        if not charge_period.empty:
            avg_price = charge_period["Price kWh"].mean()
            plt.axvspan(start_charge, end_charge, color="red", alpha=0.4)
            plt.hlines(y=avg_price, xmin=start_charge, xmax=end_charge, colors="black", linestyles="dashed", linewidth=2) 
    
    plt.xlabel("Hour")
    plt.ylabel("Price (kWh)")
    plt.title(f"Price from {start_day} to {end_date.date()}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

plot_5_days_prices(df_test, start_day="2023-06-12")



# nb_true = df_test['opt_charge'].sum()
# print(f"Nombre total de créneaux 'opt_charge' = True : {nb_true}")


##################### Save opt charge ######################

# df_opt = df_test[['time', 'opt_charge']].copy()
# df_opt['time'] = pd.to_datetime(df_opt['time'])
# df_opt = df_opt.drop_duplicates(subset=['time'])
# full_time_index = pd.date_range(start="2023-01-01 00:00:00", end="2023-12-31 23:00:00", freq="h")
# df_opt = df_opt.set_index('time').reindex(full_time_index, fill_value=False)
# df_opt = df_opt.reset_index().rename(columns={'index': 'time'})
# df_opt.to_csv("C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/ML/Opt_charge_2023.csv", index=False)
# print(df_opt)


