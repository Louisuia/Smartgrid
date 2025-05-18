import requests
import pandas as pd

def get_pvgis_data_hourly(latitude, longitude, start_year, end_year):
    url = "https://re.jrc.ec.europa.eu/api/seriescalc"

    params = {
        "lat": latitude,
        "lon": longitude,
        "outputformat": "json",
        "startyear": start_year,
        "endyear": end_year,
        "peakpower": 1,
        "inclination": 58,
        "azimuth": 0,
        "timescale": "hourly"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

lat, lon = 58.3405, 8.59343
start_year = 2023
end_year = 2023

data = get_pvgis_data_hourly(lat, lon, start_year, end_year)

if data and 'outputs' in data:
    hourly_data = data['outputs']['hourly']
    if hourly_data:
        df = pd.DataFrame(hourly_data)
else:
    print("No outputs")


df['time'] = df['time'].apply(lambda x: x[:-2] + '00' if x.endswith('11') else x)
df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
df = df.drop(columns=['WS10m','Int'])
print(df)
df.to_csv("C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/Data_creation/DNI.csv", index=False)
print('CSV saved')