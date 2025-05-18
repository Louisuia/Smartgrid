
from math import sin, cos, tan, asin, acos, atan, radians, atan2, pi
import numpy as np
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

DEG_TO_RAD = pi/180
RAD_TO_DEG = 180/pi

def calculate_declination(day_of_year):
    # Relative to the equator
    return 23.45 * sin((360 / 365) * (284 + day_of_year) * DEG_TO_RAD)

def calculate_equation_of_time(day_of_year):
    # Variation of a few minutes throughout the day
    b = (day_of_year - 81) * 360 / 364
    return 9.87 * sin(2 * b * DEG_TO_RAD) - 7.53 * cos(b * DEG_TO_RAD) - 1.5 * sin(b * DEG_TO_RAD)


def calculate_day_duration(lat, day_of_year):
    cos_hr = -tan(lat * DEG_TO_RAD) * tan(calculate_declination(day_of_year) * DEG_TO_RAD)
    if abs(cos_hr) > 1:
        Hr = 0
    else:
        Hr = acos(cos_hr)
    
    T_day = 12 - (Hr * RAD_TO_DEG / 15)
    T_night = 12 + (Hr * RAD_TO_DEG / 15)
    
    D_day = T_night - T_day
    return D_day

def calculate_solar_coor(lat, lon, day_of_year, time_hours):
    decl = calculate_declination(day_of_year) * DEG_TO_RAD
    solar_time = time_hours + (lon / 15) - (calculate_equation_of_time(day_of_year) / 60)
    Hr = 15 * (solar_time - 12)
    Hr = Hr * DEG_TO_RAD
    lat = lat * DEG_TO_RAD
    
    alt = asin(sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(Hr))
    alt = alt * RAD_TO_DEG
    
    az = atan2(sin(Hr), (sin(lat) * cos(Hr) - cos(lat) * tan(decl))) * RAD_TO_DEG
    if az < 0 and Hr * RAD_TO_DEG > 0:
        az += 180
    if az > 0 and Hr * RAD_TO_DEG < 0:
        az -= 180
    
    return alt, az

def global_trajectory(lat, lon, day_of_year,scale=10):
    times = [x * (1 / scale) for x in range(0, 24 * scale)] 
    azimuth = []
    altitude = []
    for time in times:
        alt, az = calculate_solar_coor(lat, lon, day_of_year, time)
        if alt > 0:
            altitude.append(alt), azimuth.append(az)
        else:
            altitude.append(None), azimuth.append(None)
    return azimuth, altitude, times

############################# Coordinates ###############################

lat, lon = 58.3405, 8.59343 # Grimstad
#lat, lon = 69.3856, 18.57183
period = 365

for day in [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]:
    azimuth, altitude, times = global_trajectory(lat, lon, day)
    plt.plot(times, altitude, label=f'Day {day}')

plt.xlabel("Hour")
plt.ylabel("Altitude du Soleil (°)")
plt.title("Trajectory of the sun through the months")
plt.legend()
plt.grid()
plt.show()

days = np.arange(1, 366)
equation_of_time = [calculate_equation_of_time(day) for day in days]

plt.figure(figsize=(10, 5))
plt.plot(days, equation_of_time, label="Equation of Time", color='purple')
plt.xlabel("Day of the Year")
plt.ylabel("Time Variation (minutes)")
plt.title("Equation of Time Over a Year")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8, label="Mean Solar Time")
plt.legend()
plt.grid(True)
plt.show()

def plot_daylight_duration(lat):
    days = np.arange(1, 366)
    durations = [calculate_day_duration(lat, day) for day in days]

    plt.figure(figsize=(10, 5))
    plt.plot(days, durations, label="Day duration", color="orange")

    solstices = [80, 172, 266, 355]
    labels = ["Spring Equinox", "Summer Solstice", "Autumn Equinox", "Winter Solstice"]
    for i, s in enumerate(solstices):
        plt.axvline(x=s, color="gray", linestyle="dashed", alpha=0.7)
        plt.text(s, durations[s-1] + 0.5, labels[i], ha="center", fontsize=10)

    plt.xlabel("Day")
    plt.ylabel("Day duration (hour)")
    plt.title(f"Evolution of Day Length Over a Year (Latitude {lat}°)")
    plt.ylim(0, 24)
    plt.grid(True)
    plt.legend()
    plt.show()

plot_daylight_duration(lat)

################################## PV Power ####################################

def supply_factor(az, alt, incl_x, incl_y):
    pf = []
    for i in range(len(alt)):
        if az[i] is not None and alt[i] is not None:
            theta = acos(
                sin(alt[i] * DEG_TO_RAD) * cos(incl_x * DEG_TO_RAD) +
                cos(alt[i] * DEG_TO_RAD) * sin(incl_x * DEG_TO_RAD) * cos((az[i] - incl_y) * DEG_TO_RAD)
            )
            value = max(0, cos(theta))
            pf.append(value)
        else:
            pf.append(None)
    return pf

def find_optimal_tilt(lat, lon, day_of_year, plot):
    range_incl_x = np.arange(0, 90, 1)
    range_incl_y = np.arange(-40, 40, 1)
    range_incl_y = [0] # fix at 0 deg
    global total_power
    total_power = np.zeros((len(range_incl_x), len(range_incl_y)))
    
    for i, incl_x in enumerate(range_incl_x):
        for j, incl_y in enumerate(range_incl_y):
            power_values = []
            for hour in range(0, 24, 1):
                alt, az = calculate_solar_coor(lat, lon, day_of_year, hour)
                su_f = supply_factor([az], [alt], incl_x, incl_y)[0]
                if not np.isnan(su_f):
                    power_values.append(su_f)
            total_power[i, j] = np.mean(power_values) if power_values else 0
    
    best_idx_pf = np.unravel_index(np.argmax(total_power), total_power.shape)
    best_incl_x, best_incl_y = range_incl_x[best_idx_pf[0]], range_incl_y[best_idx_pf[1]]
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(total_power, extent=[-40, 40, 0, 90], origin="lower", aspect="auto", cmap="hot")
        plt.colorbar(label="Average supply factor")
        plt.scatter(best_incl_y, best_incl_x, color='blue', label=f"Optimal: ({best_incl_x}°, {best_incl_y}°)")
        plt.xlabel("Orientation (°)")
        plt.ylabel("Inclination (°)")
        plt.title(f"Optimal tilt for day {day_of_year}")
        plt.legend()
        plt.show()
    
    return best_incl_x, best_incl_y, best_idx_pf


################################ Optimal tilt #################################

optimal_incl = []
optimal_ori = []
mean_pf_optimal = []
duration = 365

find_optimal_tilt(lat, lon, 82, plot=True)

for day_of_year in range(0, duration):
    print(f"Day {day_of_year}")
    x, y, best_idx_pf = find_optimal_tilt(lat, lon, day_of_year, plot=False)
    optimal_incl.append(x)
    optimal_ori.append(y)
    su_f_optimal = total_power[best_idx_pf[0], best_idx_pf[1]]
    mean_pf_optimal.append(su_f_optimal)

mean_incl = np.mean(optimal_incl)
mean_ori = np.mean(optimal_ori)
mean_pf_for_optimal_incl = np.mean(mean_pf_optimal)

plt.figure(figsize=(10, 5))
plt.plot(range(duration), optimal_incl, label="Optimal tilt (°)")
#plt.plot(range(duration), optimal_ori, label="Optimal orientation (°)")
plt.xlabel("Day of the year")
plt.ylabel("Angle (°)")
plt.title("Evolution of optimal tilt and orientation over the year")
plt.legend()
plt.grid()
plt.show()

print(f"Average optimal tilt: {mean_incl}°")
print(f"Average optimal orientation: {mean_ori}°")
print(f"Mean supply factor for optimal tilts: {mean_pf_for_optimal_incl:.3f}")


##################### log data #######################

incl_x = mean_incl
incl_y = mean_ori

csv_filename = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/Data_creation/Su_f_profil.csv"

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    header = ["time", "Su_f"]
    writer.writerow(header)

    for day_of_year in range(duration):
        date = datetime(2023, 1, 1) + timedelta(days=day_of_year)

        az, alt, times = global_trajectory(lat, lon, day_of_year, scale=1)
        su_f_values = supply_factor(az, alt, incl_x, incl_y)
        su_f_values = [0 if v is None else v for v in su_f_values]

        for hour, value in enumerate(su_f_values):
            timestamp = date.replace(hour=hour, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, value])

print(f"Data successfully saved in {csv_filename}")


