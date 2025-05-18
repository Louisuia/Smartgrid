from matplotlib import pyplot as plt
from numpy import size,shape
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.dates import DateFormatter

print("Code starts")

csv_filename1 = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/Data_creation/Su_f_profil.csv"
csv_filename2 = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/Data_creation/DNI.csv"
csv_filename3 = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/Data_creation/load_profil.csv"
csv_filename4 = "C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/Data_creation/NO2_2023.csv"

df_Su_f = pd.read_csv(csv_filename1)
df_DNI = pd.read_csv(csv_filename2)
df_load = pd.read_csv(csv_filename3)
df_price= pd.read_csv(csv_filename4)

#monocristallin 16-24%
#polycristalin 13-18%
#Amorphe 10%

############### Parameters #############
efficiency=0.18 # PV
pv_area=40

# Temp losses
temp_coef=0.004 # per deg
temp_stc = 25 # standard test conditions in deg 
noct = 45  # Nominal Operating Cell Temperature in deg 
temp_cell_ref = 20 # ref temp for noct
ir_cell_ref = 800 # Nominal irradiance

# EV specs 
ev_avg_cons = 13200/100 # Wh/km
ev_daily_range = 47.2*1.5 # safety coef
ev_daily_cons = ev_avg_cons*ev_daily_range
start_charge = 19 #h (EV starts charging after this hour)
stop_charge = 7

# Wall box specs (7.4 KWh)
charger_eff = 0.9 # %
charger_power = 6600 # Wh (max EV)


############## Create Data base ##############
df_dt=df_DNI
df_dt['Su_f']=df_Su_f['Su_f']
df_dt['Consumption']=round(df_load['Power consumption'],4)
df_dt['Price kWh']=df_price['Price kWh']

#Add EV 

def process_charging(df_dt, opt_charge_mode, opt_charge_csv="C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/ML/Opt_charge_2023.csv"):

    df_dt['Charger_cons'] = 0.0
    df_dt['EV_SoC'] = 0.0

    if opt_charge_mode:
        df_opt = pd.read_csv(opt_charge_csv)
        df_opt['time'] = pd.to_datetime(df_opt['time'])
        df_opt = df_opt.set_index('time')

    for i in range(1,df_dt.shape[0]):
        hour = pd.to_datetime(df_dt.at[i, 'time']).hour
        current_time = pd.to_datetime(df_dt.at[i, 'time'])

        if opt_charge_mode:
            charge_allowed = df_opt.at[current_time, 'opt_charge']
            if df_opt.at[current_time-pd.Timedelta(hours=1), 'opt_charge']:
                charge_allowed = True
        else:
            charge_allowed = (start_charge <= hour or hour < stop_charge)


        if charge_allowed:
            if i ==0:
                prev_soc = 0
            else:
                prev_soc = df_dt.at[i - 1, 'EV_SoC']

            if prev_soc < ev_daily_cons:
                charge_possible = min(charger_power * charger_eff, ev_daily_cons - prev_soc)
                df_dt.at[i, 'Charger_cons'] = charge_possible / charger_eff
                df_dt.at[i, 'EV_SoC'] = prev_soc + charge_possible
            else:
                df_dt.at[i, 'Charger_cons'] = 0
                df_dt.at[i, 'EV_SoC'] = prev_soc
        
        else :
            df_dt.at[i, 'EV_SoC'] = 0

    print(f"Charging cost over the year with optimization = {opt_charge_mode} : ",(df_dt['Charger_cons']*df_dt['Price kWh']/1000).sum())

    return df_dt

process_charging(df_dt, True)


df_dt['Total_cons']=df_dt['Consumption']+df_dt['Charger_cons']
df_dt['T_cell']=df_dt['T2m'] + ((noct - temp_cell_ref) / ir_cell_ref) * df_dt['G(i)']
df_dt['Supply']=df_dt['Su_f']*df_dt['G(i)']*(efficiency*(1-(temp_coef*abs(df_dt['T_cell']-temp_stc))))

df_dt['Supply']=df_dt['Supply']*pv_area
df_dt['Delta cons-sup']=df_dt['Consumption']-df_dt['Supply']
df_dt.to_csv('C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/Data_frame.csv', index=False)
print('CSV saved')

############### PERF ###############

def perf_energy_analysis(df_dt, use_pv, use_ev, use_batt, aff=False):
    fees = 0.004941  # euros per kWh
    contract = 8.50  # euros per month
    batt_capacity_Wh = 14000  # Battery capacity in Wh
    batt_efficiency = 0.95  # Battery efficiency
    soc_batt = 0.0  # Initial state of charge
    
    losses_year_wh = 0
    losses_year_euro = 0
    supply_year_wh = 0
    supply_year_euro = 0
    cons_year_wh = 0
    cons_year_euro = 0
    cons_year_euro_withoutPV = 0
    grid_import_wh = 0

    df_dt["Batt_SoC"] = 0.0
    df_dt["Grid_export"] = 0.0

    for i in range(df_dt.shape[0]):
        price_kwh = df_dt['Price kWh'][i]
        
        # Déterminer la consommation selon l'usage EV ou non
        if use_ev:
            consumption = df_dt['Total_cons'][i]
        else:
            consumption = df_dt['Consumption'][i]
        
        cons_year_wh += consumption
        cons_year_euro_withoutPV += (consumption / 1000) * price_kwh
        
        supply = df_dt['Supply'][i] if use_pv else 0
        supply_year_wh += supply
        
        delta = supply - consumption
        
        grid_export_now = 0.0  # export courant

        if use_batt:
            if delta > 0:
                surplus = delta * batt_efficiency
                if soc_batt + surplus <= batt_capacity_Wh:
                    soc_batt += surplus
                else:
                    excess = (soc_batt + surplus) - batt_capacity_Wh
                    soc_batt = batt_capacity_Wh
                    losses_year_wh += excess
                    losses_year_euro += (excess / 1000) * price_kwh
                    grid_export_now = excess / batt_efficiency
            else:
                if soc_batt >= abs(delta):
                    soc_batt -= abs(delta) / batt_efficiency
                else:
                    grid_needed = abs(delta) - soc_batt * batt_efficiency
                    soc_batt = 0
                    grid_import_wh += grid_needed
                    cons_year_euro += (grid_needed / 1000) * price_kwh
        else:
            if delta < 0:
                grid_import_wh += abs(delta)
                cons_year_euro += (abs(delta) / 1000) * price_kwh
            else:
                losses_year_wh += abs(delta)
                losses_year_euro += (abs(delta) / 1000) * price_kwh
                grid_export_now = delta

        df_dt.loc[i, "Batt_SoC"] = float(soc_batt)
        df_dt.loc[i, "Grid_export"] = grid_export_now

    total_fees = ((grid_import_wh + losses_year_wh) / 1000) * fees
    total_cost_with_op = (contract * 12) + total_fees + cons_year_euro - losses_year_euro
    total_cost_without_op = (contract * 12) + (cons_year_wh / 1000) * fees + cons_year_euro_withoutPV

    if aff:
        print("====================================")
        print(f"      PERFORMANCE ANALYSIS")
        print("====================================")
        print(f"Total consumption/year: {cons_year_wh / 1000:.2f} kWh")
        print(f"Total PV production/year: {supply_year_wh / 1000:.2f} kWh")
        print("------------------------------------")
        print(f"Total overprod/year: {losses_year_wh / 1000:.2f} kWh sold for {losses_year_euro:.2f} euros in the grid")
        print(f"Losses/year: {(losses_year_wh / supply_year_wh) * 100 if supply_year_wh > 0 else 0:.2f} % of PV production not directly used")
        print("------------------------------------")
        print(f"Energy bought from the grid: {grid_import_wh / 1000:.2f} kWh")
        print(f"Electricity cost/year: {cons_year_euro:.2f} €")
        print("------------------------------------")
        print(f"Total fixed fees: {contract * 12:.2f} €")
        print(f"Network surcharge: {total_fees:.2f} €")
        print(f"Total cost with selected options: {total_cost_with_op:.2f} €")
        print(f"Total cost without PV: {total_cost_without_op:.2f} €")

    df_dt.to_csv('C:/Users/louis/OneDrive/Documents/pro/isat/4A/UIA/Smart_grid/code/Data_frame.csv', index=False)

    return total_cost_with_op, total_cost_without_op

# perf_energy_analysis(df_dt,use_pv=True,use_ev=True,use_batt=True,aff=True)
# perf_energy_analysis(df_dt,use_pv=True,use_ev=True,use_batt=True,aff=True)

def compare_all_options(df_dt):
    configurations = [
        (False, False, False), (True, False, False), (False, True, False), (False, False, True),
        (True, True, False), (True, False, True), (False, True, True), (True, True, True)
    ]
    
    results = []
    
    for use_pv, use_ev, use_batt in configurations:
        total_cost, base_cost = perf_energy_analysis(df_dt, use_pv, use_ev, use_batt)
        savings = base_cost - total_cost
        results.append({
            "PV": use_pv, "EV": use_ev, "Battery": use_batt,
            "Base cost (€)": round(base_cost,2),
            "Total Cost (€)": round(total_cost, 2),
            "Savings (€)": round(savings, 2),
            "Percentage of Saving (%)":round((savings/base_cost)*100,2)
        })
    
    return pd.DataFrame(results)

#print(compare_all_options(df_dt))




############## GRAPH #################

def plot_energy_comparison(df_dt, selected_days):
    df_dt['time'] = pd.to_datetime(df_dt['time'])
    df_dt.set_index('time', inplace=True)
    
    fig, axes = plt.subplots(len(selected_days), 1, figsize=(12, 6 * len(selected_days)))
    
    if len(selected_days) == 1:
        axes = [axes]
    
    for i, day in enumerate(selected_days):
        start_time = pd.to_datetime(day) + pd.Timedelta(hours=7)
        end_time = start_time + pd.Timedelta(hours=24)
        
        day_data = df_dt.loc[(df_dt.index >= start_time) & (df_dt.index < end_time)].copy()
        
        day_data['Energy_bought'] = day_data['Total_cons'] - (day_data['Supply'] + day_data['Batt_SoC'])
        day_data['Energy_bought'] = day_data['Energy_bought'].apply(lambda x: max(x, 0))
        
        day_data['Energy_sold'] = day_data['Supply'] - (day_data['Total_cons'] + day_data['Batt_SoC'])
        day_data['Energy_sold'] = day_data['Energy_sold'].apply(lambda x: max(x, 0))
        
        day_data['Cost'] = (day_data['Energy_bought'] / 1000) * day_data['Price kWh'] - (day_data['Energy_sold'] / 1000) * day_data['Price kWh']
        
        axes[i].plot(day_data.index, day_data['Total_cons'], label='Consumption with EV (W)', color='red')
        axes[i].plot(day_data.index, day_data['Supply'], label='PV supply (W)', color='green')
        axes[i].plot(day_data.index, day_data['Batt_SoC'], label='Battery SoC (Wh)', color='blue', linestyle='dashed')
        
        axes[i].set_xlabel(f'Hours from {start_time.strftime("%Y-%m-%d %H:%M")} to {end_time.strftime("%Y-%m-%d %H:%M")}')
        axes[i].set_ylabel('Power (W) / Storage (Wh)')
        axes[i].legend()
        axes[i].grid()

        total_bought = day_data['Energy_bought'].sum()
        total_sold = day_data['Energy_sold'].sum()
        total_cost = day_data['Cost'].sum()
        
        axes[i].text(0.15, 0.95, f"Buy: {total_bought / 1000:.2f} kWh\nSell: {total_sold / 1000:.2f} kWh\nCost: {total_cost:.3f} €", 
                     transform=axes[i].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='center')

    plt.tight_layout()
    plt.show()

selected_days = ['2023-01-02', '2023-06-01', '2023-09-01']
#plot_energy_comparison(df_dt, selected_days)

def plot_delta(df_dt, day=None):
    df_dt['Date'] = pd.to_datetime(df_dt['time'])

    if day:
        day = pd.to_datetime(day)
        df_dt = df_dt[df_dt['Date'].dt.date == day.date()]


    df_dt.set_index('Date', inplace=True)
    df_time = df_dt.copy()

    x = (df_time.index - df_time.index[0]).total_seconds()
    y = df_time['Delta cons-sup'].values

    interp_func = interp1d(x, y, kind='linear')

    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = interp_func(x_new)
    time_new = df_time.index[0] + pd.to_timedelta(x_new, unit='s')

    mean = round(np.mean(y_new), 2)
    print(f"Average delta: {mean} kWh")

    y_pos = np.where(y_new > 0, y_new, 0)
    y_neg = np.where(y_new < 0, y_new, 0)

    plt.plot(time_new, y_new, label="Delta", color="black", linewidth=1)
    plt.axhline(y=mean, color='blue', linestyle='--', label="Average")
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.fill_between(time_new, y_pos, 0, color='red', alpha=0.3)
    plt.fill_between(time_new, y_neg, 0, color='green', alpha=0.3)
    plt.xlabel("Hour")
    plt.ylabel("Delta in Wh")
    plt.legend()
    plt.tight_layout()
    plt.show()

# plot_delta(df_dt, day="2023-06-01")

def plot_mix_from_df(df_dt, debut=5000, fin=5100):
    # Paramètres pour les calculs PV et EV déjà inclus dans df_dt
    df = df_dt.copy()

    # Paramètres batterie
    batt_capacity = 14000  # Wh
    charging_efficiency = 0.95
    discharging_efficiency = 0.95

    # Simulation batterie simple (pas de stratégie complexe ici)
    soc = 0
    batt_discharge = []
    batt_charge = []

    for i in range(len(df)):
        delta = df["Supply"].iloc[i] - df["Consumption"].iloc[i]
        if delta > 0:
            charge = min(delta * charging_efficiency, batt_capacity - soc)
            batt_discharge.append(0)
            batt_charge.append(-charge / charging_efficiency)
            soc += charge
        else:
            discharge = min(abs(delta) / discharging_efficiency, soc)
            batt_discharge.append(discharge * discharging_efficiency)
            batt_charge.append(0)
            soc -= discharge

    df["battery_discharge"] = batt_discharge
    df["battery_charge"] = batt_charge
    df['Grid_export']= -df['Grid_export']

    df['Prod PV'] = df['Supply']
    df["import"] = np.maximum(0, df["Consumption"] - (df["Supply"] + df["battery_discharge"]))

    colonnes_positives = ["Prod PV", "battery_discharge", "import"]
    colonnes_negatives = ["Grid_export", "battery_charge"]

    colors_positives = ["gold", "blue", "gray"]
    colors_negatives = ["purple", "green"]

    df_positive = df[colonnes_positives]
    df_negative = df[colonnes_negatives]

    plt.figure(figsize=(12, 6))
    plt.stackplot(df['time'][debut:fin], df_positive[debut:fin].T, labels=colonnes_positives, colors=colors_positives, alpha=0.8)
    plt.stackplot(df['time'][debut:fin], df_negative[debut:fin].T, labels=colonnes_negatives, colors=colors_negatives, alpha=0.8)

    # Ajout de la consommation réelle (Total_cons)
    plt.plot(df['time'][debut:fin], df['Consumption'][debut:fin], color='red', label='demand')
    plt.axhline(0, color="black", linewidth=1)

    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.title("Electricity Production and Usage Mix")
    
    # Formatage des ticks pour l'axe x
    indices_ticks = range(debut + debut % 24, fin, 24)
    ticks_values = df['time'].iloc[indices_ticks]
    plt.xticks(ticks_values, labels=[t.strftime('%d/%m') for t in pd.to_datetime(ticks_values)], rotation=45)

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

# plot_mix_from_df(df_dt, debut=5000, fin=5200)

def plot_hist_pv():

    df_dt['time'] = pd.to_datetime(df_dt['time'])

    df_dt['Month'] = df_dt['time'].dt.month

    monthly_production = df_dt.groupby('Month')['Supply'].sum()

    plt.figure(figsize=(10,6))
    bars = plt.bar(monthly_production.index, monthly_production.values/1000, color='#4db8b8')
    plt.bar(monthly_production.index, monthly_production.values/1000, color='#4db8b8')

    plt.title('Monthly PV Production (kWh)', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Production (kWh)', fontsize=12)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(monthly_production.index, month_names)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, round(yval, 2), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

# plot_hist_pv()