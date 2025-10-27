import pandas as pd


# INITIAL DATASET - CONTAINS ALL DATA - WE NEED ONLY DE
#

print("STARTED THE INITIAL SCRIPT TO TRIM THE TIME SERIES")
initial_df = pd.read_csv("dataset//raw//time_series_60min_singleindex.csv")

df = initial_df.filter(["utc_timestamp","cet_cest_timestamp","DE_load_actual_entsoe_transparency", "DE_load_forecast_entsoe_transparency", "DE_solar_capacity", 
           "DE_solar_generation_actual", "DE_solar_profile", "DE_wind_capacity", "DE_wind_generation_actual", 
           "DE_wind_profile", "DE_wind_offshore_capacity", "DE_wind_offshore_generation_actual", 
           "DE_wind_offshore_profile", "DE_wind_onshore_capacity", "DE_wind_onshore_generation_actual", 
           "DE_wind_onshore_profile", "DE_50hertz_load_actual_entsoe_transparency", 
           "DE_50hertz_load_forecast_entsoe_transparency", "DE_50hertz_solar_generation_actual", 
           "DE_50hertz_wind_generation_actual", "DE_50hertz_wind_offshore_generation_actual", 
           "DE_50hertz_wind_onshore_generation_actual", "DE_LU_load_actual_entsoe_transparency", 
           "DE_LU_load_forecast_entsoe_transparency", "DE_LU_price_day_ahead", "DE_LU_solar_generation_actual", 
           "DE_LU_wind_generation_actual", "DE_LU_wind_offshore_generation_actual", "DE_LU_wind_onshore_generation_actual", 
           "DE_amprion_load_actual_entsoe_transparency", "DE_amprion_load_forecast_entsoe_transparency", 
           "DE_amprion_solar_generation_actual", "DE_amprion_wind_onshore_generation_actual", 
           "DE_tennet_load_actual_entsoe_transparency", "DE_tennet_load_forecast_entsoe_transparency", 
           "DE_tennet_solar_generation_actual", "DE_tennet_wind_generation_actual", 
           "DE_tennet_wind_offshore_generation_actual", "DE_tennet_wind_onshore_generation_actual", 
           "DE_transnetbw_load_actual_entsoe_transparency", "DE_transnetbw_load_forecast_entsoe_transparency", 
           "DE_transnetbw_solar_generation_actual", "DE_transnetbw_wind_onshore_generation_actual"])

print(df)

#WE SAVE THE DE ONLY COLUMNS FROM THE ENTIRE TIMESERIES INTO THE FOLLOWING CSV

# df.to_csv("only_de_energy.csv",index=False)
# df = pd.read_csv("data//only_de_energy.csv")

#WE TRIM IT ONCE MORE FOR WIND RELATED ONLY FIELDS - we also did only for national not regional
dataset = df[['utc_timestamp','DE_wind_generation_actual', 'DE_wind_capacity', 'DE_wind_offshore_capacity', 'DE_wind_onshore_capacity', 'DE_wind_profile', 'DE_wind_offshore_profile', 'DE_wind_onshore_profile']].copy()
dataset.to_csv("dataset//processed//de_energy_dataset.csv",index=False)
print("Exploratory dataset saved!")
#DROPPING THE FIRST ROW BECAUSE WE DONT NEED IT - ITS 2014 AND TOO MANY NULLS VALUES - DOSENT BRING ANY VALUE
dataset = dataset.drop(0)

#WE ALSO NEED THE WIND SPEED FOR BETTER ACCURACY
#THIS WILL BE ADDED TO THE FINAL DATASET
de_wind_initial = pd.read_csv("dataset//raw//renewables_ninja_country_DE_wind-speed_merra-2_land-wtd.csv")
de_wind_initial['time'] = pd.to_datetime(de_wind_initial['time'],utc = True)

de_temperature_initial = pd.read_csv("dataset//raw//renewables_ninja_country_DE_temperature_merra-2_land-wtd.csv")
de_temperature_initial['time'] = pd.to_datetime(de_temperature_initial['time'],utc = True)

de_air_density_initial = pd.read_csv("dataset//raw//renewables_ninja_country_DE_air-density_merra-2_land-wtd.csv")
de_air_density_initial['time'] = pd.to_datetime(de_air_density_initial['time'],utc = True)


#TRIMMING THE WIND TO FIT THE TIME PERIOD OF THE TIME SERIES
de_wind = de_wind_initial[( de_wind_initial['time'] >= '2015-01-01T00:00:00Z') & (de_wind_initial['time'] <= '2020-09-30T23:00:00Z')].copy()
# print(de_wind.dtypes)
de_wind['time'] = de_wind['time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

de_temperature = de_temperature_initial[( de_temperature_initial['time'] >= '2015-01-01T00:00:00Z') & (de_temperature_initial['time'] <= '2020-09-30T23:00:00Z')].copy()
de_temperature['time'] = de_temperature['time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

de_air_density = de_air_density_initial[( de_air_density_initial['time'] >= '2015-01-01T00:00:00Z') & (de_air_density_initial['time'] <= '2020-09-30T23:00:00Z')].copy()
de_air_density['time'] = de_air_density['time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')


#ONLY WIND SPEED FOR NATIONAL LEVEL
de_wind = de_wind[['time','DE']].rename(columns = {"DE" : "DE_wind_speed","time": "utc_timestamp"})
de_temperature = de_temperature[['time', 'DE']].rename(columns = {"DE" : "DE_temperature","time": "utc_timestamp"})
de_air_density = de_air_density[['time', 'DE']].rename(columns = {"DE" : "DE_air_density","time": "utc_timestamp"})


#FINAL DATASET
dataset = dataset.merge(de_wind[['utc_timestamp','DE_wind_speed']],on="utc_timestamp",how='left')
dataset = dataset.merge(de_temperature[['utc_timestamp','DE_temperature']],on="utc_timestamp",how='left')
dataset = dataset.merge(de_air_density[['utc_timestamp','DE_air_density']],on="utc_timestamp",how='left')

# dataset.drop(columns=['time'],inplace=True)
# dataset.rename({"DE" : "DE_wind_speed"}, axis= "columns", inplace = True)

print(dataset)


#SAVE TO dataset.csv
dataset.to_csv("dataset//processed//dataset.csv",index = False)