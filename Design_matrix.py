# Design Matrix for Regression Analysis
# This code generates and saves a pandas dataframe of labels abd features for a specific travel route in downtown Toronto before being fed for regression model predictions

import pandas as pd
import preprocess as func

#%% Load data matrix parameters
route_segment = 'LY_D'
df = pd.read_csv('/Users/tung-linwu/Desktop/Insight/data/compiled_data/'+route_segment+'_bt_combined.csv')

#%% Pre-processing
start_date = '2014-04-10'
end_date = '2017-01-31'
resamp_method = 'H'

df = func.anomally_z_filter(df, 3) #Filter out anomally large and small travel times
df = func.timestamp_resample(df, resamp_method, start_date,end_date) #Binning of Data

#%% Add features to design matrix
# Day of the Week
df = func.day_of_week(df)

# Hour of the day
df = func.hour_of_day(df)

# Weather (temperature, snow and visibility)
df = func.weather(df, resamp_method, start_date,end_date)

# Holidays
df = func.holiday(df, resamp_method, start_date, end_date)

# NBA games
df = func.NBA(df, resamp_method, start_date, end_date)

# MLB games
df = func.MLB(df, resamp_method, start_date, end_date)

# NHL games
df = func.NHL(df, resamp_method, start_date,end_date)

# Scotiabank Arena Events
df = func.Scotiabank_arena_events(df, resamp_method, start_date, end_date)

# Replacing backfilled NaNs with 0 and converting attendances to binary
df = func.cleanup(df)
df = func.attendance_to_binary(df)

#%%
# df.to_csv('/Users/tung-linwu/Desktop/Insight/data/design_matrix/D_matrix_' +route_segment+'_bt.csv')
