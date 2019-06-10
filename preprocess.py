import pandas as pd
import numpy as np

def anomally_z_filter(df,z_thres):
    df['timeInSecondsZ'] = abs(df['timeInSeconds'] - df['timeInSeconds'].mean())/df['timeInSeconds'].std(ddof=0)
    df = df[df['timeInSecondsZ'] < z_thres]
    return df

def timestamp_resample(df,method,start_date,end_date):
    format2 = '%Y-%m-%dT%H:%M:%S-%f'
    df['Datetime'] = pd.to_datetime(df['updated'], format=format2)
    df = df.set_index(pd.DatetimeIndex(df['Datetime']))
    df=df.loc[start_date:end_date]
    df = df[['timeInSeconds']]
    df=df.loc[start_date:end_date]
    df = df.resample(method).mean()
    return df

def day_of_week(df):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i in range(7):
        df[days[i]] = (df.index.dayofweek == i).astype(float)
        df.head()
    return df

def hour_of_day(df):
    df['hourfloat2']=df.index.hour
    df['dummy'] = df['hourfloat2'].apply(lambda x: -4 if x >= 4  else 20) # offsets to 5am instead to at midnight since time is entered here as linear
    df['hourfloat'] = df['dummy'] + df['hourfloat2']
    df = df.drop(labels='hourfloat2', axis=1)
    df = df.drop(labels='dummy', axis=1)
    df['x']=np.sin(2.*np.pi*df.hourfloat/24.) # sine and cosine to ensure time is cyclic
    df['y']=np.cos(2.*np.pi*df.hourfloat/24.)
    df['annual'] = (df.index - df.index[0]).days/365 # annual count of days 
    return df

def weather(df,method,start_date,end_date):
    format3 = '%Y-%m-%d %H:%M'
    df2 = pd.read_csv('/Users/tung-linwu/Desktop/Insight/data/compiled_data/Weather_combined.csv')
    df2['Snow'] = np.where(df2['Weather']=='Snow', 1, 0)
    df2['Datetime'] = pd.to_datetime(df2['Date/Time'], format=format3)
    df2 = df2.set_index(pd.DatetimeIndex(df2['Datetime']))
    df2 = df2[['Temp (°C)','Snow','Visibility (km)']]
    df2 = df2.resample(method).mean()
    df_w=df2.sort_values(by=['Datetime'])
    df_w=df_w.loc[start_date:end_date]
    df=pd.concat((df, df_w), axis=1)
    return df 

def holiday(df,method,start_date,end_date):
    holiday = pd.read_csv('/Users/tung-linwu/Desktop/Insight/data/compiled_data/holiday_combined.csv')
    holiday = holiday.set_index(pd.DatetimeIndex(holiday['Datetime']))
    holiday = holiday[['Event Value']]
    holiday=holiday.loc[start_date:end_date]
    holiday=holiday.resample(method).pad(limit=23)
    df=pd.concat((df, holiday), axis=1)
    df.rename(columns={'Event Value': 'holiday'}, inplace=True)
    return df
    
def NBA(df,method,start_date,end_date):
    nba = pd.read_csv('/Users/tung-linwu/Desktop/Insight/data/compiled_data/nba_combined.csv')
    nba=nba[nba['Home/Neutral'].str.contains("Toronto Raptor")]
    nba['Datetime'] = pd.to_datetime(nba['Date']+' '+nba['Start (ET)'], infer_datetime_format=True)
    nba = nba.set_index(pd.DatetimeIndex(nba['Datetime']))
    nba = nba[['Attend.']]
    nba2=nba.resample(method).backfill(limit=5)# assuming game affects traffic up to 5 hours before starting game time
    nba2=nba2.dropna()
    nba2 = nba2.resample(method).sum() 
    nba2=nba2.loc[start_date:end_date]
    df=pd.concat((df, nba2), axis=1)
    df.rename(columns={'Attend.': 'NBA'}, inplace=True)
    return df

def MLB(df,method,start_date,end_date):
    mlb = pd.read_csv('/Users/tung-linwu/Desktop/Insight/data/compiled_data/mlb_combined_time.csv')
    mlb = mlb.set_index(pd.DatetimeIndex(mlb['Datetime']))
    mlb = mlb[['attendance']]
    mlb2=mlb.resample(method).backfill(limit=5) # assuming game affects traffic up to 5 hours before starting game time
    mlb2=mlb2.dropna()
    mlb2 = mlb2.resample(method).sum()
    mlb2=mlb2.loc[start_date:end_date]
    df=pd.concat((df, mlb2), axis=1)
    df.rename(columns={'attendance': 'MLB'}, inplace=True)
    return df

def NHL(df,method,start_date,end_date):
    nhl = pd.read_csv('/Users/tung-linwu/Desktop/Insight/data/compiled_data/nhl_combined_time.csv')
    nhl = nhl.set_index(pd.DatetimeIndex(nhl['Datetime']))
    nhl = nhl[['attendance']]
    nhl2=nhl.resample(method).backfill(limit=5) # assuming game affects traffic up to 5 hours before starting game time
    nhl2=nhl2.dropna()
    nhl2 = nhl2.resample(method).sum()
    nhl2=nhl2.loc[start_date:end_date]
    df=pd.concat((df, nhl2), axis=1)
    df.rename(columns={'attendance': 'NHL'}, inplace=True)
    return df

def Scotiabank_arena_events(df,method,start_date,end_date):
    events = pd.read_csv('/Users/tung-linwu/Desktop/Insight/data/events/table-2_edit.txt')
    events['Datetime'] = pd.to_datetime(events['Dates'], infer_datetime_format=True)
    events = events.set_index(pd.DatetimeIndex(events['Datetime']))
    events['Events'] = 1
    events = events[['Events']]
    events=events.loc[start_date:end_date]
    events=events.resample(method).pad(limit=24)
    df=pd.concat((df, events), axis=1)
    return df

def cleanup(df):
    df['NBA'].fillna(0, inplace=True)
    df['MLB'].fillna(0, inplace=True)
    df['NHL'].fillna(0, inplace=True)
    df['Events'].fillna(0, inplace=True)
    df['holiday'].fillna(0, inplace=True)
    return df
    
def attendance_to_binary(df):
    df['NBA'] = np.where(df['NBA']>0, 1, 0)
    df['MLB'] = np.where(df['MLB']>0, 1, 0)
    df['NHL'] = np.where(df['NHL']>0, 1, 0)
    df=df.dropna()
    return df


