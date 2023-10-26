import pandas as pd
import numpy as np


def growing_degree_day(temperature):
    orig_freq = temperature.index.freq
    if not orig_freq:
        orig_freq = temperature.index[1]-temperature.index[0]
    Tmax = temperature.resample('D',closed='left',label='left').max()
    Tmin = temperature.resample('D',closed='left',label='left').min()
    GDD = (Tmax + Tmin)/2
    GDD = GDD.where(GDD>0, other = 0)

    #cummulated for each year
    cumsum = []
    for year, data in GDD.groupby(GDD.index.year):
        cumsum.append(data.cumsum())

    GDD = pd.concat(cumsum, axis=0)
    GDD = GDD.resample('H').ffill()
    return GDD

def time_since_rain(rain, P_colname = 'P_1_1_1', threshold = 0.1):
    '''
    calculates time since precipitation event

    rain (pandas DataFrame): dataframe with precipitation data in column
    'P_colname' and datetime index.

    P_colname (str): column name of precipitation data in dataframe. Default is
    'P_1_1_1'

    threshold (float): the threshold in mm of what should be counted as an
    precipitation event. Default is 0.1 mm
    '''
    df = rain.copy()
    #set threshold for defining rain event in mm
    df['rain_event'] = df[P_colname]>threshold

    #estimate time since rain event
    df['date'] = df.index
    time = df.date - df.date.where(df['rain_event']).ffill()

    #Convert from pandas timedelta to int (number of hours)
    time = time/np.timedelta64(1,'h')

    return time
