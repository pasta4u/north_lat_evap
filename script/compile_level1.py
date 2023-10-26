import pandas as pd
import numpy as np
import os, glob, json

#import soil porosity
with open("..\\metadata\\soil_porosity.txt") as file:
    soil_porosity = json.load(file)

def get_snow_season(station, index):

    with open(os.path.join('..','..',
                        'latice_flux_sites','data',
                        'snowcover',f'{station}_snowseasons.json')) as file:
        snowfree_seasons = json.load(file)

    snowfree = {}
    shoulder = {}

    #Make new boolean variable with value True for snowfree timesteps
    for year in snowfree_seasons:
        data = snowfree_seasons[year]
        ss_start = pd.to_datetime(data['spring_shoulder_start'], unit='ms')
        ss_end = pd.to_datetime(data['spring_shoulder_end'], unit='ms')
        fs_start = pd.to_datetime(data['fall_shoulder_start'], unit='ms')
        fs_end = pd.to_datetime(data['fall_shoulder_end'], unit='ms')
        snowfree[year]= (index>=ss_end)&(index<=fs_start)
        shoulder[year]=((index>ss_start)&(index<ss_end)) | ((index>fs_start)&(index<fs_end))

    snowfree = pd.DataFrame(snowfree,index=index)
    snowfree = snowfree.any(axis=1)

    shoulder = pd.DataFrame(shoulder,index=index)
    shoulder = shoulder.any(axis=1)
    return snowfree, shoulder

def get_wind_dir_sector(station,wind_dir):
    #get sector definition from metadata
    with open(os.path.join('..','metadata','wind_dir_sectors.json')) as file:
        wind_dir_sectors = json.load(file)
    station_sectors = wind_dir_sectors[station]

    #define new series of strings with same index as original data
    sectors = pd.Series(None,index=wind_dir.index, dtype='string')

    #set the value for the new sector series
    for sector in station_sectors:
        sector_min,sector_max = station_sectors[sector]
        if sector == 'all':
            continue
        elif sector_min<sector_max:
            sector_criterium =  (wind_dir>sector_min)&(wind_dir<sector_max)
        elif sector_min>sector_max:
            sector_criterium = (wind_dir>sector_min)|(wind_dir<sector_max)
        sectors.loc[sector_criterium] = sector
    return sectors

def compile_level1(project_id, 
                    variable_list=None,
                    get_sectors=False):
    '''
    Henter inn filterete eddypro resultat (level1_no_biomet) frå
    EddyPro-kjøringar i mappestruktur: '../../eddypro/station/project_id'.

    Samt filtrerte biomet-data frå mappestruktur '../../wsn/station'

    Bør inkludere plotting av data?
    '''
    #Changeing dir to file dir
    file_path = os.path.realpath(__file__)
    os.chdir(os.path.dirname(file_path))
    station = project_id.split('_')[0]

    #read filtered and timestamp-adjusted eddypro data
    #timestamp is end of period
    eddypro_path = os.path.join('..','..','eddypro')
    project_path = os.path.join(eddypro_path, station, project_id)
    files = glob.glob(os.path.join(project_path,'*level1*.csv'))
    ep_file = files[0]
    print(f'Reading eddypro data from file: {ep_file}')
    ep_df = pd.read_csv(ep_file,
                        index_col=[0],
                        parse_dates=[0],
                        na_values=[])


    #Quickfix to remove duplicates in index
    ep_df.dropna(how='all', inplace=True)
    ep_df = ep_df.loc[~ep_df.index.duplicated(),:]

    #read filtered biomet data averaged to 30 min, timestamp is end of period
    biomet_file = os.path.join('..','..','wsn',station,'level_1_biomet.csv')

    print(f'Reading biomet data from file: {biomet_file}')
    biomet_df = pd.read_csv(biomet_file,
                        index_col=[0],
                        parse_dates=[0])

    #Make continuous timeseries by resampling (will insert np.nan) in timesteps
    #with all data missing
    ep_df = ep_df.asfreq('30T')
    biomet_df = biomet_df.asfreq('30T')

    #concatenate dataframes
    df = ep_df.join(biomet_df, how='outer', rsuffix="_rsuffix")

    #return only selected variables
    if (variable_list is not None):
        df = df[variable_list]

    #Get wind direction sector
    if get_sectors:
        df['sector'] = get_wind_dir_sector(station,df['wind_dir'])
        
    #Get dates of snowfree season for station
    snowfree,shoulder = get_snow_season(station, index = df.index)

    #One-hot encoding of snow cover categories
    # shoulder = 1 if ground is partly or intermittently snow covered
    # snowfree = 1 if ground is continuously and completely snow free.
    zeros = pd.Series(0, index=snowfree.index)
    df['shoulder'] = zeros
    df['snowfree'] = zeros
    df.loc[shoulder,'shoulder'] = 1
    df.loc[snowfree,'snowfree'] = 1

    #Get soil porosity to calculate degree of saturation (from eq. 6-7 in Dingman (2002), 2nd ed.)
    moisture_vars = [var for var in df.columns if var.startswith('SWC')]

    for var in moisture_vars:
        porosity = soil_porosity[station]
        if (var in porosity.keys()):
            df[f'{var}_saturation'] = df[var]/porosity[var]
            #New location of soil water content sensor afer 28.7.2020 13:30.
            if station == 'iskoras':
                df.loc['2020-07-28 13:30':,f'{var}_saturation'] = df[var]/0.95


    return df

def compile_for_gapfilling(project_id, variable_list=None, get_sectors=False):
    '''
    Henter inn filterete eddypro resultat (level1_no_biomet) frå
    EddyPro-kjøringar i mappestruktur: '../../eddypro/station/project_id'.
    Resampler til timesverdiar (middel) for å matche level2-biometdata

    Samt filtrerte og kompletterte biomet-data (level2) frå mappestruktur
    '../../wsn/station'
    '''

    #Changeing dir to file dir
    file_path = os.path.realpath(__file__)
    os.chdir(os.path.dirname(file_path))
    station = project_id.split('_')[0]

    #read filtered and timestamp-adjusted eddypro data
    #timestamp is end of period
    eddypro_path = os.path.join('..','..','eddypro')
    project_path = os.path.join(eddypro_path, station, project_id)
    files = glob.glob(os.path.join(project_path,'*level1*.csv'))
    ep_file = files[0]
    print(f'Reading eddypro level1 data from file: {ep_file}')
    ep_df = pd.read_csv(ep_file,
                        index_col=[0],
                        parse_dates=[0],
                        na_values=[])


    #Quickfix to remove duplicated rows
    ep_df.dropna(how='all', inplace=True)
    ep_df = ep_df.loc[~ep_df.index.duplicated(),:]

    ep_df = ep_df.resample('1H',closed='right',label='right').mean(numeric_only=True)

    #read filtered biomet data averaged to 30 min, timestamp is end of period
    biomet_file = os.path.join('..','..','wsn',station,'level2.csv')
    print(f'Reading biomet level2 data from file: {biomet_file}')
    biomet_df = pd.read_csv(biomet_file,
                        index_col=[0],
                        parse_dates=True)


    #concatenate
    df = ep_df.join(biomet_df, how='outer', rsuffix="_rsuffix")

    #Get dates of snowfree season for station
    snowfree,shoulder = get_snow_season(station, index = df.index)

    #One hot coding of snow cover categories
    # shoulder = 1 if ground is partly or intermittently snow covered
    # snowfree = 1 if ground is continuously and completely snow free.
    zeros = pd.Series(0, index=snowfree.index)
    df['shoulder'] = zeros
    df['snowfree'] = zeros
    df.loc[shoulder,'shoulder'] = 1
    df.loc[snowfree,'snowfree'] = 1

    #calculate albedo
    albedo = df['SWOUT_1_1_1'].where(df['SWOUT_1_1_1']>10)/df['SWIN_1_1_1'].where(df['SWIN_1_1_1']>10)
    albedo = albedo.where(albedo<1)
    #smooth?
    # albedo = albedo.rolling('D',center=True).median()
    albedo.interpolate(inplace=True)
    df['albedo'] = albedo

    #Get wind direction sector
    if get_sectors:
        df['sector'] = get_wind_dir_sector(station,df['wind_dir'])

    #return only selected variables
    if (variable_list is not None):
        df = df[variable_list]

    return df
