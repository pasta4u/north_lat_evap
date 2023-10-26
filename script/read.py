import os,glob
import pandas as pd
import numpy as np
from zipfile import ZipFile

def latice_data(stations = ['hisaasen','finse','iskoras','adventdalen'],
                gapfilled_ET = False):
        
    #Initiating dict to store data
    data_dict = {}

    for station in stations:
        df = pd.read_csv(os.path.join('..','data',f'{station}.csv'),
                        sep=',',
                        header = 0,
                        index_col = 0,
                        parse_dates = True)
        #snowfree season only (if requested)
        if snowfree:
            snowfree_season = df.snowfree == 1
            df = df.where(df['snowfree']==1)
            df.drop(['snowfree','shoulder'], axis=1,inplace=True)
            subset_id='snowfree'
        else:
            subset_id='allYear'

        df['Enet'] = (df.SWin-df.SWout+df.LWin-df.LWout-df.SHF)
        

        #return only oberved ET if requested
        if not gapfilled_ET:
            df = df.loc[df.ET_flag==0,:]

        #save station dataframe to data dict
        data_dict[station] = df.copy()

    return data_dict

def fluxnet(category = 'SUBSET',
            timeres = 'DD', 
            evap=True,
            variables = ['TA_F','LE_F_MDS','SW_IN_F','LW_IN_F',
                        'VPD_F','P_F','ET']):
    date_formats = {'HH': '%Y%m%d%H%M',
                    'DD': '%Y%m%d',
                    'MM': '%Y%m',
                    'YY': '%Y'}

    sites = {}
    sites_extra = {}
    zip_path = os.path.join('..','..','fluxnet','*.zip')
    zipfile_list = glob.glob(zip_path)

    for zipfile in zipfile_list:
        with ZipFile(zipfile) as zf:
            for file in zf.namelist():
                if not f'{category}_{timeres}' in file:
                    continue
                    
                site_ID = '_'.join(file.split('_')[1:2])
                if site_ID != 'SJ-Adv':
                    with zf.open(file) as data:
                        df = pd.read_csv(data, 
                                        index_col=False)
                        if timeres=='HH':
                            df['datetime'] = pd.to_datetime(df['TIMESTAMP_END'], format=date_formats[timeres])
                            df.drop(['TIMESTAMP_START','TIMESTAMP_END'], axis=1, inplace=True)
                        else:
                            df['datetime'] = pd.to_datetime(df['TIMESTAMP'], format=date_formats[timeres])
                            df.drop('TIMESTAMP', axis=1, inplace=True)

                        df.set_index('datetime',inplace=True)

                        #Convert from LE to ET
                        #find conversion factor hours in timeres
                        if (timeres == 'HH'):
                            conversion_factor = 1
                        elif (timeres == 'DD'):
                            conversion_factor = 24
                        elif (timeres == 'MM'):
                            conversion_factor = df.index.days_in_month*24
                        elif (timeres == 'YY'):
                            conversion_factor = np.where(df.index.is_leap_year,366*24,365*24)
                        
                        df['ET'] = conversion_factor*LE_to_ET(LE = df['LE_F_MDS'], temperature = df['TA_F'])

                        #monthly precipitation  in fluxnet files is mean of daily precipitation (mm/day)
                        if 'P_F' in df.columns:
                            df['P_F'] = df['P_F']*df.index.days_in_month
                        
                        #add to dict
                        sites[site_ID] = df[variables]
    sites = pd.concat(sites,axis=1)
    return sites

def fluxnet_metadata():
    file_path = os.path.join('..','..','fluxnet','fluxnet_above_LAT60.csv')
    metadata = pd.read_csv(file_path, sep=';',index_col=0)
    return metadata

def frost_monthly():
    source_to_station = {'SN25830': 'finseflux',
                        'SN97251':'iskoras',
                        'SN99840':'adventdalen',
                        'SN180':'myr1',
                        'SN180':'myr2'
                        }
    sources = ['SN25830', 'SN180', 'SN97251','SN99840']
    data = {}
    for source in sources:
        file_pattern = os.path.join('..','..','frost','data','raw',
                                    'monthly',source,
                                    f'monthly*.csv')
        files = glob.glob(file_pattern)
        frames = []
        for file in files:
            variable = os.path.basename(file).split('_')[1]
            df = pd.read_csv(file,
                            header=0,
                            index_col=0,
                            parse_dates=True)
            df = df.loc[(df.timeResolution=='P1M')&(df.timeOffset=='PT6H'),['value']]
            df.rename(columns={'value': variable},inplace=True)
            df.index = pd.to_datetime(df.index.strftime('%Y-%m-%d'), format = '%Y-%m-%d')
            frames.append(df)
        dfs = pd.concat(frames,axis=1)
        data[source_to_station[source]] = dfs
    return data

def frost_climate_reference(for_latice_stations=True):
    sources = ['SN25830', 'SN180', 'SN97251', 'SN97710','SN99840']
    elementId_to_variable = {'sum(precipitation_amount P1M)': 'precipitation',
                            'mean(air_temperature P1M)': 'temperature'}
    reference_data = {}
    for source in sources:
        file_pattern = os.path.join('..','..','frost','data','raw',
                                    'mean_monthly',
                                    f'{source}_*TA_P_1991_2020.csv')
        file = glob.glob(file_pattern)

        if len(file) == 1:
            df = pd.read_csv(file[0],index_col='month',usecols=['month','elementId','normal'])
            var_dfs = {}
            for element,data in df.groupby(df.elementId):
                var_dfs[elementId_to_variable[element]] = data.normal

            reference_data[source] = pd.DataFrame(var_dfs,index=data.index)

    if for_latice_stations:
        iskoras_temperature = pd.concat([reference_data['SN97251'].temperature,
                                        reference_data['SN97710'].temperature],
                                        axis=1).mean(axis=1)
        latice_reference_data = {
                                'myr1': reference_data['SN180'],
                                'myr2': reference_data['SN180'],
                                'finseflux': reference_data['SN25830'],
                                'iskoras': pd.DataFrame({'temperature':iskoras_temperature,
                                                    'precipitation': reference_data['SN97251'].precipitation}),
                                'adventdalen': reference_data['SN99840']
                                }
        return latice_reference_data

    else:    
        return reference_data