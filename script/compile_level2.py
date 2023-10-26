import pandas as pd
import numpy as np
import os, glob, json, sys
import matplotlib.pyplot as plt
from gapfill import predict, update
from compile_level1 import compile_for_gapfilling
from conversions import ET_to_LE

compile_all_stations = False

if 'compile' in sys.argv:
    compile_all_stations = True

with open(os.path.join('..','metadata','wind_dir_sectors.json')) as file:
    wind_dir_sectors = json.load(file)

with open(os.path.join('..','metadata','level2_predictor_lists.json')) as file:
    predictor_lists = json.load(file)

def compile_level2(project_id,
                    variables_for_gapfilling=None,
                    use_predictors=None,
                    alternative_model_path=None,
                    alternative_data_path=None,
                    export_aggregated = True):

    station = project_id.split('_')[0]

    if not variables_for_gapfilling:
        variables_for_gapfilling = ['ET_filtered']

    #Default path for storing level2_data and info file
    outpath = os.path.join('..','data','level2')

    #or set to selected path
    if alternative_data_path:
        outpath = alternative_data_path

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    #default path for saving figures
    fig_path = os.path.join('..','plot','gapfill',station,project_id)

    #or set to selected path
    if alternative_data_path:
        alt_dir = os.path.normpath(alternative_data_path).split(os.path.sep)[-1]
        fig_path = os.path.join('..','plot','gapfill',station,alt_dir)

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    #Loading observed data
    print('Compiling data:')
    observed = compile_for_gapfilling(project_id)

    #Splitting in ecosystems based on predefined wind direction sectors
    for sector in wind_dir_sectors[station]:

        #writing one infofile per project_id and sector
        # info_file = open(os.path.join(outpath,f'{project_id}_{sector}.txt'),'w')
        sector_min,sector_max = wind_dir_sectors[station][sector]

        #Selecting observations in wind dir sector and make new dataframe for storing gapfilled data
        if sector == 'all':
            observed_in_sector = observed
            level2 = observed[variables_for_gapfilling].copy()

        elif sector_min<sector_max:
            observed_in_sector = observed.where((observed.wind_dir>sector_min)&(observed.wind_dir<sector_max))
            level2 = observed_in_sector[variables_for_gapfilling].copy()

        elif sector_min>sector_max:
            observed_in_sector = observed.where((observed.wind_dir>sector_min)|(observed.wind_dir<sector_max))
            level2 = observed_in_sector[variables_for_gapfilling].copy()

        #Gapfilling selected variables
        for var in variables_for_gapfilling:

            # Stating source of data where observation exists
            observations_exists = level2[var].notna().copy()
            original = level2[[var]].copy()
            original[f'{var}_source'] = np.where(observations_exists,
                                                            'obs',
                                                            None)


            print(f'Gapfilling {var}')

            # Linear interpolation between gaps up to 30 min
            # and stating source of data as 'int'
            interpolated = original[[var]].interpolate(limit=1,
                                                    limit_area='inside')

            gapfill_step_1 = update(current = original,
                                    predictions = interpolated,
                                    source_id = 'int',
                                    variable_id = var)

            if use_predictors:
                predictor_list = use_predictors
            else:
                predictor_list = predictor_lists[station]

            data_in = observed[predictor_list].copy()

            if not alternative_model_path:
                model_path = os.path.join('..',
                                        'models',
                                        project_id,
                                        sector,
                                        f'{var}_biomet_RF.sav')
            else:
                model_path = os.path.join(alternative_model_path,sector,f'{var}_biomet_RF.sav')

            predictions = predict(
                                data_in = data_in,
                                response_variable = var,
                                model_path = model_path
                                )

            gapfill_step_2 = update(current = gapfill_step_1,
                                    predictions = predictions,
                                    source_id = 'met',
                                    variable_id = var)

            #update level2 dataframe
            level2[var] = gapfill_step_2[var]
            level2[f'{var}_source'] = gapfill_step_2[f'{var}_source']

            ############################################################
            #################### PLOTTING ##############################
            ############################################################
            fig,ax = plt.subplots(figsize=(15,10))
            gapfill_step_2[var].plot(ax=ax,label='gapfilled',color='tab:blue',alpha=0.7)
            original[var].plot(ax=ax, label='observed',color='tab:blue',marker='o',ms=2, linestyle='None')
            ax.legend()
            ax.grid()
            fig.savefig(os.path.join(fig_path,f'{var}_{sector}.png'))

        plt.close('all')

        #save level2 data
        outfile = os.path.join(outpath,f'{project_id}_{sector}_level2.csv')
        level2.to_csv(outfile)

        #save yearly data if requested
        if export_aggregated:
            level2.resample('Y').sum(numeric_only=True).to_csv(os.path.join('..','data','aggregated',f'{project_id}_{sector}_level2_yearly.csv'))
    return

def compile_all():
    compile_level2(project_id = 'finseflux_database')
    compile_level2(project_id = 'adventdalen_database')
    compile_level2(project_id = 'iskoras_database')
    compile_level2(project_id = 'myr1_database')
    compile_level2(project_id = 'myr2_database')

if compile_all_stations:
    compile_all()
