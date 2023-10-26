# based on a tutorial from https://www.datacamp.com/tutorial/introduction-factor-analysis
# from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity,calculate_kmo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json,os

def factor_analysis(observations,fig_path, data_path):
    with open(os.path.join('..','metadata','level2_predictor_lists.json')) as file:
        predictor_list = json.load(file)

    results = {}

    for project in observations:
        station = project.split('_')[0]
        station_df = observations[project]

        #split in seasons (based on snow cover status)
        snowfree = station_df.snowfree == 1
        shoulder = station_df.shoulder == 1
        snowcovered = (station_df.shoulder == 0) & (station_df.snowfree == 0)
        snowfree_df = station_df.loc[snowfree,:]
        shoulder_df = station_df.loc[shoulder,:]
        snowcovered_df = station_df.loc[snowcovered,:]

        for season,season_df in zip(['snowfree','shoulder','snowcovered'],[snowfree_df,shoulder_df,snowcovered_df]):
            #make lists of sectors and df for each sector
            sectors = []
            dfs = []
            for sector,df in season_df.groupby(season_df.sector):
                sectors.append(sector)
                dfs.append(df)

            #include 'all' (i.e. all wind directions) as sector
            sectors.append('all')
            dfs.append(season_df)
            
            sector_results = {}
            for sector,df in zip(sectors,dfs):
                info = []
                labels = ['ET_filtered']
                features = predictor_list[station]

                #drop some irrelevant features:
                for feat in ['snowfree','shoulder','TS_2_1_1']:
                    if feat in features:
                        features.remove(feat)

                #select columns with variables included in features or labels
                features = [f for f in features if f in df.columns]
                df = df[features+labels].copy()

                #drop INF and NA-values
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(axis=0,how='any',inplace=True)

                #check the 'factorability' of the dataset using Barletts test
                chi_square_value,p_value=calculate_bartlett_sphericity(df)
                info.append(f'barlett_chi_square_value,{chi_square_value}\n')
                info.append(f'barlett_p_value,{p_value}\n')
                if p_value>0.05:
                    info.append('P value for Barlett test >0.05. Analyzis ended.')
                    print(f'{station} (sector {sector},  {season} data) failed Barlett test')
                    with open(os.path.join(data_path,f'factor_analysis_{season}_{station}_{sector}_info.csv'),'w') as info_file:
                        info_file.writelines(info)
                    continue

                #check the 'factorability' of the dataset using Kaiser-Meyer-Olkin test
                kmo_all,kmo_model=calculate_kmo(df)
                info.append(f'kmo_model,{kmo_model}\n')
                if kmo_model<0.6:
                    print(f'OBS: low Kaiser-Meyer-Olkin value (k= {kmo_model:.2f}) for {station} (sector {sector}, {season} data).')
                if kmo_model<0.5:
                    info.append('Kaiser-Meyer-Olkin value < 0.5. Analyzis ended.')
                    print(f'{station} (sector {sector}, {season} data) failed Kaiser-Meyer-Olkin test')
                    with open(os.path.join(data_path,f'factor_analysis_{season}_{station}_{sector}_info.csv'),'w') as info_file:
                        info_file.writelines(info)
                    continue

                #chosing numbers of factors
                fa = FactorAnalyzer(10, rotation=None)
                try:
                    fa.fit(df)
                    eigen_values,vectors = fa.get_eigenvalues()
                    n_factors = len(eigen_values[np.where(eigen_values>1)])
                    info.append(f'n_factors,{n_factors}\n')
                except:
                    n_factors = 2
                    info.append(f'n_factors,{n_factors}\n')
                    info.append('could not fit factor analysis to determine number of factors with eigen value >1. n_factors set to default')
                    
                #performing factor analysis with selected numbers of factors
                fa = FactorAnalyzer(n_factors, rotation='varimax')
                try:
                    fa.fit(df)
                    factor_df = pd.DataFrame(fa.loadings_, index=df.columns)
                except:
                    print(f'Factor analysis failed for {station}, in sector {sector}, using {season} data')
                    info.append(f'Factor analysis failed for {station}, in sector {sector}, using {season} data')
                    with open(os.path.join(data_path,f'factor_analysis_{season}_{station}_{sector}_info.csv'),'w') as info_file:
                        info_file.writelines(info)
                    continue

                #plot
                name_mapper = {var: var.split('_')[0] for var in factor_df.index}
                int_to_ord = {0: 'First factor',
                                1: 'Second factor',
                                2: 'Third factor',
                                3: 'Forth factor'}
                factor_df.rename(name_mapper, axis=0, inplace=True)
                factor_df.rename(int_to_ord, axis=1, inplace=True)

                #plot factor loadings, bar plot, subplot for each factor
                n_rows = len(factor_df.columns)
                fig,axs = plt.subplots(n_rows,1,sharex=True,sharey=True, figsize=(7,n_rows*3))
                for ax,factor in zip(axs,factor_df.columns):
                    ax.bar(factor_df.index,factor_df[factor])
                    ax.set_title(factor)
                    ax.grid()
                ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
                fig.supxlabel('Observed variable')
                fig.supylabel('Factor loading')
                fig.savefig(os.path.join(fig_path,f'factor_analysis_{season}_{station}_{sector}.png'))

                factor_variance = cummulative_variance = fa.get_factor_variance()
                factor_df.loc['total_variance',:] = factor_variance[0] #equal to the sum of square of loadings
                factor_df.loc['proportion_of_variance',:] = factor_variance[1]
                factor_df.loc['cummulative_variance',:] = factor_variance[-1]

                factor_df.to_csv(os.path.join(data_path,f'factor_analysis_{season}_{station}_{sector}.csv'))

                #plot results of first two factors

                with open(os.path.join(data_path,f'factor_analysis_{season}_{station}_{sector}_info.csv'),'w') as info_file:
                    info_file.writelines(info)
                
                sector_results[sector] = factor_df
            results[(season,station)] = sector_results
            plt.close('all')
    return results




