import statsmodels.api as sm
from scipy.stats import ttest_ind
from matplotlib.legend import _get_legend_handles_labels
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import read
import pandas as pd
import numpy as np
import os,json

plt.rc('axes', labelsize=9)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=9)   # fontsize of the tick labels
plt.rc('ytick', labelsize=9)    # fontsize of the tick labels
plt.rc('legend', fontsize=9)    # legend fontsize
plt.rc('axes', titlesize=9)     # fontsize of the axes title
plt.rc('figure', titlesize=9)  # fontsize of the figure title
plt.rc('figure', labelsize=9)  # fontsize of the figure title

#carbon fluxes are in micromol m-2 s-1.
co2_molar_mass = 44.01 #g/mol
ch4_molar_mass = 16.04 #g/mol

#to covert from /s to /day and /year (and from hourly mean flux to cummulative)
sek_per_hour = 60*60
sek_per_day = sek_per_hour*24
sek_per_year = sek_per_day*365

#dict to store aggregated values
aggregated = {}

#for figure labels and titles
flux_long = {'co2_flux': '$CO_2$',
            'ch4_flux': '$CH_4$',
            'H': 'Sensible Heat',
            'LE': 'Latent heat',
            'ET': 'Evaporation',
            'Rnet': 'Net radiation',
            'Enet': 'Net energy',
            'SWIN': 'Incoming solar radiation',
            'VPD': 'Vapor pressure deficit'}

ylabs_cum = {'co2_flux': '$g m^{-2}$',
            'ch4_flux': '$g m^{-2}$',
            'H': '$MJ m^{-2}$',
            'LE': '$MJ m^{-2}$',
            'ET':'$mm$',
            'SWIN': '$MJ m^{-2}$'}

ylabs_cum_daily = {'ET': '$mm day^{-1}$',
                    'Rnet': '$MJ day^{-1} m^{-2}$',
                    'Enet': '$MJ day^{-1} m^{-2}$',
                    'VPD': '$kPa$'}

ylabs_inst = {
            'LE': '$W m^{-2}$',
            'Rnet': '$W m^{-2}$',
            'Enet': '$W m^{-2}$',
            'ET':'$mm$',
            'VPD': '$kPa$',
            'SWIN': '$W m^{-2}$'}

colorlist = {'iskoras': 'tab:orange',
            'finseflux': 'tab:blue',
            'myr2': 'tab:green',
            'adventdalen': 'tab:red'}

yearly_linestyle = {
                    '2013': 'dotted',
                    '2015': 'solid',
                    '2016': 'dashed',
                    '2019': 'dotted',
                    '2020': 'solid',
                    '2021': 'dashed',
                    '2022': 'dashdot'
                    }

titlelist = {'iskoras': 'I\u0161koras',
            'finseflux': 'Finse',
            'myr2': 'Hisåsen',
            'adventdalen': 'Adventdalen'}

ylims = {'LE': (-10,550),
        'H': (-400,200),
        'ET':(-10,210),
        'SWin': (0,4500)}

daily_coeffs = {'LE_filtered': 0.000001*60*60*24,   #W/m2 to MJ/day m2
                'ET': 24,                  # mm/hour to mm/day
                'SWin': 0.000001*60*60*24,     #W/m2 to MJ/day m2
                'Enet': 0.000001*60*60*24,     #W/m2 to MJ/day m2
                'Rnet': 0.000001*60*60*24     #W/m2 to MJ/day m2
                }

TA_clim = {'iskoras': -1.7,
        'finseflux': -1.1,
        'myr2': 2.7,
        'adventdalen': -3.9}

P_clim = {'iskoras': 420,
        'finseflux': 970,
        'myr2': 860,
        'adventdalen': 220}

#plot yearly cummulated evaporation and duration of snow cover
def yearly_cummulation_snow(level2, fig_path):
    ylim_top = (-10,205)
    snowfree = {}
    for station in level2:
        with open(os.path.join('..','..',
                            'latice_flux_sites','data',
                            'snowcover',f'{station}_snowseasons.json')) as file:
            snowfree_seasons = json.load(file)
            snowfree[station] = snowfree_seasons

    n_sites = len(level2)
    flux = 'ET'

    fig,axs = plt.subplots(2,n_sites, sharex=True, figsize = (7,4), height_ratios = [3,1])
    for col,station in zip(range(n_sites),level2):
        df = level2[station][flux]
        years=[]
        for year,data in df.groupby(df.index.year):
            #Find daily mean for plotting vs DOY
            daily_mean = data.resample('D').mean()
            #convert flux from W/m2 to MJ/m2  and plot cummulative sum
            coeff = daily_coeffs[flux]
            cumsum = (daily_mean*coeff).cumsum()
            axs[0,col].plot(cumsum.index.dayofyear, cumsum, color=colorlist[station],
                    linestyle = yearly_linestyle[str(year)],
                    linewidth=0.75,
                    label = f'{str(year)}')
            #Get snowfree season
            snowfree_day_spring = snowfree[station][str(year)]['spring_shoulder_end']
            snowfree_day_spring = pd.to_datetime(snowfree_day_spring, unit='ms')
            snowfree_day_fall = snowfree[station][str(year)]['fall_shoulder_start']
            snowfree_day_fall = pd.to_datetime(snowfree_day_fall, unit='ms')
            #Get snowcovered season
            snowcovered_day_spring = snowfree[station][str(year)]['spring_shoulder_start']
            snowcovered_day_spring = pd.to_datetime(snowcovered_day_spring, unit='ms')
            snowcovered_day_fall = snowfree[station][str(year)]['fall_shoulder_end']
            snowcovered_day_fall = pd.to_datetime(snowcovered_day_fall, unit='ms')
            #plot snowcovered season
            axs[1,col].fill_between(x=range(0,snowcovered_day_spring.dayofyear), y1=year-0.2, y2=year+0.2,
                                    facecolor = 'snow', edgecolor='k', alpha=0.6, label = 'Snow-covered')
            axs[1,col].fill_between(x=range(snowcovered_day_fall.dayofyear,365), y1=year-0.2, y2=year+0.2,
                                    facecolor = 'snow', edgecolor='k', alpha=0.6, label = '')
            #plot shoulder season
            axs[1,col].fill_between(x=range(snowcovered_day_spring.dayofyear,snowfree_day_spring.dayofyear), y1=year-0.2, y2=year+0.2,
                                    facecolor = 'tan', edgecolor='k', alpha=0.6, label = 'Partly snow-covered')
            axs[1,col].fill_between(x=range(snowfree_day_fall.dayofyear,snowcovered_day_fall.dayofyear), y1=year-0.2, y2=year+0.2, 
                                    facecolor = 'tan', edgecolor='k', alpha=0.6, label = '')
            #plot snowfree season
            axs[1,col].fill_between(x=range(snowfree_day_spring.dayofyear,snowfree_day_fall.dayofyear), y1=year-0.2, y2=year+0.2,
                                    facecolor = 'limegreen', edgecolor='k', alpha=0.6, label = 'Snow-free')
            years.append(year)
        axs[0,col].grid()
        axs[0,col].set_ylim(ylim_top)
        axs[1,col].grid()
        axs[0,col].set_title(titlelist[station])
        axs[0,col].set_xticks([1,60,121,182,244,305])
        axs[1,col].set_xticks([1,60,121,182,244,305])
        axs[1,col].set_yticks(years)
        axs[1,col].set_yticklabels([str(y) for y in years], fontsize=8, rotation = 45)
        axs[1,col].set_xticklabels(['Jan','Mar','May','Jul','Sep','Nov'], rotation = 90)
        axs[1,col].invert_yaxis()
        # ax.set_xticks(np.linspace(0,335,12))
        # ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], rotation = 90)
        axs[0,col].set_ylim(-10,250)
        axs[0,col].legend(loc='upper left', fontsize=8,)
    axs[0,0].set_ylabel(f'{flux_long["ET"]} in {ylabs_cum["ET"]}')
    axs[1,0].set_ylabel(f'Year')
    fig.tight_layout()
    handles,labels = axs[1,col].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.subplots_adjust(bottom=0.2)
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3)
    axs[0,0].set_xlim(0,365)
    fig.savefig(os.path.join(fig_path,'annual','yearly_cummulation_ET2.pdf'))
    plt.close(fig)
    return fig

#make table of yearly total evaporation, mean incoming and net radiation and VPD
def yearly_magnitudes(level2, table_path):
    for station in level2:
        ET = daily_coeffs['ET']*level2[station].loc[:,['ET']].resample('D').mean()
        ET = ET.resample('Y').sum(min_count=365)
        rad = daily_coeffs['SWin']*level2[station].loc[:,['SWin']].resample('D').mean()
        rad = rad.resample('Y').sum(min_count=365)
        rnet = daily_coeffs['SWin']*level2[station].loc[:,['Rnet']].resample('D').mean()
        rnet = rnet.resample('Y').sum(min_count=365)
        VPD = level2[station].loc[:,['VPD']].resample('Y').mean()
        df = pd.concat([ET,rad,rnet,VPD],axis=1)
        df.dropna(axis=0, how='all', inplace=True)
        df.to_csv(os.path.join(table_path,'annual',f'{station}_yearly_totals.csv'), float_format = '%.2f')
    return

def snowfree_mean_magnitudes(level2, table_path):
    table_path = os.path.join(table_path,'snowfree_snowcovered_stats')
    if not os.path.exists(table_path):
        os.makedirs(table_path)

    snowfree = {}
    for station in level2:
        with open(os.path.join('..','..',
                            'latice_flux_sites','data',
                            'snowcover',f'{station}_snowseasons.json')) as file:
            snowfree_seasons = json.load(file)
            snowfree[station] = snowfree_seasons

    all_sites_snowfree = {}
    all_sites_snowcovered = {}
    all_sites_shoulder = {}
    all_sites_all_year = {}

    for station in level2:
        #dicts for saving stats per year and per snow category
        snowfree_stats = {}
        snowcovered_stats = {}
        shoulder_stats = {}
        all_year_stats = {}
        #lists for dataframes per snow category
        snowfree_dfs = []
        snowcovered_dfs = []
        shoulder_dfs = []
        all_year_dfs = []
        for year,data in level2[station].loc[:,['ET','VPD','SWin','Rnet']].groupby(level2[station].index.year):
            #get snow cover dates
            snowcovered_day_spring = pd.to_datetime(snowfree[station][str(year)]['spring_shoulder_start'], unit='ms')
            snowcovered_day_fall = pd.to_datetime(snowfree[station][str(year)]['fall_shoulder_end'], unit='ms')
            snowfree_day_spring = pd.to_datetime(snowfree[station][str(year)]['spring_shoulder_end'], unit='ms')
            snowfree_day_fall = pd.to_datetime(snowfree[station][str(year)]['fall_shoulder_start'], unit='ms')
    
            #resample to daily mean
            df = data.resample('D').mean()
    
            #change to daily cummulative values for ET and SWIN
            df.loc[:,'ET'] *= daily_coeffs['ET']
            df.loc[:,'SWin'] *= daily_coeffs['SWin']
            df.loc[:,'Rnet'] *= daily_coeffs['SWin']
            df_snowfree = df.loc[snowfree_day_spring:snowfree_day_fall,:].copy()
            df_snowcovered = pd.concat([df.loc[str(year)+'-01-01':snowcovered_day_spring],
                                        df.loc[snowcovered_day_fall:str(year)+'-12-31']],axis=0).copy()
            df_shoulder = pd.concat([df.loc[snowcovered_day_spring:snowfree_day_spring],
                                        df.loc[snowfree_day_fall:snowcovered_day_fall]],axis=0).copy()
                                          
            #get sum, mean and std of daily values for each year
            snowfree_stats[year] = df_snowfree.resample('Y').sum().join(df_snowfree.resample('Y').mean().join(df_snowfree.resample('Y').std(),lsuffix='_mean',rsuffix='_std'),lsuffix='_sum')
            snowcovered_stats[year] = df_snowcovered.resample('Y').sum().join(df_snowcovered.resample('Y').mean().join(df_snowcovered.resample('Y').std(),lsuffix='_mean',rsuffix='_std'),lsuffix='_sum')
            shoulder_stats[year] = df_shoulder.resample('Y').sum().join(df_shoulder.resample('Y').mean().join(df_shoulder.resample('Y').std(),lsuffix='_mean',rsuffix='_std'),lsuffix='_sum')
            all_year_stats[year] = df.resample('Y').sum().join(df.resample('Y').mean().join(df.resample('Y').std(),lsuffix='_mean',rsuffix='_std'),lsuffix='_sum')

            #save dfs for across year stats
        snowfree_dfs.append(df_snowfree)
        snowcovered_dfs.append(df_snowcovered)
        shoulder_dfs.append(df_shoulder)
        all_year_dfs.append(df)
    
        #save per-year stats
        pd.concat(snowfree_stats).to_csv(os.path.join(table_path,f'{station}_snowfree_stats.csv'), float_format = '%.2f')
        pd.concat(snowcovered_stats).to_csv(os.path.join(table_path,f'{station}_snowcovered_stats.csv'), float_format = '%.2f')
        pd.concat(shoulder_stats).to_csv(os.path.join(table_path,f'{station}_shoulder_stats.csv'), float_format = '%.2f')
        pd.concat(all_year_stats).to_csv(os.path.join(table_path,f'{station}_all_year_stats.csv'), float_format = '%.2f')
    
        #concat data for all year for snowfree, snowcovered and all year
        snowfree_df = pd.concat(snowfree_dfs)
        snowcovered_df = pd.concat(snowcovered_dfs)
        shoulder_df = pd.concat(shoulder_dfs)
        all_year_df = pd.concat(all_year_dfs)
    
        #get mean and std per station
        all_sites_snowfree[station] = pd.DataFrame({'mean':snowfree_df.mean(),'std' :snowfree_df.std()}).transpose()
        all_sites_snowcovered[station] = pd.DataFrame({'mean':snowcovered_df.mean(),'std' :snowcovered_df.std()}).transpose()
        all_sites_shoulder[station] = pd.DataFrame({'mean':shoulder_df.mean(),'std' :shoulder_df.std()}).transpose()
        all_sites_all_year[station] = pd.DataFrame({'mean':all_year_df.mean(),'std' :all_year_df.std()}).transpose()
    
    pd.concat(all_sites_snowfree).to_csv(os.path.join(table_path,f'snowfree_stats.csv'), float_format = '%.2f')
    pd.concat(all_sites_snowcovered).to_csv(os.path.join(table_path,f'snowcovered_stats.csv'), float_format = '%.2f')
    pd.concat(all_sites_shoulder).to_csv(os.path.join(table_path,f'shoulder_stats.csv'), float_format = '%.2f')
    pd.concat(all_sites_all_year).to_csv(os.path.join(table_path,f'all_year_stats.csv'), float_format = '%.2f')
    return

def distribution_of_daily(level2,fig_path, data_path):
    #use data path for daily data
    data_path = os.path.join(data_path,'daily')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    ylims = {'ET': (-1,4),
            'Rnet': (-10,25),
            'VPD': (0,2)}
    fig,axs = plt.subplots(3,len(level2),
                            figsize=(7,7),
                            sharex=True, sharey='row')
    for col,key in enumerate(level2):
        station = key.split('_')[0]
        for row,var in enumerate(['ET','Rnet','VPD']):
            df = level2[key].loc[:,[var]].resample('D').mean()
            df['month'] = df.index.month
            if var in daily_coeffs:
                df[var] *= daily_coeffs[var]
            df.groupby(df.month).describe().to_csv(os.path.join(data_path,f'{station}_daily_{var.split("_")[0]}_per_month.csv'), float_format = '%.2f')
            df.boxplot(column=var, ax=axs[row,col], by='month', whis=(0,100), 
                        showmeans=True, meanline=False,
                        patch_artist=True,
                        boxprops= dict(facecolor=colorlist[station],color='k'),
                        whiskerprops=dict(color='k'),
                        medianprops = dict(color='k'),
                        meanprops = dict(marker='o', markeredgecolor='k',
                                        markerfacecolor='white',markersize=2.5)
                        )
            axs[row,col].set_xlabel('')
            axs[row,col].set_title('')
        axs[0,col].set_title(titlelist[station])
    #set ylabels at first row
    for row,var in enumerate(['ET','Rnet','VPD']):
        var_shrt = var.split('_')[0]
        axs[row,0].set_ylabel(f'{flux_long[var_shrt]}\n({ylabs_cum_daily[var_shrt]})')
    #align y labels
    fig.align_ylabels(axs[:,0])
    #set new x-ticks and x-ticklabels
    axs[0,0].set_xticks([1,3,5,7,9,11])
    [axs[2,col].set_xticklabels(['Jan','Mar','May','Jul','Sep','Nov'], rotation = 90) for col in range(len(level2))]
    fig.supxlabel('Month')
    fig.suptitle('')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'distribution_of_daily.pdf'))
    return

def distribution_of_daily_per_year(level2,fig_path):
    for key in level2:
        fig,axs = plt.subplots(3,1,
                            figsize=(7,6),
                            sharex=True, sharey=False)
        station = key.split('_')[0]
        for row,var in enumerate(['ET','Rnet','VPD']):
            daily_df = level2[key].loc[:,[var]].resample('D').mean()
            daily_df.dropna(how='all',inplace=True)
            if var in daily_coeffs:
                daily_df[var] *= daily_coeffs[var]
            sns.boxplot(x=daily_df.index.month,
                        y=daily_df[var].values,
                        hue=daily_df.index.year, 
                        ax=axs[row],whis=20,
                        palette='colorblind',
                        showmeans=True,
                        meanprops = dict(marker='o', markeredgecolor='black',
                                        markerfacecolor='white',markersize=2.5)
                        )
            axs[row].set_xlabel('')
            axs[row].set_title('')
            var_shrt = var.split('_')[0]
            axs[row].set_ylabel(f'{flux_long[var_shrt]}\n({ylabs_cum_daily[var_shrt]})')
        #set new x-ticks and x-ticklabels
        [axs[row].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11]) for row in range(3)]
        axs[2].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                                'Jul','Aug','Sep','Oct','Nov','Dec'], rotation = 90)
        [axs[row].grid() for row in range(3)]
        #get unique legend entries
        all_handles = []
        all_labels = []
        for ax in axs.flatten():
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)
            ax.get_legend().remove()

        unique = [(h, l) for i, (h, l) in enumerate(zip(all_handles, all_labels)) if l not in all_labels[:i]]
        fig.subplots_adjust(bottom=0.12)
        fig.legend(*zip(*unique),loc='lower center',ncol=len(unique))
        fig.align_ylabels(axs)
        fig.suptitle(f'{titlelist[station]}')
        fig.savefig(os.path.join(fig_path,f'distribution_of_daily_{key}.pdf'))
    plt.close('all')
    return

#plot monthly evaporation and evaporation as ratio of precipitation
def monthly_E_frac(level2,fig_path,data_path):
    data_path = os.path.join(data_path,'monthly')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    #get monthly precipitation from nearest MET-station (frost-data)
    frost = read.frost_monthly() 

    ncols= len(level2)
    fig,axs = plt.subplots(2,ncols,
                        figsize=(7,5),
                        sharex=True,
                        sharey='row')

    axs[0,0].set_xlim(-1,12)

    #errorbar properties
    error_kw=dict(lw=0.5, capsize=1, capthick=0.5)

    #create twin axes for first row
    twin_axs = []
    for ax in axs[0,:]:
        twin_axs.append(ax.twinx())

    monthly_ratios = {}
    warm_season_ratios = {}
    annual_ratios = {}

    for col,station in enumerate(level2):
        P = frost[station]['precipitation'] #OBS MET index is label on left side of aggregation interval
        df=level2[station].loc[:,['ET']]
        df=df.resample('M').sum()
        #reset index to get index at 1st of month, to match MET-data
        df.set_index(pd.to_datetime(df.index.strftime('%Y-%m')), inplace=True)
        df['P'] = P
        df['E_frac'] = 100*df['ET']/df['P']
        if station == 'adventdalen':
            df = df.loc[df.index.year != 2014]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        #extract annual and warm season evaporation ratios
        years = []
        annual_list = []
        warm_season_list = []
        for year, yearly_data in df.groupby(df.index.year):
            years.append(year)
            annual_list.append(yearly_data.ET.sum()/yearly_data.P.sum())
            warm_season = (yearly_data.index.month>4) & (yearly_data.index.month<10)
            warm_season_data = yearly_data.loc[warm_season,:]
            warm_season_list.append(warm_season_data.ET.sum()/warm_season_data.P.sum())

        annual_ratios[station] = pd.Series(data=annual_list,index=years)
        warm_season_ratios[station] = pd.Series(data=warm_season_list,index=years)

        monthly_ratios[station]= df
        df['month'] = df.index.month
        df.groupby(df.month).describe().loc[1:12,(['ET','P','E_frac'],['mean','min','max'])].to_csv(os.path.join(data_path,
                                                                                    f'{station}_summary.csv'), 
                                                                                    float_format = '%.2f')
        df.rename({'ET':'Evaporation',
                    'P':'Precipitation',
                    'E_frac':'Evaporation ratio',},axis=1,inplace=True)

        #extract mean, 
        mean_val = df.groupby(df.index.month).mean()

        #extract min-max relative to mean for error bars showing range of monthly values
        E_err = [df['Evaporation'].groupby(df.index.month).mean()-df['Evaporation'].groupby(df.index.month).min(),
                df['Evaporation'].groupby(df.index.month).max()-df['Evaporation'].groupby(df.index.month).mean()]
        P_err = [df['Precipitation'].groupby(df.index.month).mean()-df['Precipitation'].groupby(df.index.month).min(),
                df['Precipitation'].groupby(df.index.month).max()-df['Precipitation'].groupby(df.index.month).mean()]
        frac_err = [df['Evaporation ratio'].groupby(df.index.month).mean()-df['Evaporation ratio'].groupby(df.index.month).min(),
                df['Evaporation ratio'].groupby(df.index.month).max()-df['Evaporation ratio'].groupby(df.index.month).mean()]

        bar_width = 0.3
        mean_val.plot(kind = 'bar',
                        ax=axs[0,col],
                        x = 'month',
                        y = 'Evaporation',
                        legend=False,
                        yerr=E_err,
                        color='coral',
                        width=bar_width,
                        position=0,
                        error_kw = error_kw
                        )

        mean_val.plot(kind = 'bar',
                        ax=twin_axs[col],
                        x = 'month',
                        y = 'Precipitation',
                        legend=False,
                        yerr=P_err,
                        secondary_y=True,
                        facecolor='skyblue',
                        width=bar_width,
                        position=1,
                        error_kw = error_kw
                        )
        mean_val.plot(kind = 'bar',
                        ax=axs[1,col],
                        x = 'month',
                        y = 'Evaporation ratio',
                        legend=False,
                        yerr=frac_err,
                        color=colorlist[station],
                        error_kw = error_kw)

        axs[0,col].set_title(titlelist[station])
        [ax.set_xlabel('') for ax in axs[:,col]]

    #set major xticks 
    axs[0,0].set_xticks([0,2,4,6,8,10], labels=['Jan','Mar','May','Jul','Sep','Nov']) 
    axs[0,0].tick_params(labelrotation=90)

    #set major yticks
    axs[0,0].set_yticks([0,25,50,75,100])
    [ax.right_ax.set_yticks([0,50,100,150,200]) for ax in twin_axs]
    [ax.grid(linewidth=0.5) for ax in axs.flatten()]
    
    #remove minor ticks
    axs[0,0].minorticks_off()
    [ax.minorticks_off() for ax in twin_axs]

    #set ylims and y labels
    axs[0,0].set_ylim(-10,100)
    axs[0,0].set_ylabel('Evaporation (mm/month)')
    axs[1,0].set_ylabel('Evaporation ratio (%)')
    fig.align_ylabels(axs[:,0])

    #set ylim for secondary axes in first row, remove yticklabels for all except last
    [ax.right_ax.set_ylim(-20,200) for ax in twin_axs]
    [ax.right_ax.set_yticklabels([]) for ax in twin_axs[0:-1]]
    twin_axs[-1].right_ax.set_ylabel('Precipitaion (mm/month)')

    #Get legend entries
    all_handles = []
    all_labels = []

    for ax in twin_axs:
        handles, labels = ax.right_ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    for ax in axs[0,:]:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    unique = [(h, l) for i, (h, l) in enumerate(zip(all_handles, all_labels)) if l not in all_labels[:i]]
    
    fig.subplots_adjust(top=0.85)
    fig.suptitle('')
    fig.supxlabel('Month')
    fig.legend(*zip(*unique),loc='upper center',ncol=3)
    fig.savefig(os.path.join(fig_path,'monthly_E_frac.pdf'))

    #save monthly, warm season and annual evaporation ratios
    monthly_ratios = pd.concat(monthly_ratios, axis=1)
    warm_season_ratios = pd.concat(warm_season_ratios, axis=1)
    annual_ratios = pd.concat(annual_ratios, axis=1)
    monthly_ratios.to_csv(os.path.join(data_path,'monthly_evaporation_ratios.csv'), 
                        float_format = '%.2f')
    warm_season_ratios.to_csv(os.path.join(data_path,'warm_season_evaporation_ratios.csv'), 
                        float_format = '%.2f')
    annual_ratios.to_csv(os.path.join(data_path,'annual_evaporation_ratios.csv'), 
                        float_format = '%.2f')
    return

def compare_to_fluxnet(level2,fig_path,data_path):

    fig_path = os.path.join(fig_path,'fluxnet_comparison')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    data_path = os.path.join(data_path,'fluxnet_comparison')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    #read fluxnet metadata and make PFT_mapper
    fluxnet_metadata = read.fluxnet_metadata()
    pft_mapper = fluxnet_metadata.PFT_group.to_dict()

    #get fluxnet symbology and create mappers for each PFT
    with open(os.path.join('..','metadata','fluxnet_symbology.json')) as file:
        fluxnet_symbology = json.load(file)

    fluxnet_color_mapper = {key: fluxnet_symbology[key]['color'] for key in fluxnet_symbology}
    fluxnet_marker_mapper = {key: fluxnet_symbology[key]['marker'] for key in fluxnet_symbology}

    #read data from fluxnet
    fluxnet_monthly = read.fluxnet(timeres='MM')

    #select only ET, TA and P
    fluxnet_monthly = fluxnet_monthly.loc[:,(slice(None),['ET','TA_F','P_F','VPD_F','NETRAD'])]

    #aggregate to yearly May-Sept sum (for E and P)/mean (for TA) and yearly sum/mean
    #aggregate one station at a time to not end up with 0 as sum of NAs
    annual_data = []
    mean_data = []
    for station,data in fluxnet_monthly.groupby(level=0,axis=1):
        data = data.dropna(how='any',axis=0)
        criteria = (data.index.month>=5) & (data.index.month<=9)
        #aggregate E and P as sum
        may_sept_E_P = data.loc[criteria,(slice(None),['ET','P_F'])].resample('Y').sum()
        may_sept_E_P.rename({'ET':'ET_may_sept','P_F':'P_may_sept'},axis=1,level=1,inplace=True)
        annual_E_P = data.loc[:,(slice(None),['ET','P_F'])].resample('Y').sum()
        annual_E_P.rename({'ET':'ET_annual','P_F':'P_annual'},axis=1,level=1,inplace=True)

        #aggregate TA and VPD as mean
        may_sept_TA = data.loc[criteria,(slice(None),['TA_F','VPD_F','NETRAD'])].resample('Y').mean()
        may_sept_TA.rename({'TA_F':'TA_may_sept','VPD_F':'VPD_may_sept','NETRAD':'Rnet_may_sept'},axis=1,level=1,inplace=True)
        annual_TA = data.loc[:,(slice(None),['TA_F','VPD_F','NETRAD'])].resample('Y').mean()
        annual_TA.rename({'TA_F':'TA_annual','VPD_F':'VPD_annual','NETRAD':'Rnet_annual'},axis=1,level=1,inplace=True)

        #aggregate November-March precip as sum
        winter_criteria = (data.index.month>=11) | (data.index.month<=3)
        nov_mar_P = data.loc[winter_criteria,(slice(None),'P_F')].resample('Y').sum()
        nov_mar_P.rename({'P_F':'P_nov_mar'},axis=1,level=1,inplace=True)
        
        #concat data for station and add to list of all stations
        station_data = pd.concat([may_sept_E_P,annual_E_P,may_sept_TA,annual_TA,nov_mar_P],axis=1)
        annual_data.append(station_data)
        mean_data.append(station_data.describe().loc[['count','mean','min','max'],:])

    #join station to one dataframe
    annual_data = pd.concat(annual_data,axis=1)
    mean_data = pd.concat(mean_data,axis=1)
    annual_data.to_csv(os.path.join(data_path,'fluxnet_annual_data.csv'), float_format='%.1f')
    mean_data.to_csv(os.path.join(data_path,'fluxnet_mean_annual_data.csv'), float_format='%.1f')

    #extract same stats for LATICE(+)-stations
    level2_annual_data = {}
    level2_mean_data = {}

    #get met precip data
    frost = read.frost_monthly()

    #annual level2 data
    for station in level2:
        data = level2[station]
        met_data = frost[station]
    
        #quickfix until P_1_1_1 is included for adventdalen
        if (station == 'adventdalen') and ('P_1_1_1' not in data.columns):
            data['P_1_1_1'] = pd.Series(np.nan, index=data.index)
    
        criteria = (data.index.month>=5) & (data.index.month<=9)

        #aggregate E and P as sum
        may_sept_E_P = data.loc[criteria,['ET','P_1_1_1']].resample('Y').sum()
        may_sept_E_P.rename({'ET':'ET_may_sept','P_1_1_1':'P_may_sept'},axis=1,inplace=True)
        may_sept_met_P = met_data.loc[(met_data.index.month>=5) & (met_data.index.month<=9),['precipitation']].resample('Y').sum()
        may_sept_met_P.rename({'precipitation':'P_met_may_sept'},axis=1,inplace=True)
        annual_E_P = data.loc[:,['ET','P_1_1_1']].resample('Y').sum()
        annual_E_P.rename({'ET':'ET_annual','P_1_1_1':'P_annual'},axis=1,inplace=True)
        annual_met_P = met_data.loc[:,['precipitation']].resample('Y').sum()
        annual_met_P.rename({'precipitation':'P_met_annual'},axis=1,inplace=True)
    
        #aggregate TA as mean
        may_sept_TA = data.loc[criteria,['TA','VPD','Rnet']].resample('Y').mean()
        may_sept_TA.rename({'TA':'TA_may_sept','VPD':'VPD_may_sept','Rnet':'Rnet_may_sept'},axis=1,inplace=True)
        annual_TA = data.loc[:,['TA','VPD','Rnet']].resample('Y').mean()
        annual_TA.rename({'TA':'TA_annual','VPD':'VPD_annual','Rnet':'Rnet_annual'},axis=1,inplace=True)
    
        #aggregate November-March precip as sum
        winter_criteria = (data.index.month>=11) | (data.index.month<=3)
        nov_mar_P = data.loc[winter_criteria,['P_1_1_1']].resample('Y').sum()
        nov_mar_P.rename({'P_1_1_1':'P_nov_mar'},axis=1,inplace=True)
        nov_mar_met_P = met_data.loc[(met_data.index.month>=11) | (met_data.index.month<=3),['precipitation']].resample('Y').sum()
        nov_mar_met_P.rename({'precipitation':'P_met_nov_mar'},axis=1,inplace=True)
    
        #concat station data
        station_data = pd.concat([may_sept_E_P,annual_E_P,may_sept_TA,annual_TA,nov_mar_P,nov_mar_met_P,annual_met_P,may_sept_met_P],axis=1)
        if station == 'adventdalen':
            station_data = station_data.loc[station_data.index.year != 2014]
    
        #save in dicts
        level2_annual_data[station] = station_data
        level2_mean_data[station] = station_data.describe().loc[['count','mean','min','max'],:]

    level2_annual_data = pd.concat(level2_annual_data,axis=1)
    level2_mean_data = pd.concat(level2_mean_data,axis=1)
    level2_annual_data.to_csv(os.path.join(data_path,'latice_annual_data.csv'), float_format='%.1f')
    level2_mean_data.to_csv(os.path.join(data_path,'latice_mean_annual_data.csv'), float_format='%.1f')

    #plot annual TA ET with precip as colorplot
    fig,ax = plt.subplots(figsize=(4,3))
    ax.errorbar(x = mean_data.loc['mean',(slice(None),'TA_may_sept')],
                y = mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [mean_data.loc['mean',(slice(None),'TA_may_sept')]-mean_data.loc['min',(slice(None),'TA_may_sept')],
                        mean_data.loc['max',(slice(None),'TA_may_sept')]-mean_data.loc['mean',(slice(None),'TA_may_sept')]],
                yerr = [mean_data.loc['mean',(slice(None),'ET_annual')]-mean_data.loc['min',(slice(None),'ET_annual')], 
                        mean_data.loc['max',(slice(None),'ET_annual')]-mean_data.loc['mean',(slice(None),'ET_annual')]],
                # fmt='o',
                ecolor=mean_data.loc['mean',(slice(None),'ET_annual')].index.levels[0].map(pft_mapper).map(fluxnet_color_mapper),
                zorder=0,
                linestyle='None')
    cm = ax.scatter(x = mean_data.loc['mean',(slice(None),'TA_may_sept')],
                y = mean_data.loc['mean',(slice(None),'ET_annual')],
                c = mean_data.loc['mean',(slice(None),'P_nov_mar')],
                cmap='Blues',
                vmin=0,
                vmax=500,
                zorder=10,
                linestyle='None',
                edgecolors = mean_data.loc['mean',(slice(None),'ET_annual')].index.levels[0].map(pft_mapper).map(fluxnet_color_mapper))
    
    #plot latice data
    ax.errorbar(x = level2_mean_data.loc['mean',(slice(None),'TA_may_sept')],
                y = level2_mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [level2_mean_data.loc['mean',(slice(None),'TA_may_sept')]-level2_mean_data.loc['min',(slice(None),'TA_may_sept')],
                        level2_mean_data.loc['max',(slice(None),'TA_may_sept')]-level2_mean_data.loc['mean',(slice(None),'TA_may_sept')]],
                yerr = [level2_mean_data.loc['mean',(slice(None),'ET_annual')]-level2_mean_data.loc['min',(slice(None),'ET_annual')], 
                        level2_mean_data.loc['max',(slice(None),'ET_annual')]-level2_mean_data.loc['mean',(slice(None),'ET_annual')]],
                # fmt='o',
                ecolor=[colorlist[s] for s in level2_mean_data.columns.levels[0]],
                zorder=0,
                linestyle='None')
    cm = ax.scatter(x = level2_mean_data.loc['mean',(slice(None),'TA_may_sept')],
                y = level2_mean_data.loc['mean',(slice(None),'ET_annual')],
                c = level2_mean_data.loc['mean',(slice(None),'P_met_nov_mar')],
                cmap='Blues',
                vmin=0,
                vmax=500,
                zorder=10,
                linestyle='None',
                edgecolors = [colorlist[s] for s in level2_mean_data.columns.levels[0]])
    ax.set_xlabel('Mean warm season temperature (\u2103)')
    ax.set_ylabel('Annual evaporation (mm/year)')
    fig.colorbar(cm,label='Winter precipitation')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'summerTA_annE_winterP.pdf'))

    #plot annual TA and may-sept TA vs annual ET
    fig,ax = plt.subplots(1,2,figsize=(7,4), sharey=True)

    #fit linear model
    x_may_sept = np.concatenate([mean_data.loc['mean',(slice(None),'TA_may_sept')].values,
                        level2_mean_data.loc['mean',(slice(None),'TA_may_sept')].values])
    xx_may_sept = sm.add_constant(x_may_sept)
    x_annual = np.concatenate([mean_data.loc['mean',(slice(None),'TA_annual')].values,
                        level2_mean_data.loc['mean',(slice(None),'TA_annual')].values])
    xx_annual = sm.add_constant(x_annual)
    yy_annual = np.concatenate([mean_data.loc['mean',(slice(None),'ET_annual')].values,
                        level2_mean_data.loc['mean',(slice(None),'ET_annual')].values])
    reg_fit0 = sm.OLS(yy_annual, xx_annual).fit()
    reg_fit1 = sm.OLS(yy_annual, xx_may_sept).fit()

    with open(os.path.join(data_path,'regression_summary_annTA_annE.txt'),'w') as file:
        file.write(reg_fit0.summary().as_text())

    with open(os.path.join(data_path,'regression_summary_maySeptTA_annE.txt'),'w') as file:
        file.write(reg_fit1.summary().as_text())

    #plot trend line
    x0 = np.linspace(x_annual.min(),x_annual.max())
    x1 = np.linspace(x_may_sept.min(),x_may_sept.max())
    b0_0,b1_0 = reg_fit0.params
    b0_1,b1_1 = reg_fit1.params
    y0 = b0_0+b1_0*x0
    y1 = b0_1+b1_1*x1
    ax[0].plot(x0,y0,color='k',linestyle='dashed')
    ax[0].text(x0.max(), 400, s=f'y = {b0_0:.1f}+{b1_0:.1f}x', horizontalalignment='right')
    ax[1].plot(x1,y1,color='k',linestyle='dashed')
    ax[1].text(x1.max(), 400, s=f'y = {b0_1:.1f}+{b1_1:.1f}x', horizontalalignment='right')

    ax[0].errorbar(x = mean_data.loc['mean',(slice(None),'TA_annual')],
                y = mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [mean_data.loc['mean',(slice(None),'TA_annual')]-mean_data.loc['min',(slice(None),'TA_annual')],
                        mean_data.loc['max',(slice(None),'TA_annual')]-mean_data.loc['mean',(slice(None),'TA_annual')]],
                yerr = [mean_data.loc['mean',(slice(None),'ET_annual')]-mean_data.loc['min',(slice(None),'ET_annual')], 
                        mean_data.loc['max',(slice(None),'ET_annual')]-mean_data.loc['mean',(slice(None),'ET_annual')]],
                ecolor=mean_data.loc['mean',(slice(None),'ET_annual')].index.levels[0].map(pft_mapper).map(fluxnet_color_mapper),
                zorder=0,
                linestyle='None')

    ax[0].errorbar(x = level2_mean_data.loc['mean',(slice(None),'TA_annual')],
                y = level2_mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [level2_mean_data.loc['mean',(slice(None),'TA_annual')]-level2_mean_data.loc['min',(slice(None),'TA_annual')],
                        level2_mean_data.loc['max',(slice(None),'TA_annual')]-level2_mean_data.loc['mean',(slice(None),'TA_annual')]],
                yerr = [level2_mean_data.loc['mean',(slice(None),'ET_annual')]-level2_mean_data.loc['min',(slice(None),'ET_annual')], 
                        level2_mean_data.loc['max',(slice(None),'ET_annual')]-level2_mean_data.loc['mean',(slice(None),'ET_annual')]],
                ecolor=[colorlist[s] for s in level2_mean_data.columns.levels[0]],
                zorder=0,
                linestyle='None')

    ax[1].errorbar(x = mean_data.loc['mean',(slice(None),'TA_may_sept')],
                y = mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [mean_data.loc['mean',(slice(None),'TA_may_sept')]-mean_data.loc['min',(slice(None),'TA_may_sept')],
                        mean_data.loc['max',(slice(None),'TA_may_sept')]-mean_data.loc['mean',(slice(None),'TA_may_sept')]],
                yerr = [mean_data.loc['mean',(slice(None),'ET_annual')]-mean_data.loc['min',(slice(None),'ET_annual')], 
                        mean_data.loc['max',(slice(None),'ET_annual')]-mean_data.loc['mean',(slice(None),'ET_annual')]],
                zorder=0,
                linestyle='None',
                ecolor=mean_data.loc['mean',(slice(None),'ET_annual')].index.levels[0].map(pft_mapper).map(fluxnet_color_mapper))

    ax[1].errorbar(x = level2_mean_data.loc['mean',(slice(None),'TA_may_sept')],
                y = level2_mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [level2_mean_data.loc['mean',(slice(None),'TA_may_sept')]-level2_mean_data.loc['min',(slice(None),'TA_may_sept')],
                        level2_mean_data.loc['max',(slice(None),'TA_may_sept')]-level2_mean_data.loc['mean',(slice(None),'TA_may_sept')]],
                yerr = [level2_mean_data.loc['mean',(slice(None),'ET_annual')]-level2_mean_data.loc['min',(slice(None),'ET_annual')], 
                        level2_mean_data.loc['max',(slice(None),'ET_annual')]-level2_mean_data.loc['mean',(slice(None),'ET_annual')]],
                ecolor=[colorlist[s] for s in level2_mean_data.columns.levels[0]],
                zorder=10,
                linestyle='None')
    ax[0].set_xlabel('Mean annual temperature (\u2103)')
    ax[1].set_xlabel('Mean warm season temperature (\u2103)')
    fig.supylabel('Annual evaporation (mm/year)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'annTA_maySeptTA_annE.pdf'))

    #plot only annual TA annual E
    fig,ax = plt.subplots(figsize=(4,3), sharey=True)
    #plot trend line
    ax.plot(x0,y0,color='k',linestyle='dashed')
    ax.text(x0.max(), 400, s=f'y = {b0_0:.1f}+{b1_0:.1f}x', horizontalalignment='right')


    ax.errorbar(x = mean_data.loc['mean',(slice(None),'TA_annual')],
                y = mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [mean_data.loc['mean',(slice(None),'TA_annual')]-mean_data.loc['min',(slice(None),'TA_annual')],
                        mean_data.loc['max',(slice(None),'TA_annual')]-mean_data.loc['mean',(slice(None),'TA_annual')]],
                yerr = [mean_data.loc['mean',(slice(None),'ET_annual')]-mean_data.loc['min',(slice(None),'ET_annual')], 
                        mean_data.loc['max',(slice(None),'ET_annual')]-mean_data.loc['mean',(slice(None),'ET_annual')]],
                ecolor=mean_data.loc['mean',(slice(None),'ET_annual')].index.levels[0].map(pft_mapper).map(fluxnet_color_mapper),
                zorder=0,
                linestyle='None')

    ax.errorbar(x = level2_mean_data.loc['mean',(slice(None),'TA_annual')],
                y = level2_mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [level2_mean_data.loc['mean',(slice(None),'TA_annual')]-level2_mean_data.loc['min',(slice(None),'TA_annual')],
                        level2_mean_data.loc['max',(slice(None),'TA_annual')]-level2_mean_data.loc['mean',(slice(None),'TA_annual')]],
                yerr = [level2_mean_data.loc['mean',(slice(None),'ET_annual')]-level2_mean_data.loc['min',(slice(None),'ET_annual')], 
                        level2_mean_data.loc['max',(slice(None),'ET_annual')]-level2_mean_data.loc['mean',(slice(None),'ET_annual')]],
                ecolor=[colorlist[s] for s in level2_mean_data.columns.levels[0]],
                zorder=0,
                linestyle='None')

    ax.set_xlabel('Mean annual temperature (\u2103)')
    ax.set_ylabel('Annual evaporation (mm/year)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'annTA_annE.pdf'))

    #plot may-sept TA and maysept ET
    fig,ax = plt.subplots(figsize=(4,3))
    ax.errorbar(x = mean_data.loc['mean',(slice(None),'TA_may_sept')],
                y = mean_data.loc['mean',(slice(None),'ET_may_sept')],
                xerr = [mean_data.loc['mean',(slice(None),'TA_may_sept')]-mean_data.loc['min',(slice(None),'TA_may_sept')],
                        mean_data.loc['max',(slice(None),'TA_may_sept')]-mean_data.loc['mean',(slice(None),'TA_may_sept')]],
                yerr = [mean_data.loc['mean',(slice(None),'ET_may_sept')]-mean_data.loc['min',(slice(None),'ET_may_sept')], 
                        mean_data.loc['max',(slice(None),'ET_may_sept')]-mean_data.loc['mean',(slice(None),'ET_may_sept')]],
                zorder=0,
                linestyle='None',
                ecolor=mean_data.loc['mean',(slice(None),'TA_may_sept')].index.levels[0].map(pft_mapper).map(fluxnet_color_mapper))

    ax.errorbar(x = level2_mean_data.loc['mean',(slice(None),'TA_may_sept')],
                y = level2_mean_data.loc['mean',(slice(None),'ET_may_sept')],
                xerr = [level2_mean_data.loc['mean',(slice(None),'TA_may_sept')]-level2_mean_data.loc['min',(slice(None),'TA_may_sept')],
                        level2_mean_data.loc['max',(slice(None),'TA_may_sept')]-level2_mean_data.loc['mean',(slice(None),'TA_may_sept')]],
                yerr = [level2_mean_data.loc['mean',(slice(None),'ET_may_sept')]-level2_mean_data.loc['min',(slice(None),'ET_may_sept')], 
                        level2_mean_data.loc['max',(slice(None),'ET_may_sept')]-level2_mean_data.loc['mean',(slice(None),'ET_may_sept')]],
                ecolor=[colorlist[s] for s in level2_mean_data.columns.levels[0]],
                zorder=10,
                linestyle='None')
    ax.set_xlabel('Mean warm season temperature (\u2103)')
    ax.set_ylabel('Warm season evaporation (mm/year)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'maySeptTA_maySeptE.pdf'))

    #plot may-sept TA and annual ET
    fig,ax = plt.subplots(figsize=(7,4))
    ax.errorbar(x = mean_data.loc['mean',(slice(None),'TA_may_sept')],
                y = mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [mean_data.loc['mean',(slice(None),'TA_may_sept')]-mean_data.loc['min',(slice(None),'TA_may_sept')],
                        mean_data.loc['max',(slice(None),'TA_may_sept')]-mean_data.loc['mean',(slice(None),'TA_may_sept')]],
                yerr = [mean_data.loc['mean',(slice(None),'ET_annual')]-mean_data.loc['min',(slice(None),'ET_annual')], 
                        mean_data.loc['max',(slice(None),'ET_annual')]-mean_data.loc['mean',(slice(None),'ET_annual')]],
                zorder=0,
                linestyle='None',
                ecolor=mean_data.loc['mean',(slice(None),'TA_may_sept')].index.levels[0].map(pft_mapper).map(fluxnet_color_mapper))

    ax.errorbar(x = level2_mean_data.loc['mean',(slice(None),'TA_may_sept')],
                y = level2_mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [level2_mean_data.loc['mean',(slice(None),'TA_may_sept')]-level2_mean_data.loc['min',(slice(None),'TA_may_sept')],
                        level2_mean_data.loc['max',(slice(None),'TA_may_sept')]-level2_mean_data.loc['mean',(slice(None),'TA_may_sept')]],
                yerr = [level2_mean_data.loc['mean',(slice(None),'ET_annual')]-level2_mean_data.loc['min',(slice(None),'ET_annual')], 
                        level2_mean_data.loc['max',(slice(None),'ET_annual')]-level2_mean_data.loc['mean',(slice(None),'ET_annual')]],
                ecolor=[colorlist[s] for s in level2_mean_data.columns.levels[0]],
                zorder=10,
                linestyle='None')

    #plot trend line
    ax.plot(x1,y1,color='k',linestyle='dashed')
    ax.text(x1.max(), 400, s=f'y = {b0_1:.1f}+{b1_1:.1f}x', horizontalalignment='right')

    ax.set_xlabel('Mean warm season temperature (\u2103)')
    ax.set_ylabel('Annual evaporation (mm/year)')
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'maySeptTA_annualE.pdf'))

    #plot annual P and annual ET
    fig,ax = plt.subplots(figsize=(4,3))
    ax.errorbar(x = mean_data.loc['mean',(slice(None),'P_annual')],
                y = mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [mean_data.loc['mean',(slice(None),'P_annual')]-mean_data.loc['min',(slice(None),'P_annual')],
                        mean_data.loc['max',(slice(None),'P_annual')]-mean_data.loc['mean',(slice(None),'P_annual')]],
                yerr = [mean_data.loc['mean',(slice(None),'ET_annual')]-mean_data.loc['min',(slice(None),'ET_annual')], 
                        mean_data.loc['max',(slice(None),'ET_annual')]-mean_data.loc['mean',(slice(None),'ET_annual')]],
                zorder=0,
                linestyle='None',
                ecolor=mean_data.loc['mean',(slice(None),'ET_annual')].index.levels[0].map(pft_mapper).map(fluxnet_color_mapper))

    ax.errorbar(x = level2_mean_data.loc['mean',(slice(None),'P_met_annual')],
                y = level2_mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [level2_mean_data.loc['mean',(slice(None),'P_met_annual')]-level2_mean_data.loc['min',(slice(None),'P_met_annual')],
                        level2_mean_data.loc['max',(slice(None),'P_met_annual')]-level2_mean_data.loc['mean',(slice(None),'P_met_annual')]],
                yerr = [level2_mean_data.loc['mean',(slice(None),'ET_annual')]-level2_mean_data.loc['min',(slice(None),'ET_annual')], 
                        level2_mean_data.loc['max',(slice(None),'ET_annual')]-level2_mean_data.loc['mean',(slice(None),'ET_annual')]],
                ecolor=[colorlist[s] for s in level2_mean_data.columns.levels[0]],
                zorder=10,
                linestyle='None')
    ax.set_xlabel('Annual precipitation (mm)')
    ax.set_ylabel('Annual evaporation (mm)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'annP_annE.pdf'))

    #plot nov-mar P and may-sept ET
    fig,ax = plt.subplots(figsize=(4,3))
    ax.errorbar(x = mean_data.loc['mean',(slice(None),'P_nov_mar')],
                y = mean_data.loc['mean',(slice(None),'ET_may_sept')],
                xerr = [mean_data.loc['mean',(slice(None),'P_nov_mar')]-mean_data.loc['min',(slice(None),'P_nov_mar')],
                        mean_data.loc['max',(slice(None),'P_nov_mar')]-mean_data.loc['mean',(slice(None),'P_nov_mar')]],
                yerr = [mean_data.loc['mean',(slice(None),'ET_may_sept')]-mean_data.loc['min',(slice(None),'ET_may_sept')], 
                        mean_data.loc['max',(slice(None),'ET_may_sept')]-mean_data.loc['mean',(slice(None),'ET_may_sept')]],
                zorder=0,
                linestyle='None',
                ecolor=mean_data.loc['mean',(slice(None),'ET_may_sept')].index.levels[0].map(pft_mapper).map(fluxnet_color_mapper))

    ax.errorbar(x = level2_mean_data.loc['mean',(slice(None),'P_met_nov_mar')],
                y = level2_mean_data.loc['mean',(slice(None),'ET_may_sept')],
                xerr = [level2_mean_data.loc['mean',(slice(None),'P_met_nov_mar')]-level2_mean_data.loc['min',(slice(None),'P_met_nov_mar')],
                        level2_mean_data.loc['max',(slice(None),'P_met_nov_mar')]-level2_mean_data.loc['mean',(slice(None),'P_met_nov_mar')]],
                yerr = [level2_mean_data.loc['mean',(slice(None),'ET_may_sept')]-level2_mean_data.loc['min',(slice(None),'ET_may_sept')], 
                        level2_mean_data.loc['max',(slice(None),'ET_may_sept')]-level2_mean_data.loc['mean',(slice(None),'ET_may_sept')]],
                ecolor=[colorlist[s] for s in level2_mean_data.columns.levels[0]],
                zorder=10,
                linestyle='None')
    ax.set_xlabel('Winter precipitation (mm)')
    ax.set_ylabel('Warm season evaporation (mm)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'novMarP_maySeptE.pdf'))

    #plot nov-mar P and annual ET 
    fig,ax = plt.subplots(figsize=(4,3))
    ax.errorbar(x = mean_data.loc['mean',(slice(None),'P_nov_mar')],
                y = mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [mean_data.loc['mean',(slice(None),'P_nov_mar')]-mean_data.loc['min',(slice(None),'P_nov_mar')],
                        mean_data.loc['max',(slice(None),'P_nov_mar')]-mean_data.loc['mean',(slice(None),'P_nov_mar')]],
                yerr = [mean_data.loc['mean',(slice(None),'ET_annual')]-mean_data.loc['min',(slice(None),'ET_annual')], 
                        mean_data.loc['max',(slice(None),'ET_annual')]-mean_data.loc['mean',(slice(None),'ET_annual')]],
                zorder=0,
                linestyle='None',
                ecolor=mean_data.loc['mean',(slice(None),'ET_annual')].index.levels[0].map(pft_mapper).map(fluxnet_color_mapper))

    ax.errorbar(x = level2_mean_data.loc['mean',(slice(None),'P_met_nov_mar')],
                y = level2_mean_data.loc['mean',(slice(None),'ET_annual')],
                xerr = [level2_mean_data.loc['mean',(slice(None),'P_met_nov_mar')]-level2_mean_data.loc['min',(slice(None),'P_met_nov_mar')],
                        level2_mean_data.loc['max',(slice(None),'P_met_nov_mar')]-level2_mean_data.loc['mean',(slice(None),'P_met_nov_mar')]],
                yerr = [level2_mean_data.loc['mean',(slice(None),'ET_annual')]-level2_mean_data.loc['min',(slice(None),'ET_annual')], 
                        level2_mean_data.loc['max',(slice(None),'ET_annual')]-level2_mean_data.loc['mean',(slice(None),'ET_annual')]],
                ecolor=[colorlist[s] for s in level2_mean_data.columns.levels[0]],
                zorder=10,
                linestyle='None')

    ax.set_xlabel('Winter precipitation (mm)')
    ax.set_ylabel('Annual evaporation (mm)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'NovMarP_annE.pdf'))
    plt.close('all')
    return