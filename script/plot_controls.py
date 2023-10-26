import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
import os, json
import statsmodels.api as sm
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from factor_analysis import factor_analysis

daytime_start = '07:00'
daytime_end = '20:00'

#in UTC
midday_start = {
                'finseflux': '12:00', 
                'iskoras': '12:00',
                'myr2': '12:00',
                'adventdalen': '12:00',
                }
midday_end = {
                'finseflux': '16:00',
                'iskoras': '16:00',
                'myr2': '16:00',
                'adventdalen': '16:00',
                }

titlelist = {'iskoras': 'I\u0161koras',
            'finseflux': 'Finse',
            'myr1': 'Hisåsen upper',
            'finsefetene': 'Finse fen',
            'adventdalen': 'Adventdalen',
            'myr2': 'Hisåsen'}

#max and min limits for bins
bin_limits = {
                'VPD': (0.1,2.5),
                'G_a': (0,0.1),
                'temp_grad': (-20,10),
                'H_LE': (-100,500),
                'Enet': (-100,550),
                '(z-d)/L': (-0.5,0.5),
                }

plt.rc('axes', labelsize=9)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=9)   # fontsize of the tick labels
plt.rc('ytick', labelsize=9)    # fontsize of the tick labels
plt.rc('legend', fontsize=9)    # legend fontsize
plt.rc('axes', titlesize=9)     # fontsize of the axes title
plt.rc('figure', titlesize=9)  # fontsize of the figure title
plt.rc('figure', labelsize=9)  # fontsize of the figure title

colorlist = {'iskoras': 'tab:orange',
            'finseflux': 'tab:blue',
            'myr1': 'olive',
            'finsefetene': 'tab:blue',
            'myr2': 'tab:green',
            'adventdalen': 'tab:red'}

yearly_linestyle = {'2012': 'dotted',
                    '2013': 'solid',
                    '2015': 'dashed',
                    '2016': 'dashdot',
                    '2019': 'dotted',
                    '2020': 'solid',
                    '2021': 'dashed'
                    }

#plot observed and penman-monteith evaporation for bins of VPD and net energy
def VPD_Enet_ET_all_PM(projects,fig_path, PM_estimate='ET_PM_Enet'):
    bins1 = [-0.1,0,0.1,0.2,0.3,0.4,0.5]
    cmap1 = mpl.cm.plasma
    norm1 = mpl.colors.BoundaryNorm(bins1, cmap1.N)
    cmap2 = mpl.cm.coolwarm
    norm2 = mpl.colors.SymLogNorm(linthresh=10,vmin=-200,vmax=200)
    fig,axs = plt.subplots(2,len(projects),
                            sharex=True,
                            sharey=True,
                            figsize=(7,4))
    for key,ax in zip(projects,axs[0,:]):
        print(key)
        station = key.split('_')[0]
        df =  projects[key][['VPD','Enet','ET',PM_estimate]].dropna().copy()
        x = df['VPD']
        y = df['Enet']
        C = df['ET']
        z = df[PM_estimate]
        hb = ax.hexbin(x=x,
                        y=y,
                        C=C,
                        gridsize=25,
                        extent=[0,2.5,-150,700],
                        cmap = cmap1,
                        norm = norm1,
                        alpha=0.9
                        )
        #grid data for contour plot of PM-results
        x_grid,y_grid = np.mgrid[0:2.5:25j, -150:700:25j]
        zi = griddata((x.values,y.values),z.values,(x_grid,y_grid))
        CS = ax.contour(x_grid,y_grid,zi,levels=[0,0.1,0.2,0.3,0.4],linewidths=1.5,colors='k')
        ax.clabel(CS, CS.levels, inline=True, fontsize=8, fmt='%3.1f')
        # ax.legend(loc='upper right',title = 'Penman-Montheith estimates')

        ax.set_title(f'{titlelist[station]}')
        ax.grid(color='whitesmoke', which='both', axis='both', lw=0.5)
    for key,ax in zip(projects,axs[1,:]):
        station = key.split('_')[0]
        df =  projects[key][['VPD','Enet',PM_estimate,'ET']].dropna().copy()
        x = df['VPD']
        y = df['Enet']
        C = 100*(df[PM_estimate]-df['ET'])/abs(df['ET'])
        # print(f'Rel error 25-75-percentile: {C.quantile(q=0.25)} {C.quantile(q=0.75)}')
        hb = ax.hexbin(x=x,
                        y=y,
                        C=C,
                        gridsize=25,
                        extent=[0,2.5,-150,700],#xmin,xmax,ymin,ymax
                        cmap = cmap2,
                        norm = norm2,
                        # marginals=True,
                        # mincnt = 2 #only display cells with more than mincnt number of points in the cell
                        )
        ax.grid(color='whitesmoke')
    fig.supxlabel('VPD (kPa)')
    fig.supylabel('Enet (W/m2)')
    fig.tight_layout()
    #colorbar alt1
    fig.subplots_adjust(right=0.85)
    cbar_ax_1 = fig.add_axes([0.86, 0.59, 0.03, 0.35])
    cbar_ax_2 = fig.add_axes([0.86, 0.17, 0.03, 0.35])
    cb_1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1),
                    cax = cbar_ax_1,extend='min')
    cb_2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2),
                    cax = cbar_ax_2,extend='both')
    cb_1.ax.set_ylabel('Mean ET (mm/h)',rotation=270, labelpad=17)
    cb_2.ax.set_ylabel('Relative error (%)',rotation=270, labelpad=10)
    fig.savefig(os.path.join(fig_path,f'VPD_Enet_{PM_estimate}.pdf'))
    return

#plot bowen ratio vs VPD
def bowen_ratio_VPD(projects,fig_path):
    fig,axs = plt.subplots(1,len(projects),figsize=(7,2.5),sharex=True,sharey=True)
    bins = np.arange(0,2.75,0.125)
    for ax,key in enumerate(projects):
        station = key.split('_')[0]
        #extract bowen ratio and VPD where H and LE has absolute values over a threshold of 10 W/m2
        criteria = (abs(projects[key]['H_filtered'])>10) & (abs(projects[key]['LE_filtered'])>10)
        data = projects[key].loc[criteria,['VPD','bowen']].copy()
        #only midday values to avoid effect of daily cycle
        data = data.between_time(midday_start[station],midday_end[station]).copy()
        count = data.bowen.groupby(pd.cut(data.VPD,bins=bins,include_lowest=True)).count()
        mean = data.bowen.groupby(pd.cut(data.VPD,bins=bins,include_lowest=True)).mean().loc[count>=5]
        std = data.bowen.groupby(pd.cut(data.VPD,bins=bins,include_lowest=True)).std().loc[count>=5]
        midpoints = mean.index.remove_unused_categories().categories.mid
        axs[ax].scatter(data.VPD,data.bowen,alpha=0.5,c='grey',s=0.5)
        axs[ax].plot(midpoints,mean,color=colorlist[station])
        axs[ax].fill_between(midpoints,mean-std,mean+std,color=colorlist[station],alpha=0.5)
        axs[ax].grid()
        axs[ax].axhline(1,color='k',linewidth=0.5,linestyle='dashed')
        axs[ax].set_title(titlelist[station])
    fig.supylabel('Bowen ratio')
    fig.supxlabel('Vapour pressure deficit (kPa)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'bowen_vpd.pdf'))
    plt.close(fig)
    return

#plot bowen ratio vs SWC
def bowen_ratio_SWC(projects,fig_path):
    SWC_projects = {key: projects[key] for key in projects if 'SWC' in projects[key].columns}
    fig,axs = plt.subplots(1,len(SWC_projects),figsize=(7,2.5),sharey=True)
    for ax,key in enumerate(SWC_projects):
        station = key.split('_')[0]
        grouping_var = 'SWC'

        #extract bowen ratio and VPD where H and LE has absolute values over a threshold of 10 W/m2
        criteria = (abs(projects[key]['H_filtered'])>10) & (abs(projects[key]['LE_filtered'])>10)
        data = SWC_projects[key].loc[criteria,[grouping_var,'bowen']].copy()
        bins = np.linspace(data[grouping_var].min(),data[grouping_var].max(),20)
        #only midday values to avoid effect of daily cycle
        data = data.between_time(midday_start[station],midday_end[station]).copy()
        count = data.bowen.groupby(pd.cut(data[grouping_var],bins=bins,include_lowest=True)).count()
        mean = data.bowen.groupby(pd.cut(data[grouping_var],bins=bins,include_lowest=True)).mean().loc[count>=5]
        std = data.bowen.groupby(pd.cut(data[grouping_var],bins=bins,include_lowest=True)).std().loc[count>=5]
        midpoints = mean.index.remove_unused_categories().categories.mid
        axs[ax].scatter(data[grouping_var],data.bowen,alpha=0.5,c='grey',s=0.5)
        axs[ax].plot(midpoints,mean,color=colorlist[station])
        axs[ax].fill_between(midpoints,mean-std,mean+std,color=colorlist[station],alpha=0.5)
        axs[ax].grid()
        axs[ax].axhline(1,color='k',linewidth=0.5,linestyle='dashed')
        axs[ax].set_title(titlelist[station])
    fig.supylabel('Bowen ratio')
    fig.supxlabel('Soil water content')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,'bowen_swc.pdf'))
    plt.close(fig)
    return

#calculate correlations and write to table
def correlations(projects,data_path,supporting_fig_path):
    #get predictors
    with open(os.path.join('..','metadata','level2_predictor_lists.json')) as file:
        predictor_list = json.load(file)

    #dict for saving results
    pearson_coeff = {}

    #extract correlations for each site
    for key in projects:
        station = key.split('_')[0]
        features = predictor_list[station]
        df = projects[key]
        measured_features = [feat for feat in features if feat in df.columns]
        #snow cvoer status
        snowfree = df.snowfree == 1
        shoulder = df.shoulder == 1
        snowcovered = (df.shoulder == 0) & (df.snowfree == 0)
        snowfree_df= df.loc[snowfree,measured_features+['ET','Enet']]
        shoulder_df = df.loc[shoulder,measured_features+['ET','Enet']]
        snowcovered_df = df.loc[snowcovered,measured_features+['ET','Enet']]

        fig_subfolder = {'Snow-free': os.path.join(supporting_fig_path,'snowfree'),
                        'Shoulder': os.path.join(supporting_fig_path,'shoulder'),
                        'Snow-covered': os.path.join(supporting_fig_path,'snowcovered')
                        }
        for season,folder in fig_subfolder.items():
            if not os.path.exists(folder):
                os.makedirs(folder)

        for snow_status,seasonal_df in zip(['Snow-free','Shoulder','Snow-covered'],
                                        [snowfree_df,shoulder_df,snowcovered_df]):


            # plot correlations for each station and season
            fig,ax = plt.subplots(figsize=(7,6))
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([.91, .3, .03, .4])

            #drop variables stating snow-cover status
            seasonal_df.drop(['shoulder','snowfree'],axis=1,inplace=True)

            #drop deep soil temperature at Adventdalen
            if station=='adventdalen':
                seasonal_df.drop('TS_2_1_1',axis=1,inplace=True)

            #extract correlations matrix
            corr_df = seasonal_df.corr(numeric_only=True)

            #rename columns and index for better readability
            new_index = {i: i.split('_')[0] for i in corr_df.index}
            corr_df.rename(new_index,axis=0,inplace=True)
            new_columns = {i: i.split('_')[0] for i in corr_df.columns}
            corr_df.rename(new_columns,axis=1,inplace=True)
            seasonal_df.rename(new_columns,axis=1,inplace=True)

            #change order of index
            ordered_variables = ['ET','TA','VPD','WS','PA','Enet','SWIN','albedo','LWIN','LWOUT','SHF','TS','SWC','TSR','WTD','GDD']
            corr_df = corr_df.reindex(index=ordered_variables)
            seasonal_df = seasonal_df.loc[:,[var for var in ordered_variables if var in seasonal_df.columns]] 

            #extract correlation coefficients and p-values between ET and other variables
            pearson_r_p = {}
            for var in seasonal_df.columns.drop('ET'):
                no_na = seasonal_df.loc[:,[var,'ET']].dropna()
                if len(no_na)>10:
                    pearson_result = pearsonr(no_na.loc[:,'ET'],no_na.loc[:,var])
                    pearson_r_p[var] = {'r': pearson_result.statistic,
                                        'p-value': pearson_result.pvalue}
            pearson_r_p = pd.DataFrame(pearson_r_p)
            p_mask0 = pearson_r_p.loc['p-value',:]>=0.05
            p_mask1 = pearson_r_p.loc['p-value',:]<=0.05
            p_mask2 = pearson_r_p.loc['p-value',:]<=0.01
            p_mask3 = pearson_r_p.loc['p-value',:]<=0.001
            pearson_r0 = pearson_r_p.loc['r',:].round(2).where(p_mask0)
            pearson_r1 = pearson_r_p.loc['r',:].round(2).where(p_mask1)
            pearson_r1 = pearson_r1.mask(p_mask1, other = pearson_r1.astype('str')+'*')
            pearson_r2 = pearson_r_p.loc['r',:].round(2).where(p_mask2)
            pearson_r2 = pearson_r2.mask(p_mask2, other = pearson_r2.astype('str')+'**')
            pearson_r3 = pearson_r_p.loc['r',:].round(2).where(p_mask3)
            pearson_r3 = pearson_r3.mask(p_mask3, other = pearson_r3.astype('str')+'***')
            pearson_r = pearson_r3.combine_first(pearson_r2.combine_first(pearson_r1.combine_first(pearson_r0)))

            pearson_coeff[(titlelist[station],snow_status)] = pearson_r
            
            #plot correlation matrix (mask upper half)
            mask = np.triu(np.ones_like(corr_df, dtype=bool))
            sns.heatmap(corr_df, mask=mask, annot=True, cmap=plt.cm.Reds, ax=ax,vmin=-1,vmax=1, cbar_ax=cbar_ax,square=True,linewidths=.5)
            ax.set_title(snow_status)

            fig.suptitle(titlelist[station])
            fig.savefig(os.path.join(fig_subfolder[snow_status],f'{station}_correlations.png'))

    pearson_coeff = pd.DataFrame(pearson_coeff)
    ordered_variables.remove('ET')
    ordered_variables.remove('WTD')
    pearson_coeff = pearson_coeff.reindex(index=ordered_variables)
    pearson_coeff.to_csv(os.path.join(data_path,'ET_pearson_incl_shoulder_season.csv'),float_format='%.2f')
    pearson_coeff.drop(['Shoulder'],axis=1,level=1,inplace=True)
    pearson_coeff.to_csv(os.path.join(data_path,'ET_pearson.csv'),float_format='%.2f',header=False)
    pearson_coeff.to_csv(os.path.join(data_path,'ET_pearson_with_header.csv'),float_format='%.2f')
    return

def control_pairplot(projects,fig_path):
    #get predictors
    with open(os.path.join('..','metadata','level2_predictor_lists.json')) as file:
        predictor_list = json.load(file)

    for key in projects:
        station = key.split('_')[0]
        variables = predictor_list[station]
        df = projects[key]
        df['ground_cond'] = 'snow-covered'
        df['ground_cond'].loc[df.snowfree==1] = 'snow-free'
        df['ground_cond'].loc[df.shoulder==1] = 'partly snow-covered'
        variables.append('ground_cond')
        variables.append('ET')
        fig,ax=plt.subplots()
        sns.pairplot(df[variables],ax=ax,hue='ground_cond',kind = 'reg', plot_kws={'line_kws':{'color':'grey'}, 'scatter_kws': {'alpha': 0.07}})
        fig.savefig(os.path.join(fig_path,f'control_pairplot_{station}.png'))
    return

#plot factor analysis
def plot_factor_analysis(projects,fig_path,data_path):
    #perform factor analysis
    details_fig_path = os.path.join(fig_path,'factor_analysis_details')
    details_data_path = os.path.join(data_path,'factor_analysis_details')

    for path in [details_fig_path,details_data_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    results = factor_analysis(projects,details_fig_path,details_data_path)

    selected_sector =  {'finseflux': 'west',
                        'myr2':'all',
                        'iskoras':'all',
                        'adventdalen':'all'
                        }
    #Plot scatter plot of loadings for first two sectors (one plot for each station with subplots for sectors)
        #plot factor loadings, scatterplot of first two factors

    for season,station in results:
        season_fig_path = os.path.join(fig_path,season)
        if not os.path.exists(season_fig_path):
            os.makedirs(season_fig_path)

        if results[season,station]:
            fig,axs = plt.subplots(1,len(results[season,station]),sharex=True,sharey=True, figsize=(7,3), 
                                    subplot_kw=dict(aspect='equal'))
            if len(results[season,station]) == 1:
                axs = [axs]
            for sector,ax in zip(results[season,station],axs):
                loadings = results[season,station][sector].drop(['total_variance','proportion_of_variance','cummulative_variance'], axis=0)
                loadings = loadings.loc[:,['First factor','Second factor']]
                for x,y,s in zip(loadings['First factor'],loadings['Second factor'],loadings.index):
                    ax.text(x,y,s,fontdict={'size':8},
                            horizontalalignment='center',
                            verticalalignment='center') 
                ax.scatter(loadings['First factor'],loadings['Second factor'],alpha=0.7)
                ax.set_xlim(-1,1)
                ax.set_ylim(-1,1)
                ax.grid(which='both', axis='both')
                ax.set_title(sector)
            fig.supxlabel('First factor')
            fig.supylabel('Second factor')
            fig.suptitle(titlelist[station])
            fig.tight_layout()
            fig.savefig(os.path.join(details_fig_path,f'{station}_{season}_scattered_factor_loadings.png'))

    #Compare stations (using selected sector for each station)
    #extract stations with valid results and re-index to get correct shared categorical y-axis
    results_reindexed = []
    stations = []

    for season,station in results:
        if results[season,station]:
            if selected_sector[station] in results[season,station]:
                loadings = results[season,station][selected_sector[station]].drop(['total_variance','proportion_of_variance','cummulative_variance'], axis=0)
                loadings = loadings.loc[:,['First factor','Second factor']]
                results_reindexed.append(loadings)
                stations.append((season,station))
    
    results_reindexed = pd.concat(results_reindexed,keys=stations,axis=1)

    #change order of index
    results_reindexed = results_reindexed.reindex(index=['ET','TA','VPD','WS','PA','SWIN','albedo','LWIN','LWOUT','SHF','TS','SWC','TSR','GDD'])
    results_reindexed.to_csv(os.path.join(data_path,'factor_analysis_results.csv'),float_format='%.2f')

    #plot factor loadings, horizontal stacked barplot
    season_title = {'snowfree':'Snow-Free',
                    'snowcovered': 'Snow-Covered',
                    'shoulder': 'Shoulder'}
    for season, season_data in results_reindexed.groupby(level=0,axis=1):
        fig,axs = plt.subplots(1,len(season_data.columns.levels[1]),sharex=True,sharey=True, figsize=(7,4))
        
        for (station,station_loadings),ax in zip(season_data.groupby(level=1,axis=1),axs):
            station_loadings.plot.barh(stacked=True, ax=ax,legend=False)
            ax.grid()
            ax.set_title(titlelist[station])
        #handles labels are equal for all plots as long as only two factors are included in plot
        handles, labels = ax.get_legend_handles_labels()
        labels = [label.split(', ')[-1].split(')')[0] for label in labels]
        fig.supxlabel('Factor loadings')
        fig.supylabel('Observed variables')
        fig.subplots_adjust(top=0.85)
        fig.legend(handles,labels,ncols=2,loc='upper center')
        fig.savefig(os.path.join(fig_path,season,'stacked_factor_loadings.pdf')) #season fig path
        #
        #plot factor loadings, scatterplot of first two factors
        fig,axs = plt.subplots(3,4,sharex=True,sharey=True, figsize=(7,6), 
                                subplot_kw=dict(aspect='equal'))
        for station,station_loadings in season_data.groupby(level=1,axis=1):
            station_loadings = station_loadings.droplevel(level=[0,1],axis=1)
            if 'WTD' in station_loadings.index:
                station_loadings.drop('WTD',axis=0,inplace=True)
            for var,ax in zip(station_loadings.index,axs.flatten()):
                ax.scatter(station_loadings.loc[var,'First factor'],
                            station_loadings.loc[var,'Second factor'],
                            color=colorlist[station])
                ax.set_title(var)

        for ax in axs.flatten():
            ax.grid(axis='both',which='both')
            ax.set_ylim(-1,1)
            ax.set_xlim(-1,1)
        fig.supxlabel('First factor')
        fig.supylabel('Second factor')
        fig.suptitle(season_title[season])
        fig.tight_layout()
        fig.savefig(os.path.join(fig_path,season,'scattered_factor_loadings.pdf')) #season figpath
        plt.close('all')
    return

#plot sensitivity of observered and penman-monteith evaporation to controls
def E_PM_sensitivity(projects,fig_path, PM_estimate = 'ET_PM_Enet'):
    fig_path = os.path.join(fig_path,'evaporation_sensitivity')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    variable_bins = {
                'WS': np.arange(0,15,0.6),
                'Rnet': np.arange(-250,800,30),
                'Enet': np.arange(-250,800,30),
                'VPD': np.arange(0,2.75,0.125),
                'TSR': np.arange(0,408,24),
                'GDD': np.arange(0,2070,90),
                }

    xlabels = {
                'SWC': 'Soil water content ($m^3/m^3$)',
                'WS': 'Wind speed (m/s)',
                'Rnet': 'Net radiation ($W/m^2$)',
                'VPD': 'Vapour pressure deficit (kPa)',
                'Enet': 'Net energy ($W/m^2$)',
                'TSR': 'Time since rain (hours)',
                'GDD': 'Growing degree day',
                }

    for variable in ['SWC','WS','Rnet','VPD','TSR','Enet','GDD']:
        #get projects where variable is included in dataset
        var_projects = {key: projects[key] for key in projects if variable in projects[key].columns}

        #Share x-axis for all variable except SWC
        sharex = True
        if variable == 'SWC':
            sharex = False

        if len(var_projects)>0:
            #figure for sensitivity of observed and modelled evaporation (upper row) and relative error (lower row)
            fig,axs = plt.subplots(2,len(var_projects),figsize=(7,4),sharey='row', sharex=sharex)

            for ax,key in enumerate(var_projects):
                station = key.split('_')[0]
                grouping_var = variable
                if (station == 'iskoras') and (variable == 'SWC'):
                    grouping_var = 'SWC_2_1_1'

                #extract data
                data = var_projects[key].loc[:,[grouping_var,PM_estimate,'ET']].copy()
                data['rel_error'] = 100*(data[PM_estimate]-data['ET'])/abs(data['ET'])
                data.dropna(axis=0,how='any',inplace=True)

                if len(data)>40:
                    #make bins
                    if grouping_var in variable_bins:
                        bins = variable_bins[grouping_var]
                    else:
                        bins = np.linspace(data[grouping_var].min(),data[grouping_var].max(),20)

                    #exclude outliers in relative error (more than 1.5*IQR from Q1 or Q3) 
                    # and get mean and std for each bin (after excluding outliers)
                    means = []
                    stds = []
                    outliers = []
                    counts = []

                    for bin,bin_data in data.groupby(pd.cut(data[grouping_var],bins=bins, include_lowest=True)):
                        q1 = bin_data.quantile(0.25)
                        q3 = bin_data.quantile(0.75)
                        iqr = q3-q1
                        bin_outliers = (bin_data>(q3+1.5*iqr))|((bin_data<(q1-1.5*iqr)))
                        #drop time step for ALL if relative error is an outlier
                        bin_data_no_outliers = bin_data.loc[~bin_outliers['rel_error']].dropna(axis=0,how='any')
                        outliers.append(bin_data.loc[bin_outliers['rel_error']])
                        means.append(bin_data_no_outliers.mean())
                        stds.append(bin_data_no_outliers.std())
                        counts.append(bin_data_no_outliers.rel_error.count())

                    count_data = pd.Series(counts)
                    outliers = pd.concat(outliers,axis=0)
                    outliers.sort_index(inplace=True)
                    mean_data = pd.concat(means,axis=1).transpose()
                    std_data = pd.concat(stds,axis=1).transpose()

                    #only use bins with at least 10 data points
                    mean_data=mean_data.loc[count_data>10,:]
                    std_data=std_data.loc[count_data>10,:]

                    #plot sensitivity of observed and modelled evaporation to variable (mean and std)
                    axs[0,ax].plot(mean_data[grouping_var],mean_data['ET'].where(std_data['ET'].notna()),color=colorlist[station])
                    axs[0,ax].fill_between(mean_data[grouping_var],
                                        mean_data['ET']-std_data['ET'],
                                        mean_data['ET']+std_data['ET'],color=colorlist[station],alpha=0.5)
                    axs[0,ax].plot(mean_data[grouping_var],mean_data[PM_estimate].where(std_data[PM_estimate].notna()),color='grey')
                    axs[0,ax].fill_between(mean_data[grouping_var],
                                        mean_data[PM_estimate]-std_data[PM_estimate],
                                        mean_data[PM_estimate]+std_data[PM_estimate],color='grey',alpha=0.5)
                    axs[0,ax].grid()
                    axs[0,ax].set_title(titlelist[station])

                    #plot sensitivity of relative error to variable (mean and std)
                    axs[1,ax].plot(mean_data[grouping_var],mean_data['rel_error'],color='tan')
                    axs[1,ax].fill_between(mean_data[grouping_var],
                                        mean_data['rel_error']-std_data['rel_error'],
                                        mean_data['rel_error']+std_data['rel_error'],
                                        color='tan',alpha=0.5)
                    axs[1,ax].grid()
                else:
                    continue

            axs[0,0].set_ylabel('Evaporation (mm/h)')
            axs[1,0].set_ylabel('Relative error (%)')
            fig.align_ylabels(axs[:,0])
            fig.supxlabel(xlabels[grouping_var])
            fig.tight_layout()
            fig.savefig(os.path.join(fig_path,f'{variable}_{PM_estimate}.pdf'))
            plt.close('all')
        else:
            continue
    return

#plot seasonality of observed and penman-monteith evaporation
def ET_PM_seasonality(projects,fig_path, PM_estimate = 'ET_PM_Enet'):
    fig_path = os.path.join(fig_path,'evaporation_sensitivity')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    #figure for sensitivity of observed and modelled evaporation (upper row) and relative error (lower row)
    fig,axs = plt.subplots(2,len(projects),figsize=(7,4),sharey='row', sharex=True)

    for col,key in enumerate(projects):
        station = key.split('_')[0]
        #extract data
        data = projects[key].loc[:,[PM_estimate,'ET']].copy()
        data.dropna(how='any',axis=0,inplace=True)
        data['rel_error'] = 100*(data[PM_estimate]-data['ET'])/abs(data['ET'])

        #make weekly bins
        bins = list(np.linspace(1,366,52))

        #exclude outliers (more than 3*IQR from Q1 or Q3) and get mean and std for each bin (after excluding outliers)
        midpoints = []
        means = []
        stds = []
        outliers = []
        medians = []
        q25s = []
        q75s = []

        for bin,bin_data in data.groupby(pd.cut(data.index.dayofyear,bins=bins, include_lowest=True)):
            q1 = bin_data.quantile(0.25)
            q3 = bin_data.quantile(0.75)
            iqr = q3-q1
            bin_outliers = (bin_data>(q3+1.5*iqr))|((bin_data<(q1-1.5*iqr)))
            #drop time step for ALL if relative error is an outlier
            bin_data_no_outliers = bin_data.loc[~bin_outliers['rel_error']]
            outliers.append(bin_data.loc[bin_outliers['rel_error']])
            midpoints.append(bin.mid)
            means.append(bin_data_no_outliers.mean())
            stds.append(bin_data_no_outliers.std())
            medians.append(bin_data_no_outliers.median())
            q25s.append(bin_data_no_outliers.quantile(0.25))
            q75s.append(bin_data_no_outliers.quantile(0.75))

        outliers = pd.concat(outliers,axis=0)
        outliers.sort_index(inplace=True)
        mean_data = pd.concat(means,axis=1).transpose()
        std_data = pd.concat(stds,axis=1).transpose()
        median_data = pd.concat(medians,axis=1).transpose()
        q25_data = pd.concat(q25s,axis=1).transpose()
        q75_data = pd.concat(q75s,axis=1).transpose()

        #plot weekly mean and std of ET and PM-estimate
        axs[0,col].plot(midpoints,mean_data['ET'].where(std_data['ET'].notna()),color=colorlist[station])
        axs[0,col].fill_between(midpoints,
                                mean_data['ET']-std_data['ET'],
                                mean_data['ET']+std_data['ET'],
                                color=colorlist[station],alpha=0.5)
        axs[0,col].plot(midpoints,mean_data[PM_estimate].where(std_data[PM_estimate].notna()),color='grey')
        axs[0,col].fill_between(midpoints,
                                mean_data[PM_estimate]-std_data[PM_estimate],
                                mean_data[PM_estimate]+std_data[PM_estimate],
                                color='grey',alpha=0.5)

        #plot weekly median and iqr of relative error
        axs[1,col].plot(midpoints,median_data['rel_error'],color='tan')
        axs[1,col].fill_between(midpoints,
                                q25_data['rel_error'],
                                q75_data['rel_error'],
                                color='tan',alpha=0.5)

        [ax.grid() for ax in axs[:,col]]
        [ax.set_xlabel('') for ax in axs[:,col]]

    axs[0,0].set_ylabel('Evaporation (mm/h)')
    axs[1,0].set_ylabel('Relative error (%)')
    fig.supxlabel('Day of year')
    fig.align_ylabels(axs[:,0])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,f'seasonality_{PM_estimate}.pdf'))
    plt.close('all')
    return

def ET_PM_seasonality_medians(projects,fig_path, PM_estimate = 'ET_PM_Enet'):
    fig_path = os.path.join(fig_path)
    #figure for sensitivity of observed and modelled evaporation (upper row) and relative error (lower row)
    fig,axs = plt.subplots(2,len(projects),figsize=(7,4),sharey='row', sharex=True)

    for col,key in enumerate(projects):
        station = key.split('_')[0]
        #extract data
        data = projects[key].loc[:,[PM_estimate,'ET']].copy()
        data.dropna(how='any',axis=0,inplace=True)
        data['rel_error'] = 100*(data[PM_estimate]-data['ET'])/abs(data['ET'])

        weekly_median = data.groupby(data.index.isocalendar().week).median()
        weekly_q25 = data.groupby(data.index.isocalendar().week).quantile(q=0.25)
        weekly_q75 = data.groupby(data.index.isocalendar().week).quantile(q=0.75)
        #plot weekly mean and std of ET and PM-estimate
        axs[0,col].plot(weekly_median.index.astype(float),weekly_median['ET'],color=colorlist[station])
        axs[0,col].fill_between(weekly_median.index.astype(float),
                                weekly_q25['ET'],
                                weekly_q75['ET'],
                                color=colorlist[station],alpha=0.5)
        axs[0,col].plot(weekly_median.index.astype(float),weekly_median[PM_estimate],color='grey')
        axs[0,col].fill_between(weekly_median.index.astype(float),
                                weekly_q25[PM_estimate],
                                weekly_q75[PM_estimate],
                                color='grey',alpha=0.5)
        #plot weekly mean and std of relative error
        axs[1,col].plot(weekly_median.index.astype(float),weekly_median['rel_error'],color='tan')
        axs[1,col].fill_between(weekly_median.index.astype(float),
                                weekly_q25['rel_error'],
                                weekly_q75['rel_error'],
                                color='tan',alpha=0.5)

        [ax.grid() for ax in axs[:,col]]
        [ax.set_xlabel('') for ax in axs[:,col]]

    axs[0,0].set_ylabel('Evaporation (mm/h)')
    axs[1,0].set_ylabel('Relative error (%)')
    fig.supxlabel('Week of year')
    fig.align_ylabels(axs[:,0])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,f'seasonality_{PM_estimate}_medians.pdf'))
    plt.close('all')
    return

#plot energy balance
def energy_balance(projects,fig_path,table_path,available_energy='Enet'):
    table_path = os.path.join(table_path,'energy_balance')
    if not os.path.exists(table_path):
        os.makedirs(table_path)

    fig,axs= plt.subplots(1,len(projects),figsize=(7,2))
    for key,ax in zip(projects,axs):
        df = projects[key]
        if available_energy in df.columns:
            station = key.split('_')[0]
            data = df.loc[:,['LE_filtered','H_filtered',available_energy]]
            data.dropna(how='any',axis=0,inplace=True)
            ax.scatter(data[available_energy],data.H_filtered+data.LE_filtered,alpha=0.5,color=colorlist[station])
            x = np.linspace(-200,600,2)
            ax.plot(x,x,color='k',label='1:1')
            ax.grid()
            ax.set_title(titlelist[station])
            #fit linear model
            x = data[available_energy].values
            if len(x)>10:
                xx = sm.add_constant(x)
                yy = (data.H_filtered+data.LE_filtered).values
                reg_fit = sm.OLS(yy, xx).fit()
                # with open(os.path.join(table_path,f'summary_EB_{station}_{available_energy}.txt'),'w') as file:
                #     file.write(reg_fit.summary().as_text())
                #plot regression line
                x_plot = np.linspace(-200,600)
                b0,b1 = reg_fit.params
                y_plot = b0+b1*x_plot
                ax.plot(x_plot,y_plot,linestyle='dashed',color='k', label = f'slope={b1:.2f}')
                #add text stating slope of regression line
                xylabel = (30+(x_plot[0]+x_plot[-1])/2, -30+(y_plot[0]+y_plot[-1])/2)
                label = f'slope={b1:.2f}'
                p1 = ax.transData.transform_point((x_plot[0], y_plot[0]))
                p2 = ax.transData.transform_point((x_plot[-1], y_plot[-1]))
                dy = (p2[1] - p1[1])
                dx = (p2[0] - p1[0])
                rotn = np.degrees(np.arctan2(dy, dx))
                ax.annotate(label, xy=xylabel, ha='center', va='center', rotation=rotn)
                # ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.08))
            else:
                continue
    fig.supxlabel(available_energy)
    fig.supylabel('H + LE')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path,f'energy_balance_{available_energy}.pdf'))
    return