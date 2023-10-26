import pandas as pd
import matplotlib.pyplot as plt
import os, json

with open(os.path.join('..','metadata','fluxnet_symbology.json')) as file:
    fluxnet_symbology = json.load(file)

fig_path = os.path.join('..','plot')

if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title

#SUBSET of fluxnet above 60 degrees latitude
fluxnet_metadata = pd.read_csv(os.path.join('..','..','fluxnet','fluxnet_above_LAT60.csv'),
sep=';', index_col=0)

#separate wetland and open shrub sites from forest sites
ENF = fluxnet_metadata.loc[fluxnet_metadata['PFT_group'] == 'ENF']
WET = fluxnet_metadata.loc[fluxnet_metadata['PFT_group'] == 'WET']
other = fluxnet_metadata.loc[fluxnet_metadata['PFT_group'] == 'other']

#plot fluxnet data
fig,ax = plt.subplots(figsize= (4,4))
ax.scatter(ENF['MAT'], ENF['MAP'], color = fluxnet_symbology['ENF']['color'],
            marker = fluxnet_symbology['ENF']['marker'],
            label = 'High latitude forest, FLUXNET2015',alpha=0.8, s=25)
ax.scatter(WET['MAT'], WET['MAP'], color = fluxnet_symbology['WET']['color'],
            marker = fluxnet_symbology['WET']['marker'],
            label = 'High latitude wetland, FLUXNET2015',alpha=0.8, s=25)
ax.scatter(other['MAT'], other['MAP'],
            color = fluxnet_symbology['other']['color'],
            marker = fluxnet_symbology['other']['marker'],
            label = 'Other high latitue ecosystems, FLUXNET2015',alpha=0.8, s=25)
ax.scatter(-1.7,980, color = 'tab:blue', label = 'Finse', s=50)
ax.scatter(1.4,790, color = 'tab:green', label = 'Hisåsen', s=50)
ax.scatter(-2.1,390, color = 'tab:orange', label = 'I\u0161koras', s=50)
ax.scatter(-4,220, color = 'tab:red', label = 'Adventdalen', s=50)
ax.set_ylim(0,1200)
ax.set_xlim(-15,10)
ax.set_ylabel('Annual precipitation (mm)')
ax.set_xlabel('Annual temperature (\u00b0 C)')
ax.grid()

fig.tight_layout()
fig.savefig(fig_path)
