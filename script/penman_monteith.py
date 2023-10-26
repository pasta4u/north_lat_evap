import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_evaporation(Enet, gs, Ta, VPD, air_pressure, wind_speed, ustar, daily=False, ga_from_ustar=False, station=None):
    '''
    Enet: available energy in Wm-2
    gs: surface conductance in m/s
    Ta: air temperature at 2 m in deg C
    VPD: vapour pressure deficit of air in Pa
    air_pressure: atmospheric pressure in Pa
    wind_speed: wind speed in m/s
    ustar: friction velocity in m/s
    '''

    #Estimate aerodynamic conductance from ustar and wind_speed if station is not stated
    if not station:
        ga_from_ustar=True

    # median snowfree z0 values from stability ignorant estimates (z0 = z / np.exp(k*wind_speed/u_star)
    z0_values = {
                'finseflux': 0.06,
                'iskoras': 0.04,
                'myr1': 0.3,
                'myr2': 0.3,
                'adventdalen': 0.01}

    z_values = {
                'myr1': 2.7,
                'myr2': 2.8,
                'iskoras': 2.8,
                'finseflux': 4.4,
                'adventdalen': 2.8}

    #Estimate aerodynamic conductance in m/s using ustar and wind_speed
    if ga_from_ustar:
        ra = (wind_speed/ustar**2)+(4/ustar)
        ga = 1/ra
        # ga = ((wind_speed/ustar**2)+(2/(ustar))*0.89**(2/3))**(-1)

    #Estimate aerodynamic conductance in m/s using windspeed and z0
    else:
        z = z_values[station]
        z0 = z0_values[station]
        ga = wind_speed/(6.25*(np.log(z/z0)))**2


    Rd_a = 287.04 #Gas constant for dry air (J kg-1 K-1). From Dingman 3.ed table 3.1. Org.source: List (1971) Smithsonian Meteorological Tables 6th ed.
    cp = 1005 #specific heat of air for constant pressure (J kg-1 K-1) OBS: Finn kjelde for denne!
    rho_w = 1000 #mass density of water

    #latent heat of vaporization (J kg-1)
    l_v = (2.501 - 0.00236*Ta)*10**6

    #mass density of dry air (kg m-3)
    rho_a = air_pressure/(Rd_a*(Ta+273.15))

    #slope of saturation vapour pressure temperature relationship in Pa/degC
    delta = 1000*((2508.3/(Ta+237.3)**2) * np.exp((17.3 * Ta)/(Ta + 237.3)))

    #psychrometric constant  Pa/degC
    gamma = (cp * air_pressure)/(0.622 * l_v)

    #Penman-Monteith evaporation in mm/s
    E = 1000*(delta*Enet+rho_a*cp*ga*VPD)/(rho_w*l_v*(delta+gamma*(1+ga/gs)))

    #Evaporation in mm/h
    E = E*60*60

    if daily:
        #Evaporation in mm/day
        E = E*24

    return E

def optimize_gs(df, station,subset_id,biomet_level):
    paper_path = os.path.join('C:',os.sep,'Users','astridva',
                        'Dropbox','Apper','Overleaf',
                        'magnitudes_and_controls_on_evaporation')
    fig_path = os.path.join(paper_path,'images','supplementary','observations',biomet_level,'penman_monteith_optimization')
    table_path = os.path.join(paper_path,'data','supplementary','observations',biomet_level,'penman_monteith_optimization')
    for path in [fig_path,table_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    optimal_gs = {}
    fig,axs=plt.subplots(2,3,figsize=(7,4),sharex=True,sharey=True)
    for row,ga_from_ustar,ga_type in zip(range(2),[True,False],['ga_from_ustar','ga_from_z0']):
        optimal_gs[ga_type]={}
        for col,available_energy in zip(range(3),['Rnet','Enet','H_LE']):
            axs[0,col].set_title(available_energy)
            gs_list = []
            rmse_list = []
            if available_energy in df.columns:
                for gs in np.linspace(0,0.05,500):
                    pm_estimate = simulate_evaporation(Enet=df[available_energy],
                                                        gs = gs,
                                                        Ta = df.TA_1_1_1,
                                                        VPD = df.VPD_1_1_1*1000,
                                                        air_pressure = df.air_pressure,
                                                        wind_speed = df.wind_speed,
                                                        ustar = df['u*_filtered'],
                                                        station=station,
                                                        ga_from_ustar=ga_from_ustar)
                    gs_list.append(gs)
                    rmse = np.sqrt(np.mean((pm_estimate-df['ET_filtered'])**2))
                    # rmse = metrics.mean_squared_error(df['ET_filtered'], pm_estimate, squared=False) (drop na-values)
                    rmse_list.append(rmse)
            else:
                continue
            axs[row,col].scatter(gs_list,rmse_list, color='grey')
            min_rmse = min(rmse_list)
            opt_gs = gs_list[rmse_list.index(min_rmse)]
            axs[row,col].scatter(opt_gs,min(rmse_list), color='tab:red')
            axs[row,col].text(opt_gs,0.025, s=f'opt gs = {opt_gs:.4f}',color='tab:red')
            optimal_gs[ga_type][available_energy]=opt_gs
            optimal_gs[ga_type][f'RMSE_{available_energy}']=min_rmse
    axs[0,0].set_ylabel('gs (ga from u*))')
    axs[1,0].set_ylabel('gs (ga from z0))')
    fig.supylabel('RMSE')
    fig.suptitle(station)
    fig.savefig(os.path.join(fig_path,f'optimal_gs_{station}_{subset_id}.png'))
    plt.close('all')
    optimal_gs=pd.DataFrame(optimal_gs)
    optimal_gs.to_csv(os.path.join(table_path,f'optimal_gs_{station}_{subset_id}.csv'),float_format='%.4f')
    return optimal_gs