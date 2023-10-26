import pandas as pd
import numpy as np

def simulate_evaporation(Enet, Ta, air_pressure,pt_alpha, daily=False):
    '''
    Enet: available energy in Wm-2
    Ta: air temperature at 2 m in deg C
    air_pressure: atmospheric pressure in Pa
    pt_alpha: Priestley-Taylor alpha

    '''
    Rd_a = 287.04 #Gas constant for dry air (J kg-1 K-1). From Dingman 3.ed table 3.1. Org.source: List (1971) Smithsonian Meteorological Tables 6th ed.
    cp = 1005 #specific heat of (dry?) air for constant pressure (J kg-1 K-1) OBS: Finn kjelde for denne!
    rho_w = 1000 #mass density of water

    #latent heat of vaporization (J kg-1)
    l_v = (2.501 - 0.00236*Ta)*10**6

    #slope of saturation vapour pressure temperature relationship in Pa/degC
    delta = 1000*((2508.3/(Ta+237.3)**2) * np.exp((17.3 * Ta)/(Ta + 237.3)))

    #psychrometric constant  Pa/degC
    gamma = (cp * air_pressure)/(0.622 * l_v)

    #Priestley-Taylor evaporation in mm/s
    E = 1000*(pt_alpha*delta*Enet)/(rho_w*l_v*(delta+gamma))

    #Evaporation in mm/h
    E = E*60*60

    if daily:
        E = E*24

    return E
