import pandas as pd
import numpy as np

'''
Derives estimates of areodynamic conductance and ecosystem scale surface
conductance from eddyprodata.
'''


def G_a(wind_speed, ustar):
    """Computes aerodynamic conductance using equation from
    Verma (1989) Aerodynamic resistances to transfers of
    heat, mass and momentum. Estimation of Areal Evapotranspiration, 177.
    kB-1 is assumed to be 2 (as in e.g. Launiainen 2016 and Helbig 2020),
    and von Káram constant 0.4

    Parameters:
    wind_speed: wind speed (m/s)
    ustar: friction velocity (m/s)

    Returns:
    G_a: atmospheric conductance in m/s
    """
    G_a = ((wind_speed/ustar**2)+(2/(0.4*ustar))*0.89**(2/3))**(-1)
    return G_a


def G_s(T_a, air_pressure, VPD, LE, Enet, G_a):
    '''Computes surface resistance from inversed Penman-Monteith,
    when available energy is at least 100 W m-2 and LE is at least 50 W m-2. #Helbig et al. 2020 used Enet>100 and LE>50.


    Parameters:
    T_a: air temperature (deg C)
    air_pressure: atmospheric pressure (Pa)
    VPD: vapour pressure deficit (Pa)
    LE: latent heat flux (W m-1)
    Enet: Available energy (W m-1)
    G_a:  Atmospheric conductance (m s-1)

    Returns:
    G_s: surface conductance (m s-1)
    '''
    Rd_a = 287.04 #Gas constant for dry air (J kg-1 K-1). From Dingman 3.ed table 3.1. Org.source: List (1971) Smithsonian Meteorological Tables 6th ed.
    cp = 1004.834 #specific heat of air for constant pressure (J kg-1 K-1)
    l = (2.501 - 0.00236*T_a)*10**6 #latent heat of vaporization (J kg-1)
    rho_a = air_pressure/(Rd_a*(T_a+273.15)) #mass density of dry air (kg m-3). should it be mass density accounting for actual air moisture?
    gamma = cp*air_pressure/(0.622*l) #psychrometric constant (Pa K−1)
    delta = 1000*((2508.3/(T_a+237.3)**2)*np.exp((17.3 * T_a)/(T_a + 237.3))) #slope of saturation vapor pressure–temperature curve (Pa C−1) From Dingman 3rd ed. p 254
    r_s = (1/G_a)*((delta/gamma)*(Enet/LE)-(delta/gamma)-1)+(rho_a*cp*VPD)/(gamma*LE)
    G_s=1/r_s
    filter = ((G_s<0) | (G_s>0.05)) #0.05 corresponds to 50 mm/S
    G_s.loc[filter]=np.nan
    return G_s
