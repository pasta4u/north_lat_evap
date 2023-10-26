import numpy as np

def get_constants(temperature):
    '''
    Constants for the Using Magnus-Tetens relationship to calulate
    saturation vapour pressure  (see Dingman 3rd ed. page 113).
    '''
    #constants
    a = 611
    b = np.where(temperature<0,21.874,17.269)
    c = np.where(temperature<0,265.49,237.29)
    return a,b,c

def convert_dewpoint(dewpoint,temperature,output_var='VPD'):
    '''
    dewpoint_temperature: dewpoint temperature in degrees celsius
    temperature: temperature in in degrees celsius
    output_var: either VPD, RH or e.

    returns: vapour pressure deficit in PA

    Using Magnus-Tetens relationship to convert from dewpoint temperature to
    vapour pressure deficit. Same conversion as in Vuichard and Papale (2015)
    "Filling the gaps in meteorological FLUXNET data.", but with corrected sign
    in equation (see Dingman 3rd ed. page 113).
    '''

    a,b,c = get_constants(temperature)

    #vapour pressure (hPa)
    e = a*np.exp((b*dewpoint)/(dewpoint+c))

    #saturation vapour pressure (hPa)
    es = a*np.exp((b*temperature)/(temperature+c))

    #vapour pressure deficit (Pa)
    vpd = es - e

    #Relative humidity
    rh = 100*e/es

    if output_var == 'VPD':
        output = vpd

    elif output_var == 'RH':
        output = rh

    elif output_var == 'e':
        output = e

    else:
        print("Set output_var to 'VPD', 'RH' or 'e'")

    return output

def RH_to_VPD(RH, temperature):
    #constants
    a,b,c = get_constants(temperature)

    es = a*np.exp((b*temperature)/(temperature+c))
    e = (RH/100) * es

    vpd = es - e

    return vpd

def LE_to_ET(LE, temperature):
    '''
    Returns ET in mm/h from input
    LE in W/m2
    temperature in degrees celsius
    '''
    #Dingman 3ed page 116
    latent_heat_vap = (2.501 - 0.00236*temperature)*10**6 #J/kg
    latent_heat_fu = 0.334 * 10**6 #J/kg
    #Dingman 3ed page 545
    mass_density = 1000 - 0.019549*abs(temperature-3.98)**1.68
    #ET in m/s
    ET = LE/(mass_density*latent_heat_vap)
    ET.loc[temperature<=0] = LE/(mass_density*(latent_heat_vap+latent_heat_fu))
    #ET in  mm/h
    ET = ET * 1000 * 60 * 60
    return ET

def ET_to_LE(ET, temperature):
    '''
    Returns LE in W/m2 from input
    ET in mm/h
    temperature in degrees celsius
    '''
    #Dingman 3ed page 116
    latent_heat_vap = (2.501 - 0.00236*temperature)*10**6 #J/kg
    latent_heat_fu = 0.334 * 10**6 #J/kg
    # mass_density = 1000 - 0.019549*abs(temperature-3.98)**1.68 (Dingman 3ed page 545)
    #Use constant mass density = 1000 kg/m3 since this is used by eddypro in conversion from h2o-flux in mmol/m2s to ET in mm/h
    mass_density = 1000
    #ET from mm/h to m/s
    ET = ET.copy()/(1000*60*60)
    #LE in J s-1m-2 = W/m2
    LE = ET*mass_density*latent_heat_vap
    LE.loc[temperature<=0] = ET*mass_density*(latent_heat_vap+latent_heat_fu)
    return LE