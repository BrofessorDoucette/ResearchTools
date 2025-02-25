import IRBEM
import numpy as np

def calculate_lstar_at_magnetic_equator_T89(dates, alt, lat, lon, kp):
    
    equator_model = IRBEM.MagFields(path = r"/project/spdr/opt/IRBEM/libirbem.so",
                            options = [0,0,0,0,0], 
                            kext = "T89",
                            sysaxes = 0, #GDZ
                            verbose = False)
    
    locations_of_equator = []
    
    for T in range(len(dates)):
        
        sat_coords = {
            "dateTime" : dates[T],
            "x1" : alt[T],
            "x2" : lat[T],
            "x3" : lon[T],
        }

        mag_inputs = {
            "Kp" : kp[T]
        }
        
        locations_of_equator.append(equator_model.find_magequator(X = sat_coords, maginput = mag_inputs)["XGEO"])
        
    locations_of_equator = np.vstack(locations_of_equator)
    
        
    lstar_model = IRBEM.MagFields(path = r"/project/spdr/opt/IRBEM/libirbem.so",
                            options = [1,0,2,2,0], 
                            kext = "T89",
                            sysaxes = 2, #GSM!
                            verbose = False)
    
    lstar_calculated = []
        
    for T in range(0, len(locations_of_equator), 25):
        
        equator_coords = {
            
            "dateTime" : dates[T : T + 25],
            "x1" : locations_of_equator[T : T + 25, 0].flatten(),
            "x2" : locations_of_equator[T : T + 25, 1].flatten(),
            "x3" : locations_of_equator[T : T + 25, 2].flatten()
            
        }
        
        mag_inputs = {
            "Kp" : kp[T : T + 25].flatten()
        }
        
        lstar_calculated.append(lstar_model.make_lstar(X = equator_coords, maginput = mag_inputs)["Lstar"])
        
        
    return np.hstack(lstar_calculated).flatten()