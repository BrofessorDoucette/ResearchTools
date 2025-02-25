import matplotlib
import astropy
import astropy.time
from cdflib.epochs_astropy import CDFAstropy as cdfepoch
import data_loader
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os_helper
import pandas as pd
import time
import multiprocessing as mp
import os
import data_loader
import time

import useful_irbem_wrappers as irbem


if __name__ == "__main__":

    year = 2013
    sat = "a"
    
    start_date = datetime.datetime(year = year, month = 1, day = 1)
    end_date = datetime.datetime(year = year, month = 12, day = 31, hour = 23, minute = 59, second = 59)
    
    MAGEPHEM = data_loader.load_raw_data_from_config(id=["RBSP", "MAGEPHEM", "T89D"],
                                                     start = start_date,
                                                     end = end_date, 
                                                     satellite = sat,
                                                     root_data_dir = r"/project/rbsp/soc/Server/ECT/MagEphem/",
                                                     use_config_keys_in_subdir = False)
    
    JULIAN_DATES = MAGEPHEM["JulianDate"]
    ALTITUDE = MAGEPHEM["Rgeod_Height"]
    LATITUDE = MAGEPHEM["Rgeod_LatLon"][:, 0]
    LONGITUDE = MAGEPHEM["Rgeod_LatLon"][:, 1]
    LONGITUDE[LONGITUDE < 0] += 360
    L = MAGEPHEM["Lm_eq"]
    
    VALID_JULIAN_DATES = (0 < JULIAN_DATES) & (JULIAN_DATES < 2460728.5255208) #Feb / 22 / 2025... should be much greater than all dates in RBSP dataset
    
    JULIAN_DATES = JULIAN_DATES[VALID_JULIAN_DATES]
    ALTITUDE = ALTITUDE[VALID_JULIAN_DATES]
    LATITUDE = LATITUDE[VALID_JULIAN_DATES]
    LONGITUDE = LONGITUDE[VALID_JULIAN_DATES]
    L = L[VALID_JULIAN_DATES]
    
    print(JULIAN_DATES)
    
    MAGEPHEM_TIME = astropy.time.Time(JULIAN_DATES, format="jd").unix
    
    valid_altitude = (0 < ALTITUDE) & (ALTITUDE < 1000000) & np.isfinite(ALTITUDE)
    valid_latitude = (-90 <= LATITUDE) & (LATITUDE <= 90) & np.isfinite(LATITUDE)
    valid_longitude = (0 <= LONGITUDE) & (LONGITUDE <= 360) & np.isfinite(LONGITUDE)
    valid_TIME = np.isfinite(MAGEPHEM_TIME)
    valid_L = (0 < L) & (L < 10000) & np.isfinite(L)
    all_valid = valid_altitude & valid_latitude & valid_longitude & valid_TIME & valid_L
    
    MAGEPHEM_TIME = MAGEPHEM_TIME[all_valid]
    ALTITUDE = ALTITUDE[all_valid]
    LATITUDE = LATITUDE[all_valid]
    LONGITUDE = LONGITUDE[all_valid]
    L = L[all_valid]
    
    OMNI = data_loader.load_raw_data_from_config(id = ["OMNI", "ONE_HOUR_RESOLUTION"],
                                                 start = start_date,
                                                 end = end_date, 
                                                 root_data_dir = "./../raw_data/")

    OMNI_TIME = cdfepoch.unixtime(OMNI["Epoch"])
    KP = OMNI["KP"].astype(np.float64)

    invalid_omni_times = (OMNI_TIME < 0) | (KP < 0) | (KP >= 99) | np.isnan(KP) | np.isnan(OMNI_TIME)
    KP[invalid_omni_times] = np.nan
        
    KP_INTERPOLATED = np.interp(MAGEPHEM_TIME, OMNI_TIME, KP, left = np.nan, right = np.nan)    
    
    finite_kp = np.isfinite(KP_INTERPOLATED)
    
    MAGEPHEM_TIME = MAGEPHEM_TIME[finite_kp]
    ALTITUDE = ALTITUDE[finite_kp]
    LATITUDE = LATITUDE[finite_kp]
    LONGITUDE = LONGITUDE[finite_kp]
    L = L[finite_kp]
    KP_INTERPOLATED = KP_INTERPOLATED[finite_kp]
    
    print(ALTITUDE)
    print(MAGEPHEM_TIME.shape)
    print(ALTITUDE.shape)
    print(LATITUDE.shape)
    print(LONGITUDE.shape)
    print(L.shape)
    
    
    queued_work = []
    num_processes = mp.cpu_count() - 10
    pool = mp.Pool(processes = num_processes)
    
    N = len(MAGEPHEM_TIME)
    CUT_SIZE = N // num_processes
    
    for p in range(num_processes):
        
        if p < num_processes - 1:
            queued_work.append((p * CUT_SIZE, (p+1) * CUT_SIZE))
        else:            
            queued_work.append((p * CUT_SIZE, N))
            

    chunks_to_process = []
        
    dates = np.array([datetime.datetime.isoformat(datetime.datetime.fromtimestamp(MAGEPHEM_TIME[t])) for t in range(len(MAGEPHEM_TIME))])
    
    for work in queued_work:
        
        chunk = (
            
            dates[work[0] : work[-1]], #TIME
            ALTITUDE[work[0] : work[-1]],
            LATITUDE[work[0] : work[-1]], #LATITUDE
            LONGITUDE[work[0] : work[-1]], #LONGITUDE
            KP_INTERPOLATED[work[0] : work[-1]] #KP INDEX FOR T89
        )
        
        chunks_to_process.append(chunk)
    
    t1 = time.perf_counter()
    
    results = pool.starmap(irbem.calculate_lstar_at_magnetic_equator_T89, chunks_to_process)
    
    t2 = time.perf_counter()
    
    print(f"Time taken to calculate whole year : {t2 - t1} seconds")
    
    pool.close() 
    pool.join()
            
    Lstar_calculated = np.abs(np.hstack(results))
    Lstar_calculated[(Lstar_calculated > 100)] = np.nan

    np.savez(file = os.path.abspath(f"./../processed_data/chorus_neural_network/STAGE_1/Lstar/RBSP_{sat.upper()}_T89_{year}.npz"), 
             UNIX_TIME = MAGEPHEM_TIME,
             Lstar = Lstar_calculated,
             L = L)
    
    