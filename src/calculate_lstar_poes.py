from cdflib.epochs_astropy import CDFAstropy as cdfepoch
import astropy.time
import data_loader
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os_helper
import pandas as pd
import tqdm
import time
import multiprocessing as mp
import os

import IRBEM

import data_loader
import chorus_machine_learning_helper
import rbsp_chorus_tool


def calculate_mag_coords_for_chunk(dates, alt, lat, lon, kp):
    
    equator_model = IRBEM.MagFields(path = r".\..\IRBEM\libirbem.dll",
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
    
        
    lstar_model = IRBEM.MagFields(path = r".\..\IRBEM\libirbem.dll",
                            options = [1,0,0,0,0], 
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


if __name__ == "__main__":
    
    altitudes = { #IN KILOMETERS
        "n15" : 816,
        "n16" : 846.5,
        "n17" : 815,
        "n18" : 854,
        "n19" : 856,
        "m01" : 827,
        "m02" : 830,
        "m03" : 827
    }
    
    year = 2020
    start_of_year = datetime.datetime(year = year, month = 1, day = 1)
    end_of_year = datetime.datetime(year = year + 1, month = 1, day = 1)
    equally_spaced_half_minutes = np.arange(start = start_of_year.timestamp(), stop = end_of_year.timestamp() + 30, step = 30)


    OMNI = data_loader.load_raw_data_from_config(id = ["OMNI", "ONE_HOUR_RESOLUTION"],
                                                start = start_of_year,
                                                end = end_of_year)

    OMNI_TIME = cdfepoch.unixtime(OMNI["Epoch"])
    KP = OMNI["KP"].astype(np.float64)

    invalid_omni_times = (OMNI_TIME < 0) | (KP < 0) | (KP >= 99)
    KP[invalid_omni_times] = np.nan

    MPE = chorus_machine_learning_helper.load_MPE_year(year)

    print("Loaded the following satellites:")
    print([(s["SATID"], len(s["time"])) for s in MPE])
    
    
    data_processed = {}

    for SAT in MPE:
        
        t1 = time.perf_counter()
        
        KP_INTERPOLATED = np.interp(SAT["UNIX_TIME"], OMNI_TIME, KP, left=np.nan, right=np.nan).astype(np.float64)
        
        unix_times_of_averages = []
        avg_geog_lat = []
        avg_geog_lon = []
        avg_kp = []
        
        for HALF_MINUTE in equally_spaced_half_minutes:
            
            TIME_RANGE = np.searchsorted(a = SAT["UNIX_TIME"], v = [(HALF_MINUTE - 15), (HALF_MINUTE + 15)])
            
            if (TIME_RANGE[0] != TIME_RANGE[1]):
                
                unix_times_of_averages.append(HALF_MINUTE)    
                
                avg_geog_lat.append(np.nanmean(SAT["geogLat"][TIME_RANGE[0] : TIME_RANGE[1]]))
                
                y_of_lon = np.nanmean(np.sin(np.deg2rad(SAT["geogLon"][TIME_RANGE[0] : TIME_RANGE[1]])))
                x_of_lon = np.nanmean(np.cos(np.deg2rad(SAT["geogLon"][TIME_RANGE[0] : TIME_RANGE[1]])))            
                avg_geog_lon.append(np.mod((np.rad2deg(np.arctan2(y_of_lon, x_of_lon)) + 360), 360))
                
                avg_kp.append(np.nanmean(KP_INTERPOLATED[TIME_RANGE[0] : TIME_RANGE[1]]))
            
        unix_times_of_averages = np.array(unix_times_of_averages)
        avg_geog_lat = np.array(avg_geog_lat)
        avg_geog_lon = np.array(avg_geog_lon)
        avg_kp = np.array(avg_kp)
        
        all_finite = np.isfinite(unix_times_of_averages) & np.isfinite(avg_geog_lat) & np.isfinite(avg_geog_lon) & np.isfinite(avg_kp)
        
        unix_times_of_averages = unix_times_of_averages[all_finite]
        avg_geog_lat = avg_geog_lat[all_finite]
        avg_geog_lon = avg_geog_lon[all_finite]
        avg_kp = avg_kp[all_finite]
        
        big_distances = np.nonzero(((unix_times_of_averages[1:] - unix_times_of_averages[:-1]) > 60))[0] + 1
                        
        queued_work = []
        num_processes = mp.cpu_count() - 2
        pool = mp.Pool(processes = num_processes)
        
        N = len(unix_times_of_averages)
        CUT_SIZE = N // num_processes
        
        for p in range(num_processes):
            
            if p < num_processes - 1:
                queued_work.append((p * CUT_SIZE, (p+1) * CUT_SIZE))
            else:            
                queued_work.append((p * CUT_SIZE, N))

        #--------------------------------------------------

        print(queued_work)
        
        chunks_to_process = []
        
        dates = np.array([datetime.datetime.isoformat(datetime.datetime.fromtimestamp(unix_times_of_averages[t])) for t in range(len(unix_times_of_averages))])
        
        for work in queued_work:
            
            chunk = (
                
                dates[work[0] : work[-1]], #TIME
                [altitudes[SAT["SATID"]] for i in range(len(unix_times_of_averages[work[0] : work[-1]]))], #ALTITUDE
                avg_geog_lat[work[0] : work[-1]], #LATITUDE
                avg_geog_lon[work[0] : work[-1]], #LONGITUDE
                avg_kp[work[0] : work[-1]] #KP INDEX FOR T89
            )
            
            chunks_to_process.append(chunk)
                
        results = pool.starmap(calculate_mag_coords_for_chunk, chunks_to_process)
                
        pool.close() 
        pool.join()
                
        Lstar_calculated = np.abs(np.hstack(results))
        
        Lstar_calculated[(Lstar_calculated > 100)] = np.nan
        
        Lstar_interpolated = np.zeros_like(SAT["UNIX_TIME"])
        Lstar_interpolated[:] = np.nan
        
        if len(big_distances) > 0:
            
            print("There were big distances to deal with, probably should check the data!")
                    
            for m, d in enumerate(big_distances):
                
                if m == 0 :
                    
                    start_index = 0
                    end_index = d
                    
                else:
                    
                    start_index = big_distances[m - 1]
                    end_index = d
                    
                interpolated_between_big_distances = np.interp(SAT["UNIX_TIME"], unix_times_of_averages[start_index:end_index], Lstar_calculated[start_index:end_index], left=np.nan, right=np.nan)
                non_nan_values = np.isfinite(interpolated_between_big_distances)
                Lstar_interpolated[non_nan_values] = interpolated_between_big_distances[non_nan_values]
            
            #Get the last chunk
            
            start_index = big_distances[-1]
            
            interpolated_between_big_distances = np.interp(SAT["UNIX_TIME"], unix_times_of_averages[start_index:], Lstar_calculated[start_index:], left=np.nan, right=np.nan)
            non_nan_values = np.isfinite(interpolated_between_big_distances)
            Lstar_interpolated[non_nan_values] = interpolated_between_big_distances[non_nan_values]

        else:
            
            Lstar_interpolated = np.interp(SAT["UNIX_TIME"], unix_times_of_averages, Lstar_calculated, left=np.nan, right=np.nan)
        
        
        finite_time = np.isfinite(SAT["UNIX_TIME"])
        finite_blc_angle = np.isfinite(SAT["BLC_Angle"])
        finite_blc_flux = np.all(np.isfinite(SAT["BLC_Flux"][:, :8]), axis=1) & np.all((0 < SAT["BLC_Flux"][:, :8]), axis=1)
        finite_mlt = np.isfinite(SAT["MLT"])
        finite_Lstar = np.isfinite(Lstar_interpolated)
        finite_L = np.isfinite(SAT["lValue"])
        finite_geog_lat = np.isfinite(SAT["geogLat"])
        finite_geog_lon = np.isfinite(SAT["geogLon"])
        
        all_valid_data = finite_time & finite_blc_angle & finite_blc_flux & finite_mlt & finite_Lstar & finite_L & finite_geog_lat & finite_geog_lon
        
        print(f"Finished processing data for : {SAT["SATID"]}")
        t2 = time.perf_counter()
        print(f"Time taken: {t2 - t1} seconds.")
        
        data_processed[SAT["SATID"]] = {"UNIX_TIME" : SAT["UNIX_TIME"][all_valid_data],
                                        "BLC_Angle" : SAT["BLC_Angle"][all_valid_data],
                                        "BLC_Flux" : SAT["BLC_Flux"][all_valid_data],
                                        "MLT" : SAT["MLT"][all_valid_data],
                                        "Lstar": Lstar_interpolated[all_valid_data],
                                        "L" : SAT["lValue"][all_valid_data],
                                        "geogLat" : SAT["geogLat"][all_valid_data],
                                        "geogLon" : SAT["geogLon"][all_valid_data],
                                        "Alt" : altitudes[SAT["SATID"]]}

    output_dir = os.path.abspath(os.path.join("./../processed_data_chorus_neural_network/STAGE_0/MPE_DATA_PREPROCESSED_WITH_LSTAR"))
    os_helper.verify_output_dir_exists(output_dir, force_creation = True, hint="Output directory for L*")

    print(f"Saving data for : {year} to : {output_dir}")

    np.savez(file = os.path.abspath(os.path.join(output_dir, f"MPE_PREPROCESSED_DATA_T89_{year}.npz")), DATA = data_processed)