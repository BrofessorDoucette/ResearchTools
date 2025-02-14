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


def calculate_mag_coords_for_chunk(time, alt, lat, lon, kp):
    
    equator_model = IRBEM.MagFields(path = r".\..\IRBEM\libirbem.dll",
                            options = [0,0,0,0,0], 
                            kext = "T89",
                            sysaxes = 0, #GDZ
                            verbose = False)
    
    locations_of_equator = []
    
    for T in range(len(time)):
        
        sat_coords = {
            "dateTime" : time[T],
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
            
            "dateTime" : time[T : T + 25],
            "x1" : locations_of_equator[T : T + 25, 0].flatten(),
            "x2" : locations_of_equator[T : T + 25, 1].flatten(),
            "x3" : locations_of_equator[T : T + 25, 2].flatten()
            
        }
        
        mag_inputs = {
            "Kp" : kp[T : T + 25].flatten()
        }
        
        lstar_calculated.append(lstar_model.make_lstar(X = equator_coords, maginput = mag_inputs)["Lstar"])
        
        
    return locations_of_equator, np.hstack(lstar_calculated).flatten()


if __name__ == "__main__":
    
    
    year = 1998
    start_of_year = datetime.datetime(year = year, month = 1, day = 1)
    end_of_year = datetime.datetime(year = year + 1, month = 1, day = 1)
    equally_spaced_minutes = np.arange(start = start_of_year.timestamp(), stop = end_of_year.timestamp() + 60, step = 60)


    OMNI = data_loader.load_raw_data_from_config(id = ["OMNI", "ONE_HOUR_RESOLUTION"],
                                                start = start_of_year,
                                                end = end_of_year)

    OMNI_TIME = cdfepoch.unixtime(OMNI["Epoch"])
    KP = OMNI["KP"]

    invalid_omni_times = (OMNI_TIME < 0) | (KP < 0) | (KP >= 99)
    OMNI_TIME = OMNI_TIME[~invalid_omni_times]
    KP = KP[~invalid_omni_times]

    MPE = chorus_machine_learning_helper.load_MPE_year(year)

    print("Loaded the following satellites:")
    print([(s["SATID"], len(s["time"])) for s in MPE])
    
    
    data_processed = {}

    for SAT in MPE:
        
        '''unix_times_of_averages = []
        avg_geog_lat = []
        avg_geog_lon = []
        
        for MINUTE in equally_spaced_minutes:
            
            TIME_RANGE = np.searchsorted(a = SAT["UNIX_TIME"], v = [(MINUTE - 30), (MINUTE + 30)])
            
            if (TIME_RANGE[0] != TIME_RANGE[1]):
                
                unix_times_of_averages.append(MINUTE)    
                avg_geog_lat.append(np.nanmean(SAT["geogLat"][TIME_RANGE[0] : TIME_RANGE[1]]))
                y_of_lon = np.nanmean(np.sin(np.deg2rad(SAT["geogLon"][TIME_RANGE[0] : TIME_RANGE[1]])))
                x_of_lon = np.nanmean(np.cos(np.deg2rad(SAT["geogLon"][TIME_RANGE[0] : TIME_RANGE[1]])))            
                avg_geog_lon.append(np.mod((np.rad2deg(np.arctan2(y_of_lon, x_of_lon)) + 360), 360))
            
        unix_times_of_averages = np.array(unix_times_of_averages)
        avg_geog_lat = np.array(avg_geog_lat)
        avg_geog_lon = np.array(avg_geog_lon)
        
        unix_times_of_averages = unix_times_of_averages[np.isfinite(unix_times_of_averages)]
        avg_geog_lat = avg_geog_lat[np.isfinite(avg_geog_lat)]
        avg_geog_lon = avg_geog_lon[np.isfinite(avg_geog_lon)]
        
        big_distances = np.nonzero(((unix_times_of_averages[1:] - unix_times_of_averages[:-1]) > 60))[0] + 1'''
            
        KP_INTERPOLATED = np.interp(SAT["UNIX_TIME"], OMNI_TIME, KP, left=np.nan, right=np.nan).astype(np.float64)
            
        queued_work = []
        num_processes = mp.cpu_count() - 1
        pool = mp.Pool(processes = num_processes)
        
        N = len(SAT["UNIX_TIME"])
        CUT_SIZE = N // num_processes
        
        for p in range(num_processes):
            
            if p < num_processes - 1:
                queued_work.append((p * CUT_SIZE, (p+1) * CUT_SIZE))
            else:            
                queued_work.append((p * CUT_SIZE, N))

        #--------------------------------------------------

        print(queued_work)
        
        chunks_to_process = []
        
        dates = np.array([datetime.datetime.isoformat(datetime.datetime.fromtimestamp(SAT["UNIX_TIME"][t])) for t in range(len(SAT["UNIX_TIME"]))])
        
        for work in queued_work:
            
            chunk = (
                
                dates[work[0] : work[-1]], #TIME
                [817 for i in range(len(SAT["UNIX_TIME"][work[0] : work[-1]]))], #ALTITUDE
                SAT["geogLat"][work[0] : work[-1]], #LATITUDE
                SAT["geogLon"][work[0] : work[-1]], #LONGITUDE
                KP_INTERPOLATED[work[0] : work[-1]] #KP INDEX FOR T89
            )
            
            chunks_to_process.append(chunk)
                
        results = pool.starmap(calculate_mag_coords_for_chunk, chunks_to_process)
        
        pool.close() 
        pool.join()
        
        t2 = time.perf_counter()
        print(f"Time taken: {t2 - t1}")
        
        mag_equator_of_field_line_x_gsm = np.vstack([result[0] for result in results])
        lstar_calculated = np.hstack([result[1] for result in results])
        
        '''lstar_calculated = np.abs(np.hstack(Lstar_in_chunks))
        lm_calculated = np.abs(np.hstack(Lm_in_chunks))

        
        lstar_interpolated = np.zeros_like(SAT["UNIX_TIME"])
        lstar_interpolated[:] = np.nan
        
        if len(big_distances) > 0:
            
            print("There were big distances to deal with, probably should check the data!")
                    
            for m, d in enumerate(big_distances):
                
                if m == 0 :
                    
                    start_index = 0
                    end_index = d
                    
                else:
                    
                    start_index = big_distances[m - 1]
                    end_index = d
                    
                interpolated_between_big_distances = np.interp(SAT["UNIX_TIME"], unix_times_of_averages[start_index:end_index], lstar_calculated[start_index:end_index], left=np.nan, right=np.nan)
                non_nan_values = np.isfinite(interpolated_between_big_distances)
                lstar_interpolated[non_nan_values] = interpolated_between_big_distances[non_nan_values]
            
            #Get the last chunk
            
            start_index = big_distances[-1]
            
            interpolated_between_big_distances = np.interp(SAT["UNIX_TIME"], unix_times_of_averages[start_index:], lstar_calculated[start_index:], left=np.nan, right=np.nan)
            non_nan_values = np.isfinite(interpolated_between_big_distances)
            lstar_interpolated[non_nan_values] = interpolated_between_big_distances[non_nan_values]

        else:
            
            lstar_interpolated = np.interp(SAT["UNIX_TIME"], unix_times_of_averages, lstar_calculated, left=np.nan, right=np.nan)
        '''
        
        print(f"Finished processing data for : {SAT["SATID"]}")
        
        data_processed[SAT["SATID"]] = {"UNIX_TIME" : SAT["UNIX_TIME"],
                                        "BLC_Angle" : SAT["BLC_Angle"],
                                        "BLC_Flux" : SAT["BLC_Flux"],
                                        "MLT" : SAT["MLT"],
                                        "MAG_EQUATOR_OF_FIELD_LINE_IN_GSM" : mag_equator_of_field_line_x_gsm,
                                        "Lstar": lstar_calculated,
                                        "L" : SAT["lValue"],
                                        "geogLat" : SAT["geogLat"],
                                        "geogLon" : SAT["geogLon"]}

    output_dir = os.path.abspath(os.path.join("./../processed_data_chorus_neural_network/STAGE_0/MPE_DATA_PREPROCESSED_WITH_LSTAR"))
    os_helper.verify_output_dir_exists(output_dir, force_creation = True, hint="Output directory for L*")

    print(f"Saving data for : {year} to : {output_dir}")

    np.savez(file = os.path.abspath(os.path.join(output_dir, f"MPE_PREPROCESSED_DATA_T89_{year}_Test.npz")), DATA = data_processed)