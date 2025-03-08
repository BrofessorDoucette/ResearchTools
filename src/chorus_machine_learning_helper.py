import data_loader
import datetime
from cdflib.epochs_astropy import CDFAstropy as cdfepoch
import numpy as np
import astropy.time
import pandas as pd
import tqdm

def load_MPE_year(year : int) -> list:
     
     
    POES = []

    for SAT in ["m01", "m02", "m03", "n15", "n16", "n17", "n18", "n19"]:
            
            POES_sat_refs = data_loader.load_raw_data_from_config(id=["POES", "SEM", "MPE"],
                                                                satellite=SAT,
                                                                start = datetime.datetime(year = year, month = 1, day = 1, hour = 0, minute = 0, second = 0),
                                                                end = datetime.datetime(year = year, month = 12, day = 31, hour = 23, minute = 59, second = 59))
            
            if POES_sat_refs:
                
                #This was done cause I wanted to scale the MLT before cleaning but Im lazy
                if year < 2014:
                    POES_sat_refs["MLT"] = (POES_sat_refs["MLT"] / 360.0) * 24.0
                
                valid_times = np.isfinite(POES_sat_refs["time"]) & (0 < POES_sat_refs["time"])
                valid_BLC_Angle = np.isfinite(POES_sat_refs["BLC_Angle"]) & (0 < POES_sat_refs["BLC_Angle"])
                valid_BLC_Flux = np.all(np.isfinite(POES_sat_refs["BLC_Flux"][:, :8]), axis=1) & np.all((0 < POES_sat_refs["BLC_Flux"][:, :8]), axis=1)
                valid_MLT = np.isfinite(POES_sat_refs["MLT"]) & (0 < POES_sat_refs["MLT"]) & (POES_sat_refs["MLT"] < 24)
                valid_L = np.isfinite(POES_sat_refs["lValue"]) & (0 < POES_sat_refs["lValue"]) & (POES_sat_refs["lValue"] < 10)
                valid_geogLat = np.isfinite(POES_sat_refs["geogLat"]) & (-90.0 <= POES_sat_refs["geogLat"]) & (POES_sat_refs["geogLat"] <= 90.0)
                valid_geogLon = np.isfinite(POES_sat_refs["geogLon"]) & (0.0 <= POES_sat_refs["geogLon"]) & (POES_sat_refs["geogLon"] <= 360.0)

                valid_points = valid_times & valid_BLC_Angle & valid_BLC_Flux & valid_MLT & valid_L & valid_geogLat & valid_geogLon
                
                if np.any(valid_points):
                    
                    POES_sat_refs["time"] = POES_sat_refs["time"][valid_points]
                    POES_sat_refs["BLC_Angle"] = POES_sat_refs["BLC_Angle"][valid_points]
                    POES_sat_refs["BLC_Flux"] = POES_sat_refs["BLC_Flux"][valid_points, :8]
                    POES_sat_refs["MLT"] = POES_sat_refs["MLT"][valid_points]
                    POES_sat_refs["lValue"] = POES_sat_refs["lValue"][valid_points]
                    POES_sat_refs["geogLat"] = POES_sat_refs["geogLat"][valid_points]
                    POES_sat_refs["geogLon"] = POES_sat_refs["geogLon"][valid_points]
                
                    if year < 2014:
                                        
                        POES_sat_refs["UNIX_TIME"] = cdfepoch.unixtime(POES_sat_refs["time"])
                    else:
                        POES_sat_refs["UNIX_TIME"] = (POES_sat_refs["time"] / 1000)
                        
                    #Sort them so assumptions for binary search are satisfied:
                    order = np.argsort(POES_sat_refs["UNIX_TIME"])
                    POES_sat_refs["time"] = POES_sat_refs["time"][order]
                    POES_sat_refs["UNIX_TIME"] = POES_sat_refs["UNIX_TIME"][order]
                    POES_sat_refs["BLC_Angle"] = POES_sat_refs["BLC_Angle"][order]
                    POES_sat_refs["BLC_Flux"] = POES_sat_refs["BLC_Flux"][order, :]
                    POES_sat_refs["MLT"] = POES_sat_refs["MLT"][order]
                    POES_sat_refs["lValue"] = POES_sat_refs["lValue"][order]
                    POES_sat_refs["geogLat"] = POES_sat_refs["geogLat"][order]
                    POES_sat_refs["geogLon"] = POES_sat_refs["geogLon"][order]
                    POES_sat_refs["SATID"] = SAT
                    POES.append(POES_sat_refs)
            
        
    if not POES:
        print(f"No POES satellite coverage found for year : {year}")
        

    print(f"Finished loading POES data for year : {year}")
    
    return POES

def load_SUPERMAG_SME_year(year : int):

    print(f"Began loading SUPERMAG data for year : {year}")
    SUPERMAG_df = pd.read_csv(fr"./../chorus_neural_network/SUPERMAG_SME/sme_{year}.csv")
    SUPERMAG = {}

    valid_SME = np.isfinite(SUPERMAG_df["SME"]) & (0 < SUPERMAG_df["SME"])

    if not np.any(valid_SME):
        print(f"No valid SME for year : {year}")

    SUPERMAG["SME"] = np.array(SUPERMAG_df["SME"][valid_SME])
    SUPERMAG["Date_UTC"] = np.array(SUPERMAG_df["Date_UTC"][valid_SME])
    SUPERMAG["UNIX_TIME"] = astropy.time.Time(SUPERMAG["Date_UTC"].astype(str), scale="utc", in_subfmt='date_hms').unix

    start_interpolation_time = datetime.datetime(year = year, month = 1, day = 1).timestamp()
    end_interpolation_time = datetime.datetime(year = year + 1, month = 1, day = 1).timestamp()
    evenly_spaced_seconds = np.arange(start = start_interpolation_time,
                                      stop = end_interpolation_time + 1,
                                      step = 1)
    
    SUPERMAG["SME"] = np.interp(x = evenly_spaced_seconds, 
                                xp = SUPERMAG["UNIX_TIME"], 
                                fp = SUPERMAG["SME"])

    order = np.argsort(evenly_spaced_seconds)
    SUPERMAG["SME"] = SUPERMAG["SME"][order]
    SUPERMAG["UNIX_TIME"] = evenly_spaced_seconds[order]
    del SUPERMAG["Date_UTC"]
    print(f"Finished loading SUPERMAG data for year : {year}")
    
    return SUPERMAG


def load_OMNI_year(year : int) -> dict:

    print(f"Began loading OMNI data for year : {year}")
    OMNI_refs = data_loader.load_raw_data_from_config(id = ["OMNI", "ONE_MIN_RESOLUTION"], 
                                                    start = datetime.datetime(year = year, month = 1, day = 1, hour = 0, minute = 0, second = 0),
                                                    end = datetime.datetime(year = year, month = 12, day = 31, hour = 23, minute = 59, second = 59),
                                                    root_data_dir = "./../raw_data/")
    OMNI = {}

    valid_times = np.isfinite(OMNI_refs["Epoch"]) & (0 < OMNI_refs["Epoch"])
    
    OMNI["UNIX_TIME"] = cdfepoch.unixtime(OMNI_refs["Epoch"][valid_times])
    OMNI["AVG_B"] = OMNI_refs["F"][valid_times]
    OMNI["FLOW_SPEED"] = OMNI_refs["flow_speed"][valid_times]
    OMNI["PROTON_DENSITY"] = OMNI_refs["proton_density"][valid_times]
    OMNI["SYM_H"] = OMNI_refs["SYM_H"][valid_times]
    
    valid_AVG_B = np.isfinite(OMNI["AVG_B"]) & (0 <= OMNI["AVG_B"]) & (OMNI["AVG_B"] < 9990)
    valid_FLOW_SPEED = np.isfinite(OMNI["FLOW_SPEED"]) & (0 <= OMNI["FLOW_SPEED"]) & (OMNI["FLOW_SPEED"] < 99900)
    valid_PROTON_DENSITY = np.isfinite(OMNI["PROTON_DENSITY"]) & (-900 <= OMNI["PROTON_DENSITY"]) & (OMNI["PROTON_DENSITY"] < 900)
    valid_SYM_H = np.isfinite(OMNI["SYM_H"]) & (-99000 <= OMNI["SYM_H"]) & (OMNI["SYM_H"] < 99900)
    valid_points = valid_AVG_B & valid_FLOW_SPEED & valid_PROTON_DENSITY & valid_SYM_H

    if(not np.any(valid_points)):
        print(f"No valid OMNI DATA for year : {year}")
        print(f"SKIPPING YEAR : {year}")

    OMNI["AVG_B"][~valid_points] = np.nan
    OMNI["FLOW_SPEED"][~valid_points] = np.nan
    OMNI["PROTON_DENSITY"][~valid_points] = np.nan
    OMNI["SYM_H"] = OMNI["SYM_H"].astype(np.float32)
    OMNI["SYM_H"][~valid_points] = np.nan
    
    if (np.max(OMNI["UNIX_TIME"][1:] - OMNI["UNIX_TIME"][:-1])) > 300:
        raise Exception("Tried to interpolate OMNI data but large gaps that are unexpected were present!")
    
    start_interpolation_time = datetime.datetime(year = year, month = 1, day = 1).timestamp()
    end_interpolation_time = datetime.datetime(year = year + 1, month = 1, day = 1).timestamp()
    evenly_spaced_seconds = np.arange(start = start_interpolation_time,
                                      stop = end_interpolation_time + 1,
                                      step = 1)
    
    OMNI["AVG_B"] = np.interp(x = evenly_spaced_seconds, 
                                xp = OMNI["UNIX_TIME"], 
                                fp = OMNI["AVG_B"])
    OMNI["FLOW_SPEED"] = np.interp(x = evenly_spaced_seconds, 
                                xp = OMNI["UNIX_TIME"], 
                                fp = OMNI["FLOW_SPEED"])
    OMNI["PROTON_DENSITY"] = np.interp(x = evenly_spaced_seconds, 
                                xp = OMNI["UNIX_TIME"], 
                                fp = OMNI["PROTON_DENSITY"])
    OMNI["SYM_H"] = np.interp(x = evenly_spaced_seconds, 
                                xp = OMNI["UNIX_TIME"], 
                                fp = OMNI["SYM_H"])
    
    order = np.argsort(evenly_spaced_seconds)
    OMNI["UNIX_TIME"] = evenly_spaced_seconds[order]
    OMNI["AVG_B"] = OMNI["AVG_B"][order]
    OMNI["FLOW_SPEED"] = OMNI["FLOW_SPEED"][order]
    OMNI["PROTON_DENSITY"] = OMNI["PROTON_DENSITY"][order]
    OMNI["SYM_H"] = OMNI["SYM_H"][order]

    not_nan = np.isfinite(OMNI["AVG_B"]) & np.isfinite(OMNI["FLOW_SPEED"]) & np.isfinite(OMNI["PROTON_DENSITY"]) & np.isfinite(OMNI["SYM_H"])
    OMNI["UNIX_TIME"] = OMNI["UNIX_TIME"][not_nan]
    OMNI["AVG_B"] = OMNI["AVG_B"][not_nan]
    OMNI["FLOW_SPEED"] = OMNI["FLOW_SPEED"][not_nan]
    OMNI["PROTON_DENSITY"] = OMNI["PROTON_DENSITY"][not_nan]
    OMNI["SYM_H"] = OMNI["SYM_H"][not_nan]

    print(f"Finished loading OMNI data for year : {year}")
    
    return OMNI

def find_average_SUPERMAG_and_OMNI_values_for_each_POES_data_point(POES : list,
                                                                   SUPERMAG : dict,
                                                                   OMNI : dict) -> dict:
    POES_UNIX_TIMES = []
    POES_L_VALUES = []
    POES_MLT_VALUES = []
    POES_FLUX_SPECTRUM = []

    for P in POES:
                
        POES_UNIX_TIMES.append(P["UNIX_TIME"])
        POES_L_VALUES.append(P["lValue"])
        POES_MLT_VALUES.append(P["MLT"])
        POES_FLUX_SPECTRUM.append(P["BLC_Flux"])
        
    POES_UNIX_TIMES = np.hstack(POES_UNIX_TIMES)
    POES_L_VALUES = np.hstack(POES_L_VALUES)
    POES_MLT_VALUES = np.hstack(POES_MLT_VALUES)
    POES_FLUX_SPECTRUM = np.vstack(POES_FLUX_SPECTRUM)

    POES_TIMES_OF_FEATURES = []
    L_FEATURES = []
    MLT_FEATURES = []
    FLUX_SPECTRUM_FEATURES = []
    SME_FEATURES = []
    B_FEATURES = []
    FLOW_SPEED_FEATURES = []
    PROTON_DENSITY_FEATURES = []
    SYM_H_FEATURES = []

    for idx, T in tqdm.tqdm(enumerate(POES_UNIX_TIMES)):
        
        TIME_RANGE = np.searchsorted(a = SUPERMAG["UNIX_TIME"], v = [(T - 60), (T + 60)])
        AVG_SME = np.nanmean(SUPERMAG["SME"][TIME_RANGE[0]:TIME_RANGE[1]])

        TIME_RANGE = np.searchsorted(a = OMNI["UNIX_TIME"], v = [(T - 60), (T + 60)])
        AVG_AVG_B = np.nanmean(OMNI["AVG_B"][TIME_RANGE[0]:TIME_RANGE[1]])
        AVG_FLOW_SPEED = np.nanmean(OMNI["FLOW_SPEED"][TIME_RANGE[0]:TIME_RANGE[1]])
        AVG_PROTON_DENSITY = np.nanmean(OMNI["PROTON_DENSITY"][TIME_RANGE[0]:TIME_RANGE[1]])
        AVG_SYM_H = np.nanmean(OMNI["SYM_H"][TIME_RANGE[0]:TIME_RANGE[1]])
        
        if np.isfinite(AVG_SME) & np.isfinite(AVG_AVG_B) & np.isfinite(AVG_FLOW_SPEED) & np.isfinite(AVG_PROTON_DENSITY) & np.isfinite(AVG_SYM_H):
            
            POES_TIMES_OF_FEATURES.append(POES_UNIX_TIMES[idx])
            L_FEATURES.append(POES_L_VALUES[idx])
            MLT_FEATURES.append(POES_MLT_VALUES[idx])
            FLUX_SPECTRUM_FEATURES.append(POES_FLUX_SPECTRUM[idx, :])
            SME_FEATURES.append(AVG_SME)
            B_FEATURES.append(AVG_AVG_B)
            FLOW_SPEED_FEATURES.append(AVG_FLOW_SPEED)
            PROTON_DENSITY_FEATURES.append(AVG_PROTON_DENSITY)
            SYM_H_FEATURES.append(AVG_SYM_H)
            
    refs = {
        
        "POES_TIMES_OF_FEATURES" : np.asarray(POES_TIMES_OF_FEATURES),
        "L_FEATURES" : np.expand_dims(np.array(L_FEATURES), axis = 1),
        "MLT_FEATURES" : np.expand_dims(np.array(MLT_FEATURES), axis = 1),
        "FLUX_SPECTRUM_FEATURES" : np.vstack(np.expand_dims(FLUX_SPECTRUM_FEATURES, axis = 0)),
        "SME_FEATURES" : np.expand_dims(np.array(SME_FEATURES), axis = 1),
        "B_FEATURES" : np.expand_dims(np.array(B_FEATURES), axis = 1),
        "FLOW_SPEED_FEATURES" : np.expand_dims(np.array(FLOW_SPEED_FEATURES), axis = 1),
        "PROTON_DENSITY_FEATURES" : np.expand_dims(np.array(PROTON_DENSITY_FEATURES), axis = 1),
        "SYM_H_FEATURES" : np.expand_dims(np.array(SYM_H_FEATURES), axis = 1)
    }
    
    return refs

def normalize_features(FEATURE_REFS : dict, version : str):
    
    MODEL_TRAINING_DATASET = np.load(f"./../chorus_neural_network/STAGE_4/{version}/MODEL_READY_DATA_{version}.npz")

    MEAN_FLUX = MODEL_TRAINING_DATASET["MEAN_FLUXES"]
    STD_FLUX = MODEL_TRAINING_DATASET["STD_FLUXES"]
    MEAN_SME = MODEL_TRAINING_DATASET["MEAN_SME"]
    STD_SME = MODEL_TRAINING_DATASET["STD_SME"]
    MEAN_AVG_B = MODEL_TRAINING_DATASET["MEAN_AVG_B"]
    STD_AVG_B = MODEL_TRAINING_DATASET["STD_AVG_B"]
    MEAN_FLOW_SPEED = MODEL_TRAINING_DATASET["MEAN_FLOW_SPEED"]
    STD_FLOW_SPEED = MODEL_TRAINING_DATASET["STD_FLOW_SPEED"]
    MEAN_AVG_PROTON_DENSITY = MODEL_TRAINING_DATASET["MEAN_AVG_PROTON_DENSITY"]
    STD_AVG_PROTON_DENSITY = MODEL_TRAINING_DATASET["STD_AVG_PROTON_DENSITY"]
    MEAN_SYM_H = MODEL_TRAINING_DATASET["MEAN_AVG_SYM_H"]
    STD_SYM_H = MODEL_TRAINING_DATASET["STD_AVG_SYM_H"]

    MODEL_TRAINING_DATASET.close()
    
    L_FEATURES_POST_PROCESSING = FEATURE_REFS["L_FEATURES"]
    MLT_FEATURES_POST_PROCESSING_1 = np.sin((FEATURE_REFS["MLT_FEATURES"] * 2 * np.pi) / 24.0)
    MLT_FEATURES_POST_PROCESSING_2 = np.cos((FEATURE_REFS["MLT_FEATURES"] * 2 * np.pi) / 24.0)
    FLUX_SPECTRUM_FEATURES_POST_PROCESSING = (np.log(FEATURE_REFS["FLUX_SPECTRUM_FEATURES"][:, 1:3]) - MEAN_FLUX) / STD_FLUX
    SME_FEATURES_POST_PROCESSING = (FEATURE_REFS["SME_FEATURES"] - MEAN_SME) / STD_SME
    FLOW_SPEED_FEATURES_POST_PROCESSING = (FEATURE_REFS["FLOW_SPEED_FEATURES"] - MEAN_FLOW_SPEED) / STD_FLOW_SPEED
    SYM_H_POST_PROCESSING = (FEATURE_REFS["SYM_H_FEATURES"] - MEAN_SYM_H) / STD_SYM_H


    FEATURES_POST_PROCESSING = np.hstack([L_FEATURES_POST_PROCESSING, 
                                          MLT_FEATURES_POST_PROCESSING_1,
                                          MLT_FEATURES_POST_PROCESSING_2,
                                          FLUX_SPECTRUM_FEATURES_POST_PROCESSING,
                                          SME_FEATURES_POST_PROCESSING,
                                          FLOW_SPEED_FEATURES_POST_PROCESSING,
                                          SYM_H_POST_PROCESSING])
    
    return FEATURES_POST_PROCESSING