from spacepy.time import Ticktock
import glob
import os
import scipy.io
import numpy as np
import yaml


def clean_solar_proton_events(satellite: str,
                              month: int,
                              year: int,
                              make_dirs = False,
                              log_events: bool = False,
                              event_log_file: str = "./../compressed_data/POES_METOP/SOLAR_PROTON_EVENT_LOG.yaml",
                              dirty_data_dir ="./../compressed_data/POES_METOP/DIRTY/",
                              clean_data_dir ="./../compressed_data/POES_METOP/CLEAN/",
                              goes_data_dir ="./../raw_data/GOES/"):
        
    _year = str(year)
    _month = str(month)
    
    if len(_month) < 2:
        _month = f"0{_month}"
        
    goes_dir = os.path.join(goes_data_dir, f"{_year}/")
    goes_file_name = f"g15_epead_cpflux_5m_{_year}{_month}*.nc"
    goes_cdf_path_or_empty = glob.glob(goes_file_name, root_dir=goes_dir)
    
    if len(goes_cdf_path_or_empty) == 0:
        
        print(f"GOES CDF PATH NOT FOUND FOR {_year}/{_month}. No solar proton events to clean. Exiting!")
        return
        
    goes_cdf_path = os.path.join(goes_dir, goes_cdf_path_or_empty[0])

    goes = scipy.io.netcdf_file(goes_cdf_path, "r", mmap=False)
    zpgt10_e = goes.variables["ZPGT10E"][:]
    zpgt10_w = goes.variables["ZPGT10W"][:]
    goes_time = Ticktock(goes.variables["time_tag"][:] / 1000, "UNX")
    goes_epoch = goes_time.getUTC()
        
    solar_p_flux = (zpgt10_e + zpgt10_w) / 2
    over_ten_mev = np.argwhere(solar_p_flux > 10)
    
    a = np.append(over_ten_mev, np.NAN)
    b = np.append(np.NAN, over_ten_mev)
    breaks = np.argwhere((a - b) != 1).flatten().astype(int)        
    start_pts = a[breaks][:-1].astype(int)
    end_pts = a[breaks - 1][1:].astype(int)
    start_times = goes_epoch[start_pts]  #These are the start and end times of period where solar proton events occurred
    end_times = goes_epoch[end_pts]

    if log_events and (not os.path.exists(event_log_file)) and (not make_dirs):
        print("Exiting! Events were set to be logged, but log file doesn't exist and make_dirs == False. Either set make_dirs to True or create the log file to log events.")
        return
    
    if log_events and (not os.path.exists(event_log_file)) and make_dirs:
        
        with open(event_log_file, "w") as f:
            
            yaml.safe_dump({"events": []}, f)
            pass
        
    if log_events and os.path.exists(event_log_file):
        
        with open(event_log_file, "r") as f:
            
            logged_events = yaml.safe_load(f)
            
        for i in range(len(start_times)):
                
            logged_events["events"].append({"begin": f"{start_times[i]:%Y-%m-%d %H:%M:%S%z}",
                                            "end": f"{end_times[i]:%Y-%m-%d %H:%M:%S%z}"})
        
        with open(event_log_file, "w") as f:
            
            yaml.safe_dump(logged_events, f)
    
    input_data_dir = os.path.join(dirty_data_dir, f"{_year}/{_month}/")
    
    if not os.path.isdir(input_data_dir):
        raise Exception(f"Input data folder does not exist! Tried: {input_data_dir}")
    
    output_data_dir = os.path.join(clean_data_dir, f"{_year}/{_month}/")
            
    if not os.path.isdir(output_data_dir):
        
        if make_dirs:
            
            os.makedirs(output_data_dir)
        else:
            
            raise Exception(f"Output data folder does not exist and make_dirs != True. Tried: {output_data_dir}")
    
    input_file_name = f"POES_METOP_{_year}{_month}_{satellite.lower()}_DIRTY.npz"
    input_data_path = os.path.join(input_data_dir, input_file_name)
        
    if not os.path.exists(input_data_path):
        raise Exception(f"\nData file not found: {input_data_path}")
        
    print(f"Loading : {input_data_path}")
    input_data = np.load(input_data_path, allow_pickle=True)
    
    dirty_epoch = input_data["EPOCH"]
    dirty_mep_ele_flux = input_data["MEP_ELE_FLUX"]
    dirty_L = input_data["L"]
    dirty_mlt = input_data["MLT"]
    
    input_data.close()
    
    for start, end in zip(start_times, end_times):
        
        exclude = np.argwhere((start <= dirty_epoch) & (dirty_epoch <= end))
                
        dirty_epoch = np.delete(dirty_epoch, exclude, axis=0)
        dirty_mep_ele_flux = np.delete(dirty_mep_ele_flux, exclude, axis=0)
        dirty_L = np.delete(dirty_L, exclude, axis=0)
        dirty_mlt = np.delete(dirty_mlt, exclude, axis=0)
        
        print(f"Due to solar proton events, data was removed between: {start} and {end}!")
    
    #Dirty variables are cleaned at this point!
    
    output_file_name = f"POES_METOP_{_year}{_month}_{satellite.lower()}_CLEAN.npz"
    output_data_path = os.path.join(output_data_dir, output_file_name)
    
    j_40 = dirty_mep_ele_flux[:, 0, 0]
    j_130 = dirty_mep_ele_flux[:, 0, 1]
    
    J = j_40 - j_130
    
    P = 100
    naive_chorus_intensity = J / (P * ((dirty_L - 3)**2 + 0.03))
    naive_chorus_amplitudes = np.sqrt(naive_chorus_intensity)
        
    print(f"Saving : {output_data_path}")
    
    np.savez_compressed(output_data_path,
                        EPOCH = dirty_epoch,
                        MEP_ELE_FLUX = dirty_mep_ele_flux,
                        L = dirty_L,
                        MLT = dirty_mlt,
                        NAIVE_CHORUS_AMPLITUDES = naive_chorus_amplitudes)
