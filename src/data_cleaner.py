from spacepy.time import Ticktock
import glob
import os
import os_helper
import scipy.io
import numpy as np
import yaml
import re


def clean_solar_proton_events_after_2012(satellite: str,
                                         make_dirs: bool = False,
                                         log_events: bool = False,
                                         event_log_file_dir: str = "./../compressed_data/POES/",
                                         dirty_data_dir: str = "./../compressed_data/POES/DIRTY/",
                                         clean_data_dir: str = "./../compressed_data/POES/CLEAN/",
                                         goes_data_dir: str = "./../raw_data/GOES/",
                                         debug_mode : bool = False) -> None:

    event_log_file = os.path.join(os.path.abspath(event_log_file_dir), f"SOLAR_PROTON_EVENTS_REMOVED_{satellite}.yaml")

    if satellite not in ["metop01", "metop02", "metop03", "noaa15", "noaa16", "noaa17", "noaa18", "noaa19"]:

        raise Exception("The compressor for satellites without SEM-2 Instrument Package is not yet implemented!")

    input_poes_dir = os.path.join(os.path.abspath(dirty_data_dir), satellite)

    os_helper.verify_input_dir_exists(input_poes_dir, hint=f"POES COMPRESSED DIRTY DIR: {satellite}")

    output_poes_dir = os.path.join(os.path.abspath(clean_data_dir), satellite)

    os_helper.verify_output_dir_exists(directory=output_poes_dir,
                                       force_creation=make_dirs,
                                       hint="CLEAN POES DIR")

    input_poes_files_or_null = glob.glob(pathname = "*.npz", root_dir = input_poes_dir)

    if len(input_poes_files_or_null) == 0:
        raise Exception(f"Found no raw input files to clean solar proton events from for: {satellite}")

    for input_poes_file in input_poes_files_or_null:

        re_match = re.search(pattern=r"([0-9]{4})", string=input_poes_file)

        if not re_match:
            continue

        date = re_match.group()

        _year = date[:4]

        ##REMOVE THIS LATER WHEN WE GET MORE GOES DATA FILES FOR CLEANING##
        if(int(_year) >= 2020):
            continue

        goes_dir = os.path.join(goes_data_dir, f"{_year}/")

        os_helper.verify_input_dir_exists(directory = goes_dir,
                                          hint = "GOES DATA DIR")

        goes_file_glob = f"g15_epead_cpflux_5m_{_year}*.nc"
        goes_file_names_or_empty = glob.glob(goes_file_glob, root_dir=goes_dir)

        if len(goes_file_names_or_empty) == 0:
            if debug_mode:
                print(f"GOES CDF PATH NOT FOUND FOR {_year}. Therefore, no solar proton events can be cleaned!")
                print("Continuing to next file...!")
            continue
        
        goes_epoch = np.zeros(shape=(0))
        zpgt10_e = np.zeros(shape=(0))
        zpgt10_w = np.zeros(shape=(0))
        
        for goes_file_name in goes_file_names_or_empty:
            
            goes_nc_path = os.path.join(goes_dir, goes_file_name)

            with scipy.io.netcdf_file(goes_nc_path, "r", mmap=False) as goes:
                zpgt10_e = np.concatenate((zpgt10_e, goes.variables["ZPGT10E"][:]), axis = 0)
                zpgt10_w = np.concatenate((zpgt10_w, goes.variables["ZPGT10W"][:]), axis = 0)
                goes_time = Ticktock(goes.variables["time_tag"][:] / 1000, "UNX")
                goes_epoch = np.concatenate((goes_epoch, goes_time.UTC), axis = 0)
        
        

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
            if debug_mode:
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

        input_data_path = os.path.join(input_poes_dir, input_poes_file)

        if not os.path.exists(input_data_path):
            raise Exception(f"\nData file not found: {input_data_path}")

        print(f"Loading : {input_data_path}")
        input_data = np.load(input_data_path, allow_pickle=True)

        #Time
        dirty_epoch = input_data["EPOCH"]
        #Coordinates
        dirty_alt = input_data["ALT"]
        dirty_lat = input_data["LAT"]
        dirty_lon = input_data["LON"]
        dirty_L = input_data["L"]
        dirty_mlt = input_data["MLT"]
        #Flux
        dirty_mep_ele_tel0_flux_e1 = input_data["MEP_ELE_TEL0_FLUX_E1"]
        dirty_mep_ele_tel0_flux_e2 = input_data["MEP_ELE_TEL0_FLUX_E2"]
        #Pitch angles
        dirty_meped_alpha_0_sat = input_data["MEPED_ALPHA_0_SAT"]
        
        input_data.close()

        print(f"Cleaning : {input_data_path}")
        for start, end in zip(start_times, end_times):

            exclude = np.argwhere((start <= dirty_epoch) & (dirty_epoch <= end))

            dirty_epoch = np.delete(dirty_epoch, exclude, axis=0)
            dirty_alt = np.delete(dirty_alt, exclude, axis = 0)
            dirty_lat = np.delete(dirty_lat, exclude, axis = 0)
            dirty_lon = np.delete(dirty_lon, exclude, axis = 0)            
            dirty_L = np.delete(dirty_L, exclude, axis=0)
            dirty_mlt = np.delete(dirty_mlt, exclude, axis=0)
            dirty_mep_ele_tel0_flux_e1 = np.delete(dirty_mep_ele_tel0_flux_e1, exclude, axis = 0)
            dirty_mep_ele_tel0_flux_e2 = np.delete(dirty_mep_ele_tel0_flux_e2, exclude, axis = 0)
            dirty_meped_alpha_0_sat = np.delete(dirty_meped_alpha_0_sat, exclude, axis = 0)

            print(f"Due to solar proton events, data was removed between: {start} and {end}!")

        #Dirty variables are cleaned at this point!

        output_file_name = f"POES_{_year}_{satellite.lower()}_CLEAN.npz"
        output_data_path = os.path.join(output_poes_dir, output_file_name)

        print(f"Saving : {output_data_path}")

        np.savez_compressed(output_data_path,
                            EPOCH = dirty_epoch,
                            ALT = dirty_alt,
                            LAT = dirty_lat,
                            LON = dirty_lon,
                            L = dirty_L,
                            MLT = dirty_mlt,
                            MEP_ELE_TEL0_FLUX_E1 = dirty_mep_ele_tel0_flux_e1,
                            MEP_ELE_TEL0_FLUX_E2 = dirty_mep_ele_tel0_flux_e2,
                            MEPED_ALPHA_0_SAT = dirty_meped_alpha_0_sat)

if __name__ == "__main__":

    _dirty_data_dir: str = os.path.abspath("./../compressed_data/POES/DIRTY/")
    satellites = []
    for x in os.scandir(_dirty_data_dir):
        if x.is_dir():
            satellites.append(x.name)

    for satellite in satellites:

        clean_solar_proton_events_after_2012(satellite=satellite,
                                             make_dirs=True,
                                             log_events=True,
                                             dirty_data_dir = _dirty_data_dir,
                                             debug_mode=True)
