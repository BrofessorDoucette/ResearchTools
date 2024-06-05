from spacepy.time import Ticktock
import glob
import os
import os_helper
import scipy.io
import numpy as np
import yaml
import re


def clean_solar_proton_events(satellite: str,
                              make_dirs: bool = False,
                              log_events: bool = False,
                              event_log_file_dir: str = "./../compressed_data/POES/",
                              dirty_data_dir: str = "./../compressed_data/POES/DIRTY/",
                              clean_data_dir: str = "./../compressed_data/POES/CLEAN/",
                              goes_data_dir: str = "./../raw_data/GOES/",
                              debug_mode : bool = False) -> None:

    event_log_file = os.path.join(os.path.abspath(event_log_file_dir), f"SOLAR_PROTON_EVENTS_REMOVED_{satellite}.yaml")

    if satellite not in ["metop1", "metop2", "noaa15", "noaa16", "noaa17", "noaa18", "noaa19"]:

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

        re_match = re.search(pattern=r"([0-9]{6})", string=input_poes_file)

        if not re_match:
            continue

        date = re_match.group()

        _year = date[:4]
        _month = date[4:]

        ##REMOVE THIS LATER WHEN WE GET MORE GOES DATA FILES FOR CLEANING##
        if(int(_year) >= 2020):
            continue

        goes_dir = os.path.join(goes_data_dir, f"{_year}/")

        os_helper.verify_input_dir_exists(directory = goes_dir,
                                          hint = "GOES DATA DIR")

        goes_file_name = f"g15_epead_cpflux_5m_{_year}{_month}*.nc"
        goes_cdf_path_or_empty = glob.glob(goes_file_name, root_dir=goes_dir)

        if len(goes_cdf_path_or_empty) == 0:
            if debug_mode:
                print(f"GOES CDF PATH NOT FOUND FOR {_year}/{_month}. Therefore, no solar proton events can be cleaned!")
                print("Continuing to next file...!")
            continue

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

        output_file_name = f"POES_{_year}{_month}_{satellite.lower()}_CLEAN.npz"
        output_data_path = os.path.join(output_poes_dir, output_file_name)

        j_40 = dirty_mep_ele_flux[:, 0, 0]
        j_130 = dirty_mep_ele_flux[:, 0, 1]

        J = j_40 - j_130
        # For some stupid reason this subtraction rarely creates negatives even though the channels are integral... :(
        J[J < 0] = np.NaN

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

if __name__ == "__main__":

    _dirty_data_dir: str = os.path.abspath("./../compressed_data/POES/DIRTY/")
    satellites = []
    for x in os.scandir(_dirty_data_dir):
        if x.is_dir():
            satellites.append(x.name)

    for satellite in satellites:

        clean_solar_proton_events(satellite=satellite,
                                  make_dirs=True,
                                  log_events=True,
                                  dirty_data_dir = _dirty_data_dir,
                                  debug_mode=True)
