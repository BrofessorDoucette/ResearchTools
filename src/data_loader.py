from dateutil import rrule
import datetime
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
import os_helper
from spacepy import pycdf
from field_models import model
import glob
import global_configuration
import date_helper
import cdflib
from netCDF4 import Dataset
import h5py
import warnings

def get_file_names_between_start_and_end(start : datetime.datetime,
                                         end : datetime.datetime,
                                         file_glob: str = "",
                                         input_dir_structure : str = "",
                                         debug : bool = False,
                                         verbose : bool = False) -> list[str]:
    
    #Maybe this code should be adapted in the future to work for more cases, but currently it works for all needed instruments as of 08/29/2024
    
    if not file_glob:
        raise Exception("Tried to load cdfs, but no file glob was given. I have no idea what the file names look like!")
    
    if not input_dir_structure:
        raise Exception("Tried to load cdfs but no input_dir was given. I have no idea where to look for the requested file names!")
    
    if debug:
        print(f"Start date: {start}")
        print(f"End date: {end}")
        print(f"Unprocessed file_glob : {file_glob}")
        print(f"Unprocessed input_dir_structure : {input_dir_structure}")

    paths = []
    
    if ("{$DAY}" in file_glob) or ("{$DAY}" in input_dir_structure):
        
        for dt in rrule.rrule(freq = rrule.DAILY, dtstart = start, until = end):
            
            formatted_file_glob = global_configuration.replace_all_keys_in_string_with_values(file_glob, {"{$YEAR}" : dt.year,
                                                                                                "{$MONTH}" : date_helper.month_str_from_int(dt.month),
                                                                                                "{$DAY}" : date_helper.day_str_from_int(dt.day)})
            
            input_dir = global_configuration.replace_all_keys_in_string_with_values(input_dir_structure, {"{$YEAR}" : dt.year,
                                                                                                          "{$MONTH}" : date_helper.month_str_from_int(dt.month),
                                                                                                          "{$DAY}" : date_helper.day_str_from_int(dt.day)})
            if debug:
                print(formatted_file_glob, input_dir)
            
            list_of_file_names_or_empty = glob.glob(pathname = formatted_file_glob, root_dir = input_dir)
            
            if len(list_of_file_names_or_empty) > 0:
                
                sorted_list_of_file_names = sorted(list_of_file_names_or_empty)
                sorted_list_of_paths = [os.path.join(os.path.abspath(input_dir), name) for name in sorted_list_of_file_names]
                paths.extend(sorted_list_of_paths)
                
            else:
                if debug and verbose:
                    warnings.warn(f"No file on disk matches the following glob: {os.path.join(os.path.abspath(input_dir), formatted_file_glob)}")

                    
    elif ("{$MONTH}" in file_glob) or ("{$MONTH}" in input_dir_structure):
                
        for dt in rrule.rrule(freq = rrule.MONTHLY, dtstart = datetime.datetime(year = start.year, month=start.month, day=1), until = end, bymonthday=(1)):
            
            formatted_file_glob = global_configuration.replace_all_keys_in_string_with_values(file_glob, {"{$YEAR}" : dt.year,
                                                                                                "{$MONTH}" : date_helper.month_str_from_int(dt.month),
                                                                                                "{$DAY}" : date_helper.day_str_from_int(dt.day)})
            
            input_dir = global_configuration.replace_all_keys_in_string_with_values(input_dir_structure, {"{$YEAR}" : dt.year,
                                                                                                          "{$MONTH}" : date_helper.month_str_from_int(dt.month),
                                                                                                          "{$DAY}" : date_helper.day_str_from_int(dt.day)})
            if debug:
                print(formatted_file_glob, input_dir)
                
            list_of_file_names_or_empty = glob.glob(pathname = formatted_file_glob, root_dir = input_dir)
            
            if len(list_of_file_names_or_empty) > 0:
                
                sorted_list_of_file_names = sorted(list_of_file_names_or_empty)
                sorted_list_of_paths = [os.path.join(os.path.abspath(input_dir), name) for name in sorted_list_of_file_names]
                paths.extend(sorted_list_of_paths)
            else:
                if debug and verbose:
                    warnings.warn(f"No file on disk matches the following glob: {os.path.join(os.path.abspath(input_dir), formatted_file_glob)}")
                    
    elif ("{$YEAR}" in file_glob) or ("{$YEAR}" in input_dir_structure):

        for dt in rrule.rrule(freq = rrule.YEARLY, dtstart = datetime.datetime(year = start.year, month=1, day=1), until = end, byyearday=(1)):
                        
            formatted_file_glob = global_configuration.replace_all_keys_in_string_with_values(file_glob, {"{$YEAR}" : dt.year,
                                                                                                "{$MONTH}" : date_helper.month_str_from_int(dt.month),
                                                                                                "{$DAY}" : date_helper.day_str_from_int(dt.day)})
            
            input_dir = global_configuration.replace_all_keys_in_string_with_values(input_dir_structure, {"{$YEAR}" : dt.year,
                                                                                                          "{$MONTH}" : date_helper.month_str_from_int(dt.month),
                                                                                                          "{$DAY}" : date_helper.day_str_from_int(dt.day)})
            if debug:
                print(formatted_file_glob, input_dir)
                
            list_of_file_names_or_empty = glob.glob(pathname = formatted_file_glob, root_dir = input_dir)
            
            if len(list_of_file_names_or_empty) > 0:
                
                sorted_list_of_file_names = sorted(list_of_file_names_or_empty)
                sorted_list_of_paths = [os.path.join(os.path.abspath(input_dir), name) for name in sorted_list_of_file_names]
                paths.extend(sorted_list_of_paths)
            else:
                if debug and verbose:
                    warnings.warn(f"No file on disk matches the following glob: {os.path.join(os.path.abspath(input_dir), formatted_file_glob)}")
    else:
        
        raise Exception("Neither the file_glob nor the input_dir_structure had {$YEAR} or {$MONTH} or {$DAY} in it. So how can I tell what the ordering of the files should be?")
    
    return paths
    

def load_data_files(paths : list[str],
                    extension : str,
                    variables : list[str],
                    variable_config : dict,
                    debug : bool = False) -> dict:
    
    if debug:
        print(f"Paths: {paths}")
        print(f"Variables requested: {variables}")
        print(f"Variable config: {variable_config}")
    
    #Concatenating all the arrays at once is much faster than copying every time?
    unconcatenated_arrays = {}
    
    for var in variables:
        unconcatenated_arrays[var] = []
    
    if extension == ".cdf":
        
        for i, path in enumerate(paths):
        
            with cdflib.CDF(path = path) as cdf_file:
                
                for var in variables:
                    
                    if (i == 0) and (variable_config[var] is None):
                        unconcatenated_arrays[var].append(cdf_file.varget(variable=var, startrec=0))
                    elif (variable_config is not None):
                        unconcatenated_arrays[var].append(cdf_file.varget(variable=var, startrec=0))
    
    if extension == ".h5":
        
        for i, path in enumerate(paths):
            
            with h5py.File(path, "r") as h5_file:
                
                for var in variables:

                    if (i == 0) and (variable_config[var] is None):
                        unconcatenated_arrays[var].append(h5_file[var][...])
                    elif (variable_config is not None):
                        unconcatenated_arrays[var].append(h5_file[var][...])
    
    if extension == ".nc":
        
        for i, path in enumerate(paths):
            
            with Dataset(path, "r") as nc_file:
                
                for var in variables:

                    if (i == 0) and (variable_config[var] is None):
                        unconcatenated_arrays[var].append(np.ma.MaskedArray.filled(nc_file.variables[var][...], fill_value = np.NaN))
                    elif (variable_config is not None):
                        unconcatenated_arrays[var].append(np.ma.MaskedArray.filled(nc_file.variables[var][...], fill_value = np.NaN))
                
    refs = {}
    
    for var in variables:
        refs[var] = np.concatenate(unconcatenated_arrays[var], axis = variable_config[var])
    
    return refs

def load_raw_data_from_config(id : list[str], 
                              start : datetime.datetime,
                              end : datetime.datetime,
                              variables : list[str] = [],
                              satellite: str = "",
                              config_path : str = "",
                              debug : bool = False,
                              verbose: bool = False) -> dict:
    
    '''If you don't specify any variables here, all the variables in the config will be loaded... Otherwise an error will be thrown.'''
    
    config, config_path = global_configuration.Config(config_path).load()
    
    id_config = config
    for level in id:
        id_config = id_config[level]
    
    if os.environ.get("RESEARCH_RAW_DATA_DIR"):
        input_dir_structure = os.path.join(os.environ["RESEARCH_RAW_DATA_DIR"], *id)
    else:
        input_dir_structure = os.path.join(os.path.abspath(os.path.dirname(config_path)), *id)
    
    if "subdir" in id_config:
        input_dir_structure = os.path.join(input_dir_structure, *id_config["subdir"].split("/"))
    
    
    if "file_glob" not in id_config.keys():
        raise Exception("Tried to load an ID with no file_glob set in the global config. I have no idea what the filename looks like or what type it is!")

    file_name, file_extension = os.path.splitext(id_config["file_glob"])
    
    #More functionality can be added here before the loading routines if absolutely needed
    if "{$SATELLITE}" in file_name or "{$SATELLITE}" in input_dir_structure:
        if not satellite:
            raise Exception("File name or input_dir for this ID requires a satellite but no satellite was specified!")
        else:
            file_name = global_configuration.replace_all_keys_in_string_with_values(file_name, {"{$SATELLITE}": satellite})
            input_dir_structure = global_configuration.replace_all_keys_in_string_with_values(input_dir_structure, {"{$SATELLITE}": satellite})
    #-------------------------------------------------------------------------------
    
    if "variables" in id_config.keys():  
        if not variables:
            variables = list(id_config["variables"].keys())
    else:
        raise Exception("The config has no variables specified!")
    
    
    paths_of_files_within_timeperiod = get_file_names_between_start_and_end(start = start,
                                                                            end = end,
                                                                            file_glob = (file_name + file_extension),
                                                                            input_dir_structure = input_dir_structure,
                                                                            debug = debug,
                                                                            verbose = verbose)
    
    return load_data_files(paths = paths_of_files_within_timeperiod,
                           extension = file_extension,
                           variables = variables, 
                           variable_config = id_config["variables"],
                           debug = debug)

def load_omni_data_1hour_res(start: datetime.datetime,
                             end: datetime.datetime,
                             raw_data_dir: str = "./../raw_data/") -> dict:
    
    _1_hour_res_dir = os.path.join(os.path.abspath(raw_data_dir), "OMNI", "_1_hour_res")
    
    os_helper.verify_input_dir_exists(directory = _1_hour_res_dir,
                                      hint = "1 HOUR OMNI INPUT DIR")
    
    DST = np.zeros(shape=0, dtype=np.int32)
    epoch = np.zeros(shape=0, dtype=datetime.datetime)
    Kp = np.zeros(shape=0, dtype=np.float32)
    
    for i, dt in enumerate(rrule.rrule(rrule.YEARLY, dtstart=start, until=end)):

        _year = str(dt.year)
        
        first_file_for_year = os.path.join(_1_hour_res_dir, f"omni2_h0_mrg1hr_{_year}0101_v01.cdf")
        second_file_for_year = os.path.join(_1_hour_res_dir, f"omni2_h0_mrg1hr_{_year}0701_v01.cdf")
        
        if not os.path.exists(first_file_for_year):
            
            raise Exception(f"The requested file does not exist: {first_file_for_year}")
        
        if not os.path.exists(second_file_for_year):
            
            raise Exception(f"The requested file does not exist: {second_file_for_year}")
        
        print(f"Loading: {first_file_for_year}")
        
        omni_1 = pycdf.CDF(first_file_for_year)
        
        print(f"Loading: {second_file_for_year}")

        omni_2 = pycdf.CDF(second_file_for_year)
                
        DST = np.concatenate((DST, omni_1["DST"], omni_2["DST"]), axis = 0)
        epoch = np.concatenate((epoch, omni_1["Epoch"], omni_2["Epoch"]), axis = 0)
        Kp = np.concatenate((Kp, omni_1["KP"][...].astype("float32") / 10.0, omni_2["KP"][...].astype("float32") / 10.0), axis = 0) #Divided by 10 here cause CDF is Kp * 10 for some stupid reason
    
    satisfies_date_extent = (start < epoch) & (epoch < end)
    DST = DST[satisfies_date_extent]
    epoch = epoch[satisfies_date_extent]
    Kp = Kp[satisfies_date_extent]
    
    refs = {
        
        "DST": DST,
        "EPOCH": epoch,
        "Kp" : Kp
    }
    
    return refs
        

def load_omni_data_1min_res(start: datetime.datetime, 
                            end: datetime.datetime,
                            raw_data_dir: str = "./../raw_data/") -> dict:
    
    _1_min_res_dir = os.path.join(os.path.abspath(raw_data_dir), "OMNI", "_1_min_res")

    bz = np.zeros(shape=0, dtype=np.float32)
    ae_index = np.zeros(shape=0, dtype=np.int32)
    epoch = np.zeros(shape=0, dtype=np.object_)
        
    print(f"Loading OMNI data between: {start} and {end}")
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):
        
        _year = str(dt.year)
        _month = str(dt.month)
        
        if len(_month) < 2:
            _month = f"0{_month}"
            
        omni_data_dir = os.path.join(_1_min_res_dir, f"{_year}/")
        
        os_helper.verify_input_dir_exists(directory = omni_data_dir,
                                          hint = "1 MIN OMNI DATA DIR")
        
        omni_file_name = f"omni_hro2_1min_{_year}{_month}*.cdf"
        omni_cdf_path_or_empty = glob.glob(omni_file_name, root_dir=omni_data_dir)

        if len(omni_cdf_path_or_empty) != 0:
            
            omni_cdf_path = os.path.join(omni_data_dir, omni_cdf_path_or_empty[0])
        
        omni = pycdf.CDF(omni_cdf_path)
        bz = np.concatenate((bz, omni["BZ_GSM"][...]), axis=0, dtype=np.float32)
        epoch = np.concatenate((epoch, omni["Epoch"][...]), axis=0)
        ae_index = np.concatenate((ae_index, omni["AE_INDEX"][...]), axis=0, dtype=np.int32)
        
        satisfies_date_extent = (start < epoch) & (epoch < end)
        bz = bz[satisfies_date_extent]
        epoch = epoch[satisfies_date_extent]
        ae_index = ae_index[satisfies_date_extent]

        bz[bz > 9999] = np.NaN

        print(f"Loaded OMNI Data for : {dt}")
        
    refs = {
        
        "Bz" : bz,
        "AE" : ae_index,
        "EPOCH": epoch
        
    }
        
    return refs
    
              
def load_compressed_rept_data(satellite: str,
                              start: datetime.datetime, end: datetime.datetime,
                              compressed_data_dir: str = "./../compressed_data/") -> dict:
    
    rept_data_dir = os.path.join(os.path.abspath(compressed_data_dir),  "RBSP", "REPT")
    
    fesa = np.zeros(shape=(0, 12), dtype=np.float64)
    L = np.zeros(shape=0, dtype=np.float64)
    mlt = np.zeros(shape=0, dtype=np.float64)
    epoch: npt.NDArray[np.object_] = np.zeros(shape=0, dtype=datetime.datetime)
    
    print(f"Loading REPT data between: {start} and {end}.")
    
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):
        
        _year = str(dt.year)
        _month = str(dt.month)
        
        if len(_month) < 2:
            _month = f"0{_month}"
        
        rept_data_dir_year = os.path.join(os.path.abspath(rept_data_dir), f"{_year}")
                
        os_helper.verify_input_dir_exists(directory = rept_data_dir_year,
                                          hint = "REPT DATA DIR")
        
        rept_file_name = f"REPT_{_year}{_month}_{satellite.upper()}.npz"
        rept_data_path = os.path.join(rept_data_dir_year, rept_file_name)

        if not os.path.exists(rept_data_path):
            raise Exception(f"\nData file not found: {rept_data_path}")
        
        print(f"Loading : {rept_file_name}")
        data = np.load(rept_data_path, allow_pickle=True)
        
        fesa = np.concatenate((fesa, data["FESA"]), axis = 0)
        L = np.concatenate((L, data["L"]), axis = 0)
        epoch = np.concatenate((epoch, data["EPOCH"]), axis = 0)
        mlt = np.concatenate((mlt, data["MLT"]), axis = 0)
        
        if i == 0:
            energies: npt.NDArray[np.float64] = data["ENERGIES"]
        
        data.close()
        
    satisfies_date_extent = (start < epoch) & (epoch < end)
    fesa = fesa[satisfies_date_extent, :]
    L = L[satisfies_date_extent]
    mlt = mlt[satisfies_date_extent]
    epoch = epoch[satisfies_date_extent]
    
    fesa[fesa < 0] = np.NaN
    
    refs = {
        
        "FESA" : fesa,
        "L" : L,
        "MLT" : mlt,
        "EPOCH" : epoch, 
        "ENERGIES" : energies
        
    }
    
    return refs


def load_compressed_poes_data(satellite: str,
                              start: datetime.datetime, end: datetime.datetime,
                              poes_dir: str = "./../compressed_data/POES/CLEAN/") -> dict:
    
    input_data_dir = os.path.join(os.path.abspath(poes_dir), f"{satellite}")
    
    if not os_helper.verify_input_dir_exists(directory = input_data_dir, hint = "POES DATA DIR", raise_exception = False):
        
        print(f"Unable to find any compressed POES data for: {satellite}")
    
    #Time
    epoch = np.zeros(shape=0, dtype=datetime.datetime)
    
    #Coordinates
    alt = np.zeros(shape=(0), dtype=np.float32)
    lat = np.zeros(shape=(0), dtype=np.float32)
    lon = np.zeros(shape=(0), dtype=np.float32)
    L = np.zeros(shape=0, dtype=np.float32)
    mlt = np.zeros(shape=0, dtype=np.float32)
    
    #Flux
    mep_ele_tel0_flux_e1 = np.zeros(shape=(0), dtype=np.float32)
    mep_ele_tel0_flux_e2 = np.zeros(shape=(0), dtype=np.float32)
    
    #Pitch Angles
    meped_alpha_0_sat = np.zeros(shape=(0), dtype=np.float32)

    
    for dt in rrule.rrule(rrule.YEARLY, dtstart=datetime.datetime(year = start.year, month=1, day=1), until = end, byyearday=(1)):
        
        year = dt.year

        poes_file_name = f"POES_{year}_{satellite.lower()}_CLEAN.npz"
        poes_data_path = os.path.join(input_data_dir, poes_file_name)

        if not os.path.exists(poes_data_path):
            print(f"\nData file not found: {poes_data_path}, continuing...!")
            continue

        print(f"Loading : {poes_file_name}")
        data = np.load(poes_data_path, allow_pickle=True)

        #Time
        epoch = np.concatenate((epoch, data["EPOCH"]), axis=0)
        
        #Coordinates
        alt = np.concatenate((alt, data["ALT"]), axis = 0)
        lat = np.concatenate((lat, data["LAT"]), axis = 0)
        lon = np.concatenate((lon, data["LON"]), axis = 0)
        L = np.concatenate((L, data["L"]), axis = 0)
        mlt = np.concatenate((mlt, data["MLT"]), axis = 0)
        
        #Flux
        mep_ele_tel0_flux_e1 = np.concatenate((mep_ele_tel0_flux_e1, data["MEP_ELE_TEL0_FLUX_E1"]), axis = 0)
        mep_ele_tel0_flux_e2 = np.concatenate((mep_ele_tel0_flux_e2, data["MEP_ELE_TEL0_FLUX_E2"]), axis = 0)
        
        #Pitch Angles
        meped_alpha_0_sat = np.concatenate((meped_alpha_0_sat, data["MEPED_ALPHA_0_SAT"]))

        data.close()

    satisfies_date_extent = (start < epoch) & (epoch < end)
    #Time
    epoch = epoch[satisfies_date_extent]
    
    #Coordinates
    alt = alt[satisfies_date_extent]
    lat = lat[satisfies_date_extent]
    lon = lon[satisfies_date_extent]
    L = L[satisfies_date_extent]
    mlt = mlt[satisfies_date_extent]
    
    #Flux
    mep_ele_tel0_flux_e1 = mep_ele_tel0_flux_e1[satisfies_date_extent]
    mep_ele_tel0_flux_e2 = mep_ele_tel0_flux_e2[satisfies_date_extent]
    
    #Pitch Angles
    meped_alpha_0_sat = meped_alpha_0_sat[satisfies_date_extent]
    
    refs = {
        
        "EPOCH" : epoch,
        "ALT" : alt,
        "LAT" : lat,
        "LON" : lon,
        "L" : L,
        "MLT" : mlt,
        "MEP_ELE_TEL0_FLUX_E1" : mep_ele_tel0_flux_e1,
        "MEP_ELE_TEL0_FLUX_E2" : mep_ele_tel0_flux_e2,
        "MEPED_ALPHA_0_SAT" : meped_alpha_0_sat
        
    }

    return refs

def load_psd(satellite: str,
             field_model: model,
             start: datetime.datetime, end: datetime.datetime, 
             compressed_data_dir: str = "./../compressed_data/") -> dict:
    
    psd_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "PSD")
    
    os_helper.verify_input_dir_exists(psd_dir, hint="PSD DIR")
    
    PSD = np.zeros((0, 35, 102), dtype=np.float64)
    JD = np.zeros((0), dtype=np.float64)
    EPOCH = np.zeros((0), dtype=datetime.datetime)
    
    ENERGIES = np.zeros((0, 102), dtype=np.float64)
    ALPHA = np.zeros((0, 35), dtype=np.float64)
    
    K = np.zeros((0, 35), dtype=np.float64)
    L_STAR = np.zeros((0, 35), dtype=np.float64)
    L = np.zeros((0, 35), np.float64)
    IN_OUT = np.zeros((0), dtype=np.int32)
    ORBIT_NUMBER = np.zeros((0), dtype=np.int32)
    
    B = np.zeros((0), dtype=np.float64)
        
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart = datetime.datetime(year=start.year, month=start.month, day=1), until = end)):
        
        print(dt)
        
        _year = str(dt.year)
        _month = str(dt.month)

        if len(_month) < 2:
            _month = f"0{_month}"
            
        file_name = f"PSD_{_year}{_month}_{satellite.upper()}_{field_model.name}.npz"
        
        psd_path = os.path.join(psd_dir, file_name)
        
        if not os.path.exists(psd_path):
            raise Exception(f"\nData file not found: {psd_path}!")
        
        
        print(f"Loading : {file_name}")
        data = np.load(psd_path, allow_pickle=True)
        
        PSD = np.concatenate((PSD, data["PSD"]), axis = 0)
        JD = np.concatenate((JD, data["JD"]), axis = 0)
        EPOCH = np.concatenate((EPOCH, data["EPOCH"]), axis = 0)
        
        ENERGIES = np.concatenate((ENERGIES, data["ENERGIES"]), axis = 0)
        ALPHA = np.concatenate((ALPHA, data["ALPHA"]), axis = 0)

        K = np.concatenate((K, data["K"]), axis = 0)
        L_STAR = np.concatenate((L_STAR, data["L_STAR"]), axis = 0)
        L = np.concatenate((L, data["L"]), axis = 0)
        IN_OUT = np.concatenate((IN_OUT, data["IN_OUT"]), axis = 0)
        ORBIT_NUMBER = np.concatenate((ORBIT_NUMBER, data["ORBIT_NUMBER"]), axis = 0)
            
        B = np.concatenate((B, data["B"]), axis = 0)
        
        data.close()
        
    satisfies_timespan = (start < EPOCH) & (EPOCH < end)
    PSD = PSD[satisfies_timespan, :, :]
    JD = JD[satisfies_timespan]
    EPOCH = EPOCH[satisfies_timespan]
    ENERGIES = ENERGIES[satisfies_timespan, :]
    ALPHA = ALPHA[satisfies_timespan, :]
    K = K[satisfies_timespan, :]
    L_STAR = L_STAR[satisfies_timespan, :]
    L = L[satisfies_timespan, :]
    IN_OUT = IN_OUT[satisfies_timespan]
    ORBIT_NUMBER = ORBIT_NUMBER[satisfies_timespan]
    B = B[satisfies_timespan]
    
    refs = {
        "PSD" : PSD,
        "JD" : JD,
        "EPOCH" : EPOCH,
        "ENERGIES" : ENERGIES,
        "ALPHA" : ALPHA,
        "K" : K,
        "L_STAR" : L_STAR,
        "L" : L,
        "IN_OUT" : IN_OUT,
        "ORBIT_NUMBER" : ORBIT_NUMBER,
        "B" : B
    }
    
    return refs
    

if __name__ == "__main__":
    
    refs = load_omni_data_1hour_res(start = datetime.datetime(year = 2013, month = 1, day = 1),
                             end = datetime.datetime(year = 2014, month = 1, day = 1))
    
    print(refs.keys())