#You need wget installed to use the methods in this file.

import subprocess
import datetime
import os
import os_helper
from dateutil import rrule
from field_models import model
import calendar
import global_configuration
import date_helper


def wget_r_directory(url: str, savdir: str, file_glob = "*.cdf") -> None:

    '''By default scrapes '*.cdf' for to support legacy code here. Probably should set file_glob to something more specific.'''

    subprocess.run(args=["wget",
                          "-4",
                          "-e robots=off",
                          "--no-check-certificate",
                          "--retry-connrefused",
                          "-t",
                          "0",
                          "-N",
                          "--no-if-modified-since",
                          "-r",
                          "-nd",
                          "--no-parent",
                          "-A",
                          file_glob,
                          url,
                          "-P",
                          savdir],
                    shell=False)

def wget_file(filename: str, url: str, savdir: str) -> None:
    
    subprocess.call(args=["wget",
                          "--recursive",
                          "-e robots=off",
                          "--user-agent=Mozilla/5.0",
                          "--retry-connrefused",
                          "--no-check-certificate",
                          "-nd",
                          "--no-parent",
                          "-A",
                          filename,
                          url],
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=savdir)


def download_from_global_config(id : list[str], 
                                replace : dict = None, 
                                config_path : str = "", 
                                raw_data_dir : str = "",
                                use_config_keys_in_subdir : bool = True,
                                make_dirs : bool = False,
                                debug : bool = False):
    '''Downloads files using the global configuration. 
       Instead of having a method for every single variation of download we want, we can use a config and replace substrings.
       This just reduces the amount of code that needs to be written and maintained substantially, in exchange for possible configuration problems.'''
       
    config, config_path = global_configuration.Config(config_path).load()
    
    id_config = config
    for level in id:
        id_config = id_config[level]
    
    if "file_glob" not in id_config.keys():
        raise Exception("Tried to download an ID with no file_glob set in the global config. I have no idea what the filename looks like or what type it is!")
    
    if "url" not in id_config.keys():
        raise Exception("Tried to download an ID with no url set in the global config. I have no idea what url to look for the file_glob within!!")
    
    file_glob = global_configuration.replace_all_keys_in_string_with_values(id_config["file_glob"], map = replace)
    url = global_configuration.replace_all_keys_in_string_with_values(id_config["url"], map = replace)
    
    if debug:
        print(f"RAW FILE_GLOB : {id_config["file_glob"]}")
        print(f"RAW URL : {id_config["url"]}")
        print(f"Processed FILE_GLOB : {file_glob}")
        print(f"Procesed URL : {url}")
        
    if use_config_keys_in_subdir:
            
        if os.environ.get("RESEARCH_RAW_DATA_DIR"):
            output_dir = os.path.join(os.path.abspath(os.environ["RESEARCH_RAW_DATA_DIR"]), *id)
        elif raw_data_dir:
            output_dir = os.path.join(os.path.abspath(os.path.dirname(raw_data_dir)), *id)
        else:
            output_dir = os.path.join(os.path.abspath(os.path.dirname(config_path)), *id)
    
    else:
        
        if os.environ.get("RESEARCH_RAW_DATA_DIR"):
            output_dir = os.path.abspath(os.environ["RESEARCH_RAW_DATA_DIR"])
        elif raw_data_dir:
            output_dir = os.path.abspath(os.path.dirname(raw_data_dir))
        else:
            output_dir = os.path.abspath(os.path.dirname(config_path))
    
    if "subdir" in id_config:
        subdir = global_configuration.replace_all_keys_in_string_with_values(id_config["subdir"], map = replace)
        output_dir = os.path.abspath(os.path.join(os.path.abspath(output_dir), *subdir.split("/")))
            
    os_helper.verify_output_dir_exists(directory = output_dir,
                                       force_creation = make_dirs,
                                       hint = f"OUTPUT DIR for: {id}")
    
    if debug:
        print(f"OUTPUT DIR: {output_dir}")
    
    wget_r_directory(file_glob = file_glob,
                     url = url,
                     savdir = output_dir)
    
    if not os.listdir(output_dir):
        os.rmdir(output_dir)


#The following methods are essentially just examples of how to use the global config to download files

def download_year_omni_one_min_resolution(year: int, 
                                          make_dirs: bool = False,
                                          config_path = "",
                                          raw_data_dir: str = "",
                                          use_config_keys_in_subdir : bool = True,
                                          debug = False) -> None:
    
    '''Example : download_year_omni_one_min_resolution(year = 2013, make_dirs = True)'''
    
    download_from_global_config(id = ["OMNI", "ONE_MIN_RESOLUTION"], 
                                replace = {"{$YEAR}" : str(year),
                                           "{$MONTH}" : ""},
                                config_path = config_path,
                                raw_data_dir = raw_data_dir,
                                use_config_keys_in_subdir = use_config_keys_in_subdir,
                                make_dirs = make_dirs,
                                debug = debug)


def download_year_omni_one_hour_resolution(year: int, 
                                           make_dirs: bool = False,
                                           config_path = "",
                                           raw_data_dir: str = "",
                                           use_config_keys_in_subdir : bool = True,
                                           debug = False) -> None:
    
    '''Example: download_year_omni_one_hour_resolution(year = 2013, make_dirs = True)'''

    
    download_from_global_config(id = ["OMNI", "ONE_HOUR_RESOLUTION"], 
                                replace = {"{$YEAR}" : str(year)},
                                config_path = config_path,
                                raw_data_dir = raw_data_dir,
                                use_config_keys_in_subdir = use_config_keys_in_subdir,
                                make_dirs = make_dirs,
                                debug = debug)

def download_year_psd_dependencies(satellite: str,
                                   field_model : model,
                                   year : int,
                                   make_dirs: bool = False,
                                   config_path = "",
                                   debug = False):
    
    '''
    Parameters:
        satellite: Either "a" for RBSPA, or "b" for RBSPB
        field_model: Field model to download. Supported: "TS04D", "T89D"
        year : The year to download
        make_dirs: Whether or not to force the creation of the output directories
        raw_data_dir: Directory where the raw data is stored.
    '''
    
    download_from_global_config(id = ["RBSP", "ECT", "L3"], 
                                replace = {"{$SATELLITE}" : satellite.lower(),
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : "",
                                           "{$DAY}" : ""},
                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)
    
    
    download_from_global_config(id = ["RBSP", "MAGEPHEM", field_model.name], 
                                replace = {"{$SATELLITE}" : satellite.lower(),
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : "",
                                           "{$DAY}" : ""},
                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)
    
    download_from_global_config(id = ["RBSP", "EMFISIS", "L3"], 
                                replace = {"{$SATELLITE}" : satellite.lower(),
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : "",
                                           "{$DAY}" : ""},
                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)
    

        
def download_month_rbsp_rept_l2(satellite: str,
                                month: int, year: int,
                                make_dirs: bool = False,
                                config_path = "",
                                debug = False) -> None:

    '''Example: download_month_rbsp_rept_l2(satellite = "a", month = 1, year = 2013, make_dirs = True)'''
    
    download_from_global_config(id = ["RBSP", "REPT", "L2"], 
                                replace = {"{$SATELLITE}" : satellite.lower(),
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : date_helper.month_str_from_int(month),
                                           "{$DAY}" : ""},
                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)

def download_month_goes_eps(satid: str,
                            month: int,
                            year: int, 
                            make_dirs: bool = False,
                            config_path = "",
                            debug = False) -> None:
    
    '''Example: download_month_goes_eps(satellite = "g08", month = 5, year = 1998, make_dirs = True)'''
    
    satid = satid.lower()
    
    download_from_global_config(id = ["GOES", "AVG", "EPS"], 
                                replace = {"{$SATELLITE}" : f"{satid[0]}{satid[-2:]}",
                                           "{$SATID}" : satid,
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : date_helper.month_str_from_int(month)},
                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)

def download_month_goes_epead(satid: str,
                              month: int,
                              year: int, 
                              make_dirs: bool = False,
                              config_path = "",
                              debug = False) -> None:
    
    '''Example: download_month_goes_epead(satellite = "g15", month = 5, year = 2014, make_dirs = True)'''
    
    satid = satid.lower()
    
    download_from_global_config(id = ["GOES", "AVG", "EPEAD"], 
                                replace = {"{$SATELLITE}" : f"{satid[0]}{satid[-2:]}",
                                           "{$SATID}" : satid,
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : date_helper.month_str_from_int(month)},
                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)
    
def download_month_goes_sgps(satid: str,
                             month: int,
                             year: int, 
                             make_dirs: bool = False,
                             config_path = "",
                             debug = False) -> None:
    
    '''Example: download_month_goes_sgps(satellite = "g08", month = 5, year = 1998, make_dirs = True)'''
    
    satid = satid.lower()
    
    download_from_global_config(id = ["GOES", "AVG", "SGPS", "V1"], 
                                replace = {"{$SATELLITE}" : f"{satid[0]}{satid[-2:]}",
                                           "{$SATID}" : satid,
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : date_helper.month_str_from_int(month),
                                           "{$DAY}": "[0-9]*[0-9]*"},

                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)
    
    download_from_global_config(id = ["GOES", "AVG", "SGPS", "V2"], 
                                replace = {"{$SATELLITE}" : f"{satid[0]}{satid[-2:]}",
                                           "{$SATID}" : satid,
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : date_helper.month_str_from_int(month),
                                           "{$DAY}": "[0-9]*[0-9]*"},
                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)
    
    download_from_global_config(id = ["GOES", "AVG", "SGPS", "V3"], 
                                replace = {"{$SATELLITE}" : f"{satid[0]}{satid[-2:]}",
                                           "{$SATID}" : satid,
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : date_helper.month_str_from_int(month),
                                           "{$DAY}": "[0-9]*[0-9]*"},
                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)


def download_year_poes_after_2012(satid: str,
                                  year: int,
                                  make_dirs: bool = False,
                                  config_path = "",
                                  debug = False) -> None:
    
    '''Example: download_year_poes_after_2012(satid = "noaa15", year = 2014, make_dirs = True)'''
    
    satid = satid.lower()
    
    download_from_global_config(id = ["POES", "SEM", "L1B"], 
                                replace = {"{$SATELLITE}" : f"{satid[0]}{satid[-2:]}",
                                           "{$SATID}" : satid,
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : "",
                                           "{$DAY}" : ""},
                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)


def download_legacy_poes_1998_to_2014(satid: str,
                                      year: int,
                                      make_dirs: bool = False,
                                      config_path = "",
                                      debug = False) -> None:
    '''Example: download_legacy_poes_1998_to_2014(satid = "noaa15", year = 2010, make_dirs = True)'''

    satid = satid.lower()

    download_from_global_config(id = ["POES_LEGACY", "SEM", "L2"], 
                                replace = {"{$SATELLITE}" : f"{satid[0]}{satid[-2:]}",
                                           "{$SATID}" : satid,
                                           "{$YEAR}" : str(year),
                                           "{$MONTH}" : "",
                                           "{$DAY}" : ""},
                                config_path = config_path,
                                make_dirs = make_dirs,
                                debug = debug)


def raster_download_goes_for_solar_cycle_study(debug: bool):
    
    #for satellite in ['goes08', 'goes09', 'goes10', 'goes11', 'goes12', 'goes13', 'goes14', 'goes15', 'goes16', 'goes17', 'goes18']:
    for satellite in ['goes16', 'goes17', 'goes18']:
        #for year in range(1998, 2025):
        for year in range(2024, 2025):
            for month in range(9, 13):
                
                #if satellite in ["goes08", "goes09", "goes10", "goes11", "goes12"]:
                #    download_month_goes_eps(satid = satellite, month = month, year = year, make_dirs = True, config_path="../config.yaml", debug=debug)
                #elif satellite in ["goes13", "goes14", "goes15"]:
                #    download_month_goes_epead(satid = satellite, month = month, year = year, make_dirs = True, config_path="../config.yaml", debug=debug)
                if satellite in ["goes16", "goes17", "goes18"]:
                    download_month_goes_sgps(satid = satellite, month = month, year = year, make_dirs = True, config_path="../config.yaml", debug=debug)

if __name__ == "__main__":

    
    for year in range(2012, 2020):
        
        download_year_omni_one_min_resolution(year = year, make_dirs = True, raw_data_dir = "./../raw_data/")