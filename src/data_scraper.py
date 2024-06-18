import glob
import subprocess
import datetime
import os
import os_helper
from dateutil import rrule
from field_models import model
import calendar
import re


def wget_r_directory(folder_url: str, savdir: str, file_glob = "*.cdf") -> None:

    '''By default scrapes '*.cdf' for to support legacy code here. Probably should set file_glob to something more specific.'''

    subprocess.call(args=["wget",
                          "-4",
                          "-e robots=off",
                          "--retry-connrefused",
                          "-nc",
                          "-c",
                          "-r",
                          "-nd",
                          "--no-parent",
                          "-A",
                          file_glob,
                          folder_url,
                          "-P",
                          savdir],
                    shell=False)


def wget_file(filename: str, folder_url: str, savdir: str) -> None:
    
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
                          folder_url],
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=savdir)


def download_year_omni_1min_res(year: int, 
                                make_dirs: bool = False, 
                                raw_data_dir: str = "./../raw_data/") -> None:

    output_dir = os.path.join(raw_data_dir, "GOES", "_1_min_res",  str(year))

    os_helper.verify_output_dir_exists(directory = output_dir, force_creation = make_dirs, hint="RAW OMNI DIR")

    start = datetime.datetime(year = year, month = 1, day = 1)
    end = datetime.datetime(year = year + 1, month = 1, day = 1)

    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until = end - datetime.timedelta(days=1))):

        _year = str(dt.year)
        _month = str(dt.month)

        if len(_month) < 2:
            _month = f"0{_month}"

        wget_file(filename = f"omni_hro2_1min_{_year}{_month}*.cdf",
                  folder_url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/{_year}/",
                  savdir = output_dir)
        
        print(f"Downloaded OMNI data for : {dt}")


def download_year_omni_1hour_res(year: int,
                                 make_dirs: bool = False,
                                 raw_data_dir: str = "./../raw_data/") -> None:
      
    
    output_dir = os.path.join(os.path.abspath(raw_data_dir), "OMNI", "_1_hour_res")
    
    os_helper.verify_output_dir_exists(directory = output_dir,
                                       force_creation = make_dirs,
                                       hint = "OMNI DATA DIR")

    wget_r_directory(folder_url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hourly/{year}/",
                     savdir = output_dir)


def download_month_rept_l2(satellite: str,
                           month: int, year: int,
                           make_dirs: bool = False,
                           raw_data_dir: str = "./../raw_data/") -> None:
    if month < 10:
        output_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "REPT", f"{year}/0{month}")

    else:
        output_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "REPT", f"{year}/{month}")

    os_helper.verify_output_dir_exists(directory = output_dir, force_creation=make_dirs, hint="RAW REPT DIR")
    
    start = datetime.datetime(year = year, month = month, day = 1)

    if month == 12:
        end = datetime.datetime(year = year + 1, month = 1, day = 1)
    else:
        end = datetime.datetime(year = year, month = month + 1, day = 1)

    curr = start
    while curr < end:

        _year = str(curr.year)
        _month = str(curr.month)
        _day = str(curr.day)

        if len(_month) < 2:
            _month = f"0{_month}"
        if len(_day) < 2:
            _day = f"0{_day}"

        wget_file(filename = f"rbsp{satellite.lower()}_rel03_ect-rept-sci-l2_{_year}{_month}{_day}*.cdf",
                  folder_url = f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite.lower()}/l2/ect/rept/sectors/rel03/{_year}/",
                  savdir = output_dir)
        
        print(f"Downloaded REPT Data for : {curr}")
        curr += datetime.timedelta(days=1)


def download_month_goes_netcdf(month: int, year: int,
                               make_dirs: bool = False,
                               raw_data_dir: str = "./../raw_data/") -> None:

    output_dir = os.path.join(raw_data_dir, "GOES", f"{year}/")

    os_helper.verify_output_dir_exists(directory = output_dir, force_creation = make_dirs, hint="RAW GOES DIR")

    daterange = calendar.monthrange(year, month)

    _year = str(year)
    _month = str(month)

    if len(_month) < 2:
        _month = f"0{_month}"

    _lastday = str(daterange[-1])

    wget_file(filename = f"g15_epead_cpflux_5m_{_year}{_month}01_{_year}{_month}{_lastday}.nc",
              folder_url = f"https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/{_year}/{_month}/goes15/netcdf/",
              savdir = output_dir)

    print(f"Downloaded GOES Data for : {datetime.datetime(year=year, month=month, day=1)}")


def download_year_goes_netcdf(year: int, 
                              make_dirs: bool = False, 
                              raw_data_dir: str = "./../raw_data/") -> None:

    start = datetime.datetime(year = year, month = 1, day = 1)
    end = datetime.datetime(year = year + 1, month = 1, day = 1)

    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until = end - datetime.timedelta(days=1))):

        download_month_goes_netcdf(month=dt.month, year=dt.year, make_dirs = make_dirs, raw_data_dir=raw_data_dir)


def download_poes_after_2012(satellite: str,
                             make_dirs: bool = False,
                             raw_data_dir: str = "./../raw_data/") -> None:

    output_dir = os.path.join(os.path.abspath(raw_data_dir), "POES", satellite.lower())

    os_helper.verify_output_dir_exists(directory=output_dir,
                                       force_creation=make_dirs,
                                       hint="RAW POES DIR")

    wget_r_directory(folder_url=f"https://spdf.gsfc.nasa.gov/pub/data/noaa/{satellite.lower()}/sem2_fluxes-2sec/",
                     savdir=output_dir)

def download_poes_1998_to_2014(year: int,
                               satellite: str,
                               make_dirs: bool = False,
                               raw_data_dir: str = "./../raw_data/") -> None:
    
    output_dir = os.path.join(os.path.abspath(raw_data_dir), "POES", satellite.lower())
    
    os_helper.verify_output_dir_exists(directory = output_dir,
                                       force_creation = make_dirs, 
                                       hint = "RAW POES DIR")
    
    wget_r_directory(folder_url = f"https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/l2/v01r00/cdf/{year}/{satellite.lower()}/",
                     savdir = output_dir)

def download_raster_poes_1998_to_2014(make_dirs = False,
                                      raw_data_dir : str = "./../raw_data/"):
    
     
    for _satellite in ["noaa15", "noaa16", "noaa17", "noaa18", "noaa19", "metop01", "metop02", "metop03"]:
        
        for _year in range(1998, 2015):
             
            download_poes_1998_to_2014(year = _year,
                                       satellite = _satellite,
                                       make_dirs = make_dirs,
                                       raw_data_dir = raw_data_dir)
    
    

def download_year_psd_dependencies(satellite: str,
                                   field_model : model,
                                   year : int,
                                   make_dirs : bool = False,
                                   raw_data_dir = "./../raw_data/"):
    
    '''
    Parameters:
        satellite: Either "a" for RBSPA, or "b" for RBSPB
        field_model: Field model to download. Supported: "TS04D", "T89D"
        year : The year to download
        make_dirs: Whether or not to force the creation of the output directories
        raw_data_dir: Directory where the raw data is stored.
    '''
    
    ect_L3_output_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "ECT", "L3")
    os_helper.verify_output_dir_exists(directory = ect_L3_output_dir,
                                       force_creation=make_dirs,
                                       hint="ECT L3 OUTPUT DIR")
    
    magephem_output_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "MAGEPHEM")
    os_helper.verify_output_dir_exists(directory = magephem_output_dir,
                                       force_creation=make_dirs,
                                       hint="MAGEPEHEM OUTPUT DIR")
    
    emfisis_output_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "EMFISIS")
    os_helper.verify_output_dir_exists(directory = emfisis_output_dir,
                                       force_creation=make_dirs,
                                       hint="EMFISIS OUTPUT DIR")
    
    ect_download_folder_url = f"https://rbsp-ect.newmexicoconsortium.org/data_pub/rbsp{satellite.lower()}/ECT/level3/{year}/"
    magephem_download_folder_url = f"https://rbsp-ect.newmexicoconsortium.org/data_pub/rbsp{satellite.lower()}/MagEphem/definitive/{year}/"
    emfisis_download_folder_url = f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite.lower()}/l3/emfisis/magnetometer/1sec/gse/{year}/"
    
    match field_model:
        
        case field_model.TS04D:
        
            magephem_file_glob = f"rbsp{satellite.lower()}_def_MagEphem_TS04D_{year}*.h5"

        case field_model.T89D:
            
            magephem_file_glob = f"rbsp{satellite.lower()}_def_MagEphem_T89D_{year}*.h5"

    wget_r_directory(folder_url = ect_download_folder_url,
                     savdir = ect_L3_output_dir)
    
    wget_r_directory(folder_url = magephem_download_folder_url,
                     file_glob = magephem_file_glob,
                     savdir = magephem_output_dir)
    
    wget_r_directory(folder_url = emfisis_download_folder_url,
                     savdir = emfisis_output_dir)
        

if __name__ == "__main__":

    download_year_psd_dependencies(satellite = "B", 
                                   field_model = model.TS04D,
                                   year = 2017,
                                   make_dirs = True)