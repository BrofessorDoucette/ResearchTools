import subprocess
import datetime
import os
from dateutil import rrule

def download_omni(month: str, year: str, directory: str):
    
    subprocess.call(["wget", "--no-check-certificate", "-r", "-nd", "--no-parent", "-A", f"omni_hro2_1min_{year}{month}*.cdf", f"https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/{year}/"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=directory)
    

def download_rept_L2(satellite: str, month: str, day: str, year: str, directory: str):
    
    subprocess.call(["wget", "--no-check-certificate", "-r", "-nd", "--no-parent", "-A", f"rbsp{satellite}_rel03_ect-rept-sci-l2_{year}{month}{day}*.cdf", f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite}/l2/ect/rept/sectors/rel03/{year}/"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=directory)
    
def download_month_REPT(satellite: str, month: int, year: int, make_dirs = False, RAW_DATA_DIR = "./../raw_data/REPT/"):
    
    if month < 10:
        OUTPUT_DIR = os.path.join(RAW_DATA_DIR, f"{year}/0{month}")
    else:
        OUTPUT_DIR = os.path.join(RAW_DATA_DIR, f"{year}/{month}")
    
    if not os.path.isdir(OUTPUT_DIR):
        if make_dirs:
            os.makedirs(OUTPUT_DIR)
        else:
            raise Exception("\nOutput directory doesn't exist, and make_dirs flag is set to false! Please make the directory or give me permission to make it for you.")
    
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
            
        download_rept_L2(satellite = satellite.lower(), month = _month, day = _day, year = _year, directory=OUTPUT_DIR)
        print(f"Downloaded REPT Data for : {curr}")
        curr += datetime.timedelta(days=1)

def download_year_OMNI(year: int, make_dirs = False, RAW_DATA_DIR = "./../raw_data/OMNI/"):
    
    OUTPUT_DIR = os.path.join(RAW_DATA_DIR, f"{year}")
    
    if not os.path.isdir(OUTPUT_DIR):
        if make_dirs:
            os.makedirs(OUTPUT_DIR)
        else:
            raise Exception("\nOutput directory doesn't exist, and make_dirs flag is set to false! Please make the directory or give me permission to make it for you.")

    start = datetime.datetime(year = year, month = 1, day = 1)
    end = datetime.datetime(year = year + 1, month = 1, day = 1)
    
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until = end - datetime.timedelta(days=1))):
        
        _year = str(dt.year)
        _month = str(dt.month)
        
        if len(_month) < 2:
            _month = f"0{_month}"

        download_omni(month=_month, year=_year, directory=OUTPUT_DIR)
        print(f"Downloaded OMNI data for : {dt}")
        
download_year_OMNI(2022, make_dirs=True)