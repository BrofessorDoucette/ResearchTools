from spacepy import pycdf
import glob
import os
import datetime
import numpy as np

def compress_month_REPT(satellite: str, month: int, year: int, make_dirs = False, RAW_DATA_DIR = "./../raw_data/REPT/", COMPRESSED_DATA_DIR = "./../compressed_data/"):
    
    OUTPUT_DIR = os.path.join(COMPRESSED_DATA_DIR, f"{year}")
    
    if not os.path.isdir(OUTPUT_DIR):
        
        if make_dirs:
            os.makedirs(OUTPUT_DIR)
        else:
            raise Exception("\nOutput directory does not exist, and make_dirs flag is False! Please make the output directory or give me permission to do so!")

    if month < 10:
        UNCOMPRESSED_DIR = os.path.join(RAW_DATA_DIR, f"{year}/0{month}")
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"REPT_{year}0{month}_{satellite.upper()}.npz")
    else:
        UNCOMPRESSED_DIR = os.path.join(RAW_DATA_DIR, f"{year}/{month}")
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"REPT_{year}{month}_{satellite.upper()}.npz")
        
    if not os.path.isdir(UNCOMPRESSED_DIR):
        raise Exception("\nRaw data folder not found!")
        
    start = datetime.datetime(year = year, month = month, day = 1)
    
    if month == 12:
        end = datetime.datetime(year = year + 1, month = 1, day = 1)
    else:
        end = datetime.datetime(year = year, month = month + 1, day = 1)
    
    FESA = np.zeros((0, 12), dtype=np.float64)
    L = np.zeros((0), dtype=np.float64)
    MLT = np.zeros((0), dtype=np.float64)
    EPOCH = np.zeros((0), dtype=datetime.datetime)
    
    curr = start
    while curr < end:
        
        _year = str(curr.year)
        _month = str(curr.month)
        _day = str(curr.day)
    
        if len(_month) < 2:
            _month = f"0{_month}"
    
        if len(_day) < 2:
            _day = f"0{_day}"
        
        REPT_CDF_PATH_OR_EMPTY = glob.glob(f"rbsp{satellite.lower()}_rel03_ect-rept-sci-l2_{_year}{_month}{_day}*.cdf", root_dir=UNCOMPRESSED_DIR)
        
        if len(REPT_CDF_PATH_OR_EMPTY) != 0:
            REPT_CDF_PATH = os.path.join(UNCOMPRESSED_DIR, REPT_CDF_PATH_OR_EMPTY[0])
        else:
            print(f"COULDN'T FIND REPT FILE FOR DATE: {_month}/{_day}/{_year}. Skipping!")
            curr += datetime.timedelta(days = 1)
            continue
        
        REPT = pycdf.CDF(REPT_CDF_PATH)
        FESA = np.concatenate((FESA, REPT["FESA"][...]), axis=0)
        L = np.concatenate((L, REPT["L"][...]), axis = 0)
        MLT = np.concatenate((MLT, REPT["MLT"][...]), axis = 0)
        EPOCH = np.concatenate((EPOCH, REPT["Epoch"][...]), axis = 0)
        
        curr += datetime.timedelta(days = 1)
    
    np.savez_compressed(OUTPUT_FILE, FESA=FESA, L=L, MLT=MLT, EPOCH=EPOCH, ENERGIES=REPT["FESA_Energy"][...])
    
    if month < 10:
        print(f"Compressed REPT Data for : {year}-0{month}")
    else:
        print(f"Compressed REPT Data for : {year}-{month}")
    