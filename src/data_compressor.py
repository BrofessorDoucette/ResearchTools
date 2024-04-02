from spacepy import pycdf
import glob
import os
import datetime
import numpy as np


def compress_month_rept_l2(satellite: str, month: int, year: int, make_dirs = False,
                           raw_data_dir ="./../raw_data/REPT/",
                           compressed_data_dir ="./../compressed_data/REPT/"):
    
    output_dir = os.path.join(compressed_data_dir, f"{year}")
    
    if not os.path.isdir(output_dir):
        
        if make_dirs:
            os.makedirs(output_dir)
        else:
            raise Exception("\nOutput directory does not exist, and make_dirs flag is False! Please make the output directory or give me permission to do so!")

    if month < 10:
        uncompressed_dir = os.path.join(raw_data_dir, f"{year}/0{month}")
        output_file = os.path.join(output_dir, f"REPT_{year}0{month}_{satellite.upper()}.npz")
    else:
        uncompressed_dir = os.path.join(raw_data_dir, f"{year}/{month}")
        output_file = os.path.join(output_dir, f"REPT_{year}{month}_{satellite.upper()}.npz")
        
    if not os.path.isdir(uncompressed_dir):
        raise Exception("\nRaw data folder not found!")
        
    start = datetime.datetime(year = year, month = month, day = 1)
    
    if month == 12:
        end = datetime.datetime(year = year + 1, month = 1, day = 1)
    else:
        end = datetime.datetime(year = year, month = month + 1, day = 1)
    
    fesa = np.zeros(shape=(0, 12), dtype=np.float64)
    L = np.zeros(shape=0, dtype=np.float64)
    mlt = np.zeros(shape=0, dtype=np.float64)
    epoch = np.zeros(shape=0, dtype=datetime.datetime)
    
    curr = start
    while curr < end:
        
        _year = str(curr.year)
        _month = str(curr.month)
        _day = str(curr.day)
    
        if len(_month) < 2:
            _month = f"0{_month}"
    
        if len(_day) < 2:
            _day = f"0{_day}"
        
        rept_cdf_path_or_empty = glob.glob(pathname=f"rbsp{satellite.lower()}_rel03_ect-rept-sci-l2_{_year}{_month}{_day}*.cdf",
                                           root_dir=uncompressed_dir)
        
        if len(rept_cdf_path_or_empty) != 0:
            rept_cdf_path = os.path.join(uncompressed_dir, rept_cdf_path_or_empty[0])
        else:
            print(f"COULDN'T FIND REPT FILE FOR DATE: {_month}/{_day}/{_year}. Skipping!")
            curr += datetime.timedelta(days = 1)
            continue
        
        rept = pycdf.CDF(rept_cdf_path)
        fesa = np.concatenate((fesa, rept["FESA"][...]), axis=0)
        L = np.concatenate((L, rept["L"][...]), axis = 0)
        mlt = np.concatenate((mlt, rept["MLT"][...]), axis = 0)
        epoch = np.concatenate((epoch, rept["Epoch"][...]), axis = 0)
        
        curr += datetime.timedelta(days = 1)
    
    np.savez_compressed(output_file, FESA=fesa, L=L, MLT=mlt, EPOCH=epoch, ENERGIES=rept["FESA_Energy"][...])
    
    if month < 10:
        print(f"Compressed REPT Data for : {year}-0{month}")
    else:
        print(f"Compressed REPT Data for : {year}-{month}")
    
    
def compress_month_poes_metop(satellite: str, month: int, year: int, make_dirs = False,
                              raw_data_dir ="./../raw_data/POES_METOP/",
                              compressed_data_dir ="./../compressed_data/POES_METOP/DIRTY/"):
        
    if month < 10:
        input_dir = os.path.join(raw_data_dir, f"{year}/0{month}")
        output_dir = os.path.join(compressed_data_dir, f"{year}/0{month}")
        
    else:
        input_dir = os.path.join(raw_data_dir, f"{year}/{month}")
        output_dir = os.path.join(compressed_data_dir, f"{year}/{month}")
        
    if not os.path.isdir(input_dir):
        raise Exception(f"\nInput directory folder not found! {input_dir}")
        
    if not os.path.isdir(output_dir):
        
        if make_dirs:
            os.makedirs(output_dir)
        else:
            raise Exception("\nOutput directory does not exist, and make_dirs flag is False! Please make the output directory or give me permission to do so!")
    
    output_file = os.path.join(output_dir, f"POES_METOP_{year}0{month}_{satellite.lower()}_DIRTY.npz")
    
    start = datetime.datetime(year = year, month = month, day = 1)
    
    if month == 12:
        end = datetime.datetime(year = year + 1, month = 1, day = 1)
    else:
        end = datetime.datetime(year = year, month = month + 1, day = 1)

    epoch = np.zeros(shape=0, dtype=datetime.datetime)
    mep_ele_flux = np.zeros(shape=(0, 2, 4), dtype=np.float32)
    L = np.zeros(shape=0, dtype=np.float32)
    mlt = np.zeros(shape=0, dtype=np.float32)

    num_loaded = 0
    
    curr = start
    while curr < end:
        
        _year = str(curr.year)
        _month = str(curr.month)
        _day = str(curr.day)
    
        if len(_month) < 2:
            _month = f"0{_month}"
    
        if len(_day) < 2:
            _day = f"0{_day}"
        
        poes_cdf_path_or_empty = glob.glob(pathname=f"{satellite.lower()}_poes-sem2_fluxes-2sec_{_year}{_month}{_day}*.cdf",
                                           root_dir=input_dir)
        
        if len(poes_cdf_path_or_empty) != 0:
            poes_cdf_path = os.path.join(input_dir, poes_cdf_path_or_empty[0])
            num_loaded += 1
        else:
            print(f"COULDN'T FIND POES FILE FOR {satellite}, FOR DATE: {_month}/{_day}/{_year}. Skipping!")
            curr += datetime.timedelta(days = 1)
            continue
        
        poes = pycdf.CDF(poes_cdf_path)
        epoch = np.concatenate((epoch, poes["Epoch"][...]), axis = 0)
        mep_ele_flux = np.concatenate((mep_ele_flux, poes["mep_ele_flux"][...]), axis=0)
        L = np.concatenate((L, poes["l_igrf"][...]), axis = 0)
        mlt = np.concatenate((mlt, poes["mlt"][...]), axis = 0)
        
        curr += datetime.timedelta(days = 1)
    
    if num_loaded != 0:
        np.savez_compressed(output_file, EPOCH=epoch, MEP_ELE_FLUX=mep_ele_flux, L=L, MLT=mlt)

        print(f"Compressed POES/METOP Data for {satellite} during: {start}")
    
    else:
        print(f"No POES/METOP Data for {satellite} was found to compress! Skipping!")
