from data_references import DataRefContainer
from dateutil import rrule
import datetime
import numpy as np
import pandas as pd
import os
from spacepy import pycdf
import glob


def load_omni_data(start: datetime.datetime, end: datetime.datetime,
                   omni_dir: str = "./../raw_data/OMNI/") -> pd.DataFrame:

    bz = np.zeros(shape=0, dtype=np.float32)
    ae_index = np.zeros(shape=0, dtype=np.int32)
    epoch = np.zeros(shape=0, dtype=datetime.datetime)
    
    print(f"Loading OMNI data between: {start} and {end}")
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):
        
        _year = str(dt.year)
        _month = str(dt.month)
        
        if len(_month) < 2:
            _month = f"0{_month}"
            
        data_dir = os.path.join(omni_dir, f"{_year}/")
        omni_file_name = f"omni_hro2_1min_{_year}{_month}*.cdf"
        omni_cdf_path_or_empty = glob.glob(omni_file_name, root_dir=data_dir)

        if len(omni_cdf_path_or_empty) != 0:
            
            omni_cdf_path = os.path.join(data_dir, omni_cdf_path_or_empty[0])
        
        omni = pycdf.CDF(omni_cdf_path)
        bz = np.concatenate((bz, omni["BZ_GSM"][...]), axis=0)
        epoch = np.concatenate((epoch, omni["Epoch"][...]), axis=0)
        ae_index = np.concatenate((ae_index, omni["AE_INDEX"][...]), axis=0)
        
        satisfies_date_extent = (start < epoch) & (epoch < end)
        bz = bz[satisfies_date_extent]
        epoch = epoch[satisfies_date_extent]
        ae_index = ae_index[satisfies_date_extent]

        bz[bz > 9999] = np.NaN

        print(f"Loaded OMNI Data for : {dt}")
        
    return pd.DataFrame(data={"Bz": bz, "AE": ae_index}, index=epoch)
    
              
def load_compressed_rept_data(satellite: str,
                              start: datetime.datetime, end: datetime.datetime,
                              rept_dir: str = "./../compressed_data/REPT/") -> DataRefContainer:
    
    fesa = np.zeros(shape=(0, 12), dtype=np.float64)
    L = np.zeros(shape=0, dtype=np.float64)
    mlt = np.zeros(shape=0, dtype=np.float64)
    epoch = np.zeros(shape=0, dtype=datetime.datetime)
    
    print(f"Loading REPT data between: {start} and {end}.")
    
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):
        
        _year = str(dt.year)
        _month = str(dt.month)
        
        if len(_month) < 2:
            _month = f"0{_month}"
        
        data_dir = os.path.join(rept_dir, f"{_year}/")
        rept_file_name = f"REPT_{_year}{_month}_{satellite.upper()}.npz"
        rept_data_path = os.path.join(data_dir, rept_file_name)

        if not os.path.exists(rept_data_path):
            raise Exception(f"\nData file not found: {rept_data_path}")
        
        print(f"Loading : {rept_file_name}")
        data = np.load(rept_data_path, allow_pickle=True)
        
        fesa = np.concatenate((fesa, data["FESA"]), axis = 0)
        L = np.concatenate((L, data["L"]), axis = 0)
        epoch = np.concatenate((epoch, data["EPOCH"]), axis = 0)
        mlt = np.concatenate((mlt, data["MLT"]), axis = 0)
        
        if i == 0:
            energies = data["ENERGIES"]
        
        data.close()
        
    satisfies_date_extent = (start < epoch) & (epoch < end)
    fesa = fesa[satisfies_date_extent, :]
    L = L[satisfies_date_extent]
    mlt = mlt[satisfies_date_extent]
    epoch = epoch[satisfies_date_extent]
    
    fesa[fesa < 0] = np.NaN
    
    return DataRefContainer(fesa, L, mlt, epoch, energies)
