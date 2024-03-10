from data_references import DataRefContainer
from dateutil import rrule
import datetime
import numpy as np
import pandas as pd
import os
from spacepy import pycdf
import glob

def load_omni_data(start: datetime.datetime, end: datetime.datetime, dir: str = "./../raw_data/OMNI/") -> pd.DataFrame:
    
    Bz = np.zeros((0), dtype=np.float32)
    AE_INDEX = np.zeros((0), dtype=np.int32)
    EPOCH = np.zeros((0), dtype=datetime.datetime)
    
    print(f"Loading OMNI data between: {start} and {end}")
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):
        
        _year = str(dt.year)
        _month = str(dt.month)
        
        if len(_month) < 2:
            _month = f"0{_month}"
            
        DATA_DIR = os.path.join(dir, f"{_year}/")
        FILE_NAME = f"omni_hro2_1min_{_year}{_month}*.cdf"
        OMNI_CDF_PATH_OR_EMPTY = glob.glob(FILE_NAME, root_dir=DATA_DIR)

        if len(OMNI_CDF_PATH_OR_EMPTY) != 0:
            
            OMNI_CDF_PATH = os.path.join(DATA_DIR, OMNI_CDF_PATH_OR_EMPTY[0])
        
        OMNI = pycdf.CDF(OMNI_CDF_PATH)
        Bz = np.concatenate((Bz, OMNI["BZ_GSM"][...]), axis=0)
        EPOCH = np.concatenate((EPOCH, OMNI["Epoch"][...]), axis=0)
        AE_INDEX = np.concatenate((AE_INDEX, OMNI["AE_INDEX"][...]), axis=0)
        
        SATISFIES_DATE_EXTENT = (start < EPOCH) & (EPOCH < end)
        Bz = Bz[SATISFIES_DATE_EXTENT]
        EPOCH = EPOCH[SATISFIES_DATE_EXTENT]
        AE_INDEX = AE_INDEX[SATISFIES_DATE_EXTENT]


        Bz[Bz > 9999] = np.NaN

        print(f"Loaded OMNI Data for : {dt}")
        
    return pd.DataFrame(data={"Bz": Bz, "AE": AE_INDEX}, index=EPOCH)
    
        
        
def load_compressed_data(satellite: str, start: datetime.datetime, end: datetime.datetime, dir: str = "./../compressed_data/") -> DataRefContainer:
    
    FESA = np.zeros((0, 12), dtype=np.float64)
    L = np.zeros((0), dtype=np.float64)
    MLT = np.zeros((0), dtype=np.float64)
    EPOCH = np.zeros((0), dtype=datetime.datetime)
    
    print(f"Loading REPT data between: {start} and {end}.")
    
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):
        
        _year = str(dt.year)
        _month = str(dt.month)
        
        if len(_month) < 2:
            _month = f"0{_month}"
        
        DATA_DIR = os.path.join(dir, f"{_year}/")
        FILE_NAME = f"REPT_{_year}{_month}_{satellite.upper()}.npz"
        DATA_PATH = os.path.join(DATA_DIR, FILE_NAME)
        
        if not os.path.exists(DATA_PATH):
            raise Exception(f"\nData file not found: {DATA_PATH}")
        
        print(f"Loading : {FILE_NAME}")
        data = np.load(DATA_PATH, allow_pickle=True)
        
        FESA = np.concatenate((FESA, data["FESA"]), axis = 0)
        L = np.concatenate((L, data["L"]), axis = 0)
        EPOCH = np.concatenate((EPOCH, data["EPOCH"]), axis = 0)
        MLT = np.concatenate((MLT, data["MLT"]), axis = 0)
        
        if i == 0:
            ENERGIES = data["ENERGIES"]
        
        data.close()
        
    SATISFIES_DATE_EXTENT = (start < EPOCH) & (EPOCH < end)
    FESA = FESA[SATISFIES_DATE_EXTENT, :]
    L = L[SATISFIES_DATE_EXTENT]
    MLT = MLT[SATISFIES_DATE_EXTENT]
    EPOCH = EPOCH[SATISFIES_DATE_EXTENT]
    
    FESA[FESA < 0] = np.NaN
    
    return DataRefContainer(FESA, L, MLT, EPOCH, ENERGIES)

