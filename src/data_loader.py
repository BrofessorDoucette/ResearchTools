from data_references import REPTDataRefContainer
from data_references import POESDataRefContainer
from dateutil import rrule
import datetime
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
import os_helper
from spacepy import pycdf
import glob


def load_omni_data(start: datetime.datetime, end: datetime.datetime,
                   omni_dir: str = "./../raw_data/OMNI/") -> pd.DataFrame:

    bz = np.zeros(shape=0, dtype=np.float32)
    ae_index = np.zeros(shape=0, dtype=np.int32)
    epoch = np.zeros(shape=0, dtype=np.object_)
        
    print(f"Loading OMNI data between: {start} and {end}")
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):
        
        _year = str(dt.year)
        _month = str(dt.month)
        
        if len(_month) < 2:
            _month = f"0{_month}"
            
        omni_data_dir = os.path.join(omni_dir, f"{_year}/")
        
        os_helper.verify_input_dir_exists(directory = omni_data_dir,
                                          hint = "OMNI DATA DIR")
        
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
        
    return pd.DataFrame(data={"Bz": bz, "AE": ae_index}, index=epoch)
    
              
def load_compressed_rept_data(satellite: str,
                              start: datetime.datetime, end: datetime.datetime,
                              rept_dir: str = "./../compressed_data/RBSP/") -> REPTDataRefContainer:
    
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
        
        rept_data_dir = os.path.join(os.path.abspath(rept_dir), f"{_year}/")
        
        os_helper.verify_input_dir_exists(directory = rept_data_dir,
                                          hint = "REPT DATA DIR")
        
        rept_file_name = f"REPT_{_year}{_month}_{satellite.upper()}.npz"
        rept_data_path = os.path.join(rept_data_dir, rept_file_name)

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
    
    return REPTDataRefContainer(fesa, L, mlt, epoch, energies)


def load_compressed_poes_data(satellite: str,
                              start: datetime.datetime, end: datetime.datetime,
                              poes_dir: str = "./../compressed_data/POES/CLEAN/") -> POESDataRefContainer:

    epoch = np.zeros(shape=0, dtype=datetime.datetime)
    mep_ele_flux = np.zeros(shape=(0, 2, 4), dtype=np.float32)
    L = np.zeros(shape=0, dtype=np.float32)
    mlt = np.zeros(shape=0, dtype=np.float32)
    naive_chorus_amplitudes = np.zeros(shape=0, dtype=np.float32)

    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):

        _year = str(dt.year)
        _month = str(dt.month)

        if len(_month) < 2:
            _month = f"0{_month}"


        poes_data_dir = os.path.join(os.path.abspath(poes_dir), f"{satellite}")

        os_helper.verify_input_dir_exists(directory=poes_data_dir,
                                          hint="POES DATA DIR")

        poes_file_name = f"POES_{_year}{_month}_{satellite.lower()}_CLEAN.npz"
        poes_data_path = os.path.join(poes_data_dir, poes_file_name)

        if not os.path.exists(poes_data_path):
            print(f"\nData file not found: {poes_data_path}, continuing...!")

        print(f"Loading : {poes_file_name}")
        data = np.load(poes_data_path, allow_pickle=True)

        epoch = np.concatenate((epoch, data["EPOCH"]), axis=0)
        mep_ele_flux = np.concatenate((mep_ele_flux, data["MEP_ELE_FLUX"]), axis=0)
        L = np.concatenate((L, data["L"]), axis=0)
        mlt = np.concatenate((mlt, data["MLT"]), axis=0)
        naive_chorus_amplitudes = np.concatenate((naive_chorus_amplitudes, data["NAIVE_CHORUS_AMPLITUDES"]), axis=0)

        data.close()

    satisfies_date_extent = (start < epoch) & (epoch < end)
    epoch = epoch[satisfies_date_extent]
    mep_ele_flux = mep_ele_flux[satisfies_date_extent, :, :]
    L = L[satisfies_date_extent]
    mlt = mlt[satisfies_date_extent]
    naive_chorus_amplitudes = naive_chorus_amplitudes[satisfies_date_extent]

    return POESDataRefContainer(epoch, mep_ele_flux, L, mlt, naive_chorus_amplitudes, satellite)



