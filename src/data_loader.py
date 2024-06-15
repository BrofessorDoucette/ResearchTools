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
from field_models import model
import glob

def load_omni_data_1hour_res(start: datetime.datetime,
                             end: datetime.datetime,
                             raw_data_dir: str = "./../raw_data/") -> dict:
    
    _1_hour_res_dir = os.path.join(os.path.abspath(raw_data_dir), "OMNI", "_1_hour_res")
    
    os_helper.verify_input_dir_exists(directory = _1_hour_res_dir,
                                      hint = "1 HOUR OMNI INPUT DIR")
    
    DST = np.zeros(shape=0, dtype=np.int32)
    epoch = np.zeros(shape=0, dtype=datetime.datetime)
    
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
    
    satisfies_date_extent = (start < epoch) & (epoch < end)
    DST = DST[satisfies_date_extent]
    epoch = epoch[satisfies_date_extent]
    
    refs = {
        
        "DST": DST,
        "EPOCH": epoch
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
                              compressed_data_dir: str = "./../compressed_data/") -> REPTDataRefContainer:
    
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
            continue

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
    
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):

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