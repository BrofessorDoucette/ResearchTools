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

def load_psd_dependencies(satellite: str,
                          field_model: model,
                          start: datetime.datetime, end: datetime.datetime, 
                          compressed_data_dir: str = "./../compressed_data/"):
    
    psd_dependency_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "PSD_Dependencies")
    
    os_helper.verify_input_dir_exists(psd_dependency_dir, hint="PSD DEPENDENCY DIR")
    
    ect_fedu = np.zeros((0, 35, 102), dtype=np.float32)
    ect_epoch = np.zeros((0), dtype=datetime.datetime)
    ect_JD = np.zeros((0), dtype=np.float64)
    
    K = np.zeros((0, 35), dtype=np.float64)
    L_star = np.zeros((0, 35), dtype=np.float64)
    in_out = np.zeros((0), dtype=np.int32)
    orbit_number = np.zeros((0), dtype=np.int32)
    
    B = np.zeros((0), dtype=np.float64)
    
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):

        _year = str(dt.year)
        _month = str(dt.month)

        if len(_month) < 2:
            _month = f"0{_month}"
            
        file_name = f"PSD_DEPENDENCIES_{_year}{_month}_{satellite.upper()}_{field_model.name}.npz"
        
        psd_dependency_path = os.path.join(psd_dependency_dir, file_name)
        
        if not os.path.exists(psd_dependency_path):
            raise Exception(f"\nData file not found: {psd_dependency_path}!")
        
        
        print(f"Loading : {file_name}")
        data = np.load(psd_dependency_path, allow_pickle=True)
        
        
        ect_fedu = np.concatenate((ect_fedu, data["ECT_FEDU"]), axis = 0)
        ect_epoch = np.concatenate((ect_epoch, data["ECT_EPOCH"]), axis = 0)
        ect_JD = np.concatenate((ect_JD, data["ECT_JD"]), axis = 0)
        
        if i == 0:
            ect_fedu_energy = data["ECT_FEDU_ENERGY"]
            ect_fedu_energy_delta_plus = data["ECT_FEDU_ENERGY_DELTA_PLUS"]
            ect_fedu_energy_delta_minus = data["ECT_FEDU_ENERGY_DELTA_MINUS"]
            ect_fedu_alpha = data["ECT_FEDU_ALPHA"]
            magephem_alpha = data["MAGEPHEM_ALPHA"]

        
        K = np.concatenate((K, data["K"]), axis = 0)
        L_star = np.concatenate((L_star, data["LSTAR"]), axis = 0)
        in_out = np.concatenate((in_out, data["IN_OUT"]), axis = 0)
        orbit_number = np.concatenate((orbit_number, data["ORBIT_NUMBER"]), axis = 0)
                
        B = np.concatenate((B, data["B"]), axis = 0)
        
        data.close()
        
    satisfies_timespan = (start < ect_epoch) & (ect_epoch < end)
    ect_fedu = ect_fedu[satisfies_timespan, :, :]
    ect_JD = ect_JD[satisfies_timespan]
    ect_epoch = ect_epoch[satisfies_timespan]
    K = K[satisfies_timespan, :]
    L_star = L_star[satisfies_timespan, :]
    in_out = in_out[satisfies_timespan]
    orbit_number = orbit_number[satisfies_timespan]
    B = B[satisfies_timespan]
    
    loaded_data = {
        "ECT_FEDU" : ect_fedu,
        "ECT_JD" : ect_JD,
        "ECT_EPOCH" : ect_epoch,
        "ECT_FEDU_ENERGY" : ect_fedu_energy,
        "ECT_FEDU_ENERGY_DELTA_PLUS" : ect_fedu_energy_delta_plus,
        "ECT_FEDU_ENERGY_DELTA_MINUS" : ect_fedu_energy_delta_minus,
        "ECT_FEDU_ALPHA" : ect_fedu_alpha,
        "MAGEPHEM_ALPHA" : magephem_alpha,
        "K" : K,
        "LSTAR" : L_star,
        "IN_OUT" : in_out,
        "ORBIT_NUMBER" : orbit_number,
        "B" : B
    }
    
    return loaded_data
    

if __name__ == "__main__":
    
    load_psd_dependencies(satellite="A", field_model=model.TS04D,  start=datetime.datetime(year=2013, month=1, day=1), end=datetime.datetime(year=2013, month=1, day=31, hour=23, minute=59, second=59))