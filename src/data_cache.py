import datetime
from field_models import model
import os
import os_helper
import numpy as np


def psd_cache_file_name(satellite: str,
                        start: datetime.datetime,
                        end: datetime.datetime,
                        mu: float,
                        k: float,
                        field_model: model) -> str:
    
    start_unix_epoch = start.timestamp()
    end_unix_epoch = end.timestamp()
        
    return f"{satellite.upper()}_{int(start_unix_epoch * 100000)}_{int(end_unix_epoch * 100000)}_{int(mu * 100000)}_{int(k * 100000)}_{field_model.name}.npz"
    


def psd_cache_exists(satellite: str,
                     start: datetime.datetime,
                     end: datetime.datetime,
                     mu: float,
                     k: float,
                     field_model: model,
                     compressed_data_dir: str = "../compressed_data/") -> bool:
    
    '''Use this to find out if a cached file exists to load from'''
    
    
    cache_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "PSD_SELECTION_CACHE")

    os_helper.verify_input_dir_exists(directory = cache_dir,
                                      hint = "PSD CACHE DIR!")
    
    expected_file_name = psd_cache_file_name(satellite = satellite.upper(),
                                             start = start,
                                             end = end,
                                             mu = mu,
                                             k = k,
                                             field_model = field_model)
    
    expected_file_path = os.path.join(cache_dir, expected_file_name)
    
    if not os.path.exists(expected_file_path):
        return False
    
    return True
    

def cache_psd_at_selected_mu_and_k(refs: dict,
                                   satellite: str,
                                   start: datetime.datetime,
                                   end: datetime.datetime,
                                   mu: float,
                                   k: float,
                                   field_model: model,
                                   make_dirs: bool = False,
                                   compressed_data_dir: str = "../compressed_data/") -> None:
    
    
    '''Use this to save the selected psd refs in a cache for later use. Always rewrites any existing cache that exists.'''
    
    
    
    
    cache_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "PSD_SELECTION_CACHE")
    
    os_helper.verify_output_dir_exists(directory = cache_dir,
                                       force_creation = make_dirs,
                                       hint = "PSD CACHE DIR!")
    
    output_file_name = psd_cache_file_name(satellite = satellite.upper(),
                                           start = start,
                                           end = end,
                                           mu = mu,
                                           k = k,
                                           field_model = field_model)
    
    output_path = os.path.join(cache_dir, output_file_name)
    
    np.savez(output_path, EPOCH = refs["EPOCH"], PSD = refs["PSD"], L_STAR = refs["L_STAR"], L = refs["L"], IN_OUT = refs["IN_OUT"],  ORBIT_NUMBER = refs["ORBIT_NUMBER"])
    

def load_psd_cache(satellite: str,
                   start: datetime.datetime,
                   end: datetime.datetime,
                   mu: float,
                   k: float,
                   field_model: model,
                   compressed_data_dir: str = "../compressed_data/") -> dict:
    
    '''Really unsure if this should be data_loader or in this file. I'll put it here for now. Self-explanatory, loads the cached results for quick access.'''
    
    
    if not psd_cache_exists(satellite = satellite.upper(),
                            start = start,
                            end = end,
                            mu = mu, 
                            k = k,
                            field_model = field_model):
                
        raise Exception("Tried to load a psd cache but it didn't exist!")
    
    cache_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "PSD_SELECTION_CACHE")
    
    os_helper.verify_input_dir_exists(directory = cache_dir,
                                      hint = "PSD CACHE DIR!")
    
    input_file_name = psd_cache_file_name(satellite = satellite.upper(),
                                           start = start,
                                           end = end,
                                           mu = mu,
                                           k = k,
                                           field_model = field_model)
    
    input_path = os.path.join(cache_dir, input_file_name)
    
    data = np.load(input_path, allow_pickle=True)
    
    refs = {
        
        "EPOCH" : data["EPOCH"],
        "PSD" : data["PSD"],
        "L_STAR" : data["L_STAR"],
        "L" : data["L"],
        "IN_OUT" : data["IN_OUT"],
        #"ORBIT_NUMBER" : data["ORBIT_NUMBER"]
        
    }
    
    data.close()
    
    return refs
    