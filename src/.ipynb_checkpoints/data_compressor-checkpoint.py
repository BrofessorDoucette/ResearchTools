import scipy.interpolate
from field_models import model
from spacepy import pycdf
import datetime
import glob
import h5py
import numpy as np
import os
import os_helper
import re
import scipy
from netCDF4 import Dataset
import h5py
import spacepy
import matplotlib.pyplot as plt
from matplotlib import colors


def compress_month_rept_l2(satellite: str,
                           month: int, year: int,
                           make_dirs: bool = False,
                           raw_data_dir: str = "./../raw_data/",
                           compressed_data_dir: str = "./../compressed_data/") -> None:

    output_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "REPT", f"{year}")

    os_helper.verify_output_dir_exists(directory=output_dir,
                                       force_creation=make_dirs,
                                       hint="COMPRESSED REPT DIR")

    if month < 10:
        input_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "REPT", f"{year}/0{month}")
        output_file = os.path.join(output_dir, f"REPT_{year}0{month}_{satellite.upper()}.npz")

    else:
        input_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "REPT", f"{year}/{month}")
        output_file = os.path.join(output_dir, f"REPT_{year}{month}_{satellite.upper()}.npz")

    os_helper.verify_input_dir_exists(directory=input_dir,
                                      hint="RAW REPT DIR")

        
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
                                           root_dir=input_dir)
        
        if len(rept_cdf_path_or_empty) != 0:
            rept_cdf_path = os.path.join(input_dir, rept_cdf_path_or_empty[0])
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
    
    
def compress_year_poes_after_2012(satellite: str,
                                  year: int,
                                  make_dirs: bool = False,
                                  raw_data_dir: str = "./../raw_data/",
                                  compressed_data_dir: str = "./../compressed_data/") -> None:

    input_dir = os.path.join(os.path.abspath(raw_data_dir), "POES", satellite.lower())
    os_helper.verify_input_dir_exists(directory=input_dir,
                                      hint=f"POES RAW DIR: {satellite.lower()}")

    output_dir = os.path.join(os.path.abspath(compressed_data_dir), "POES", "DIRTY", satellite.lower())

    os_helper.verify_output_dir_exists(directory=output_dir,
                                       force_creation=make_dirs,
                                       hint=f"POES DIRTY DIR")
    
    output_file = os.path.join(output_dir, f"POES_{year}_{satellite.lower()}_DIRTY.npz")

    if satellite not in ["metop01", "metop02", "metop03", "noaa15", "noaa16", "noaa17", "noaa18", "noaa19"]:

        raise Exception("The compressor for satellites without SEM-2 Instrument Package is not yet implemented!")

    satID = satellite.lower().strip()
    
    poes_nc_files = glob.glob(pathname=f"poes_{satID[0]}{satID[-2:]}_{year}*.nc",
                              root_dir=input_dir)

    if len(poes_nc_files) == 0:
        print(f"No raw data files found to compress for: {year}")
        return

    #Time
    epoch = np.zeros(shape=0, dtype=datetime.datetime)
    
    #Coordinates
    alt = np.zeros(shape=(0), dtype=np.float32)
    lat = np.zeros(shape=(0), dtype=np.float32)
    lon = np.zeros(shape=(0), dtype=np.float32)
    L = np.zeros(shape=0, dtype=np.float32)
    mlt = np.zeros(shape=0, dtype=np.float32)
    
    #Flux
    mep_ele_tel0_flux_e1 = np.zeros(shape=(0), dtype=np.float32)
    mep_ele_tel0_flux_e2 = np.zeros(shape=(0), dtype=np.float32)
    
    #Pitch Angles
    meped_alpha_0_sat = np.zeros(shape=(0), dtype=np.float32)

    for poes_nc_file in sorted(poes_nc_files):

        print("Compressing file: ", poes_nc_file)

        poes_nc_path = os.path.join(input_dir, poes_nc_file)
        
        with Dataset(poes_nc_path, "r") as poes:
            
            if len(poes.variables["msec"]) == 0:
                continue
            
            #Time
            years = np.ma.MaskedArray.filled(poes.variables["year"][...], fill_value = np.NaN).astype(int)
            days = np.ma.MaskedArray.filled(poes.variables["day"][...], fill_value = np.NaN).astype(int)
            msecs = np.ma.MaskedArray.filled(poes.variables["msec"][...], fill_value = np.NaN).astype(np.int64)
            
            valid_dates = np.isfinite(years) & np.isfinite(days) & np.isfinite(msecs)
            years = years[valid_dates]
            days = days[valid_dates]
            msecs = msecs[valid_dates]
                  
            epoch_slice = np.array([datetime.datetime(year = years[i], month = 1, day = 1, hour = 0, minute = 0, second = 0, microsecond = 0) + datetime.timedelta(days = int(days[i]) - 1) + datetime.timedelta(milliseconds = int(msecs[i])) for i in range(len(years))])
                   
            #Coordinates
            alt_slice = np.ma.MaskedArray.filled(poes.variables["alt"][...], fill_value = np.NaN)[valid_dates]
            lat_slice = np.ma.MaskedArray.filled(poes.variables["lat"][...], fill_value = np.NaN)[valid_dates]
            lon_slice = np.ma.MaskedArray.filled(poes.variables["lon"][...], fill_value = np.NaN)[valid_dates]
            L_slice = np.ma.MaskedArray.filled(poes.variables["L_IGRF"][...], fill_value = np.NaN)[valid_dates]
            mlt_slice = np.ma.MaskedArray.filled(poes.variables["MLT"][...], fill_value = np.NaN)[valid_dates]
            
            #Fluxes
            mep_ele_tel0_flux_e1_slice = np.ma.MaskedArray.filled(poes.variables["mep_ele_tel0_flux_e1"][...], fill_value = np.NaN)[valid_dates]
            mep_ele_tel0_flux_e2_slice = np.ma.MaskedArray.filled(poes.variables["mep_ele_tel0_flux_e2"][...], fill_value = np.NaN)[valid_dates]

            #Pitch Angle
            meped_alpha_0_sat_slice = np.ma.MaskedArray.filled(poes.variables["meped_alpha_0_sat"][...], fill_value = np.NaN)[valid_dates]
            

            if (0 < len(epoch)) and (epoch_slice[0] < epoch[-1]):
                raise Exception("Concatenating POES files would lead to out of order Epoch.")
            
            epoch = np.concatenate((epoch, epoch_slice), axis = 0)
            alt = np.concatenate((alt, alt_slice), axis = 0)
            lat = np.concatenate((lat, lat_slice), axis = 0)
            lon = np.concatenate((lon, lon_slice), axis = 0)
            L = np.concatenate((L, L_slice), axis = 0)
            mlt = np.concatenate((mlt, mlt_slice), axis = 0)
            mep_ele_tel0_flux_e1 = np.concatenate((mep_ele_tel0_flux_e1, mep_ele_tel0_flux_e1_slice), axis = 0)
            mep_ele_tel0_flux_e2 = np.concatenate((mep_ele_tel0_flux_e2, mep_ele_tel0_flux_e2_slice), axis = 0)
            meped_alpha_0_sat = np.concatenate((meped_alpha_0_sat, meped_alpha_0_sat_slice), axis = 0)

    np.savez_compressed(output_file,
                        EPOCH = epoch,
                        ALT = alt,
                        LAT = lat,
                        LON = lon,
                        L = L,
                        MLT = mlt,
                        MEP_ELE_TEL0_FLUX_E1 = mep_ele_tel0_flux_e1,
                        MEP_ELE_TEL0_FLUX_E2 = mep_ele_tel0_flux_e2,
                        MEPED_ALPHA_0_SAT = meped_alpha_0_sat)

    print(f"Compressed POES Data for {satellite} during: {year}.\nSaved to: {output_file}\n")


def compress_poes_after_2012(satellite: str,
                             make_dirs: bool = False,
                             raw_data_dir: str = "./../raw_data/",
                             compressed_data_dir: str = "./../compressed_data/") -> None:

    input_dir = os.path.join(os.path.abspath(raw_data_dir), "POES", satellite.lower())

    os_helper.verify_input_dir_exists(directory = input_dir,
                                      hint = f"RAW POES DIR: {satellite}")

    possible_ncs_to_compress = glob.glob("*.nc", root_dir=input_dir)

    years_found: set[int] = set()

    for filename in possible_ncs_to_compress:

        re_match = re.search(pattern=r"([0-9]{8})", string=filename)

        if not re_match:
            continue

        date = re_match.group()

        years_found.add(int(date[:4]))

    for _year in sorted(list(years_found)):

        compress_year_poes_after_2012(satellite = satellite,
                                      year = _year,
                                      make_dirs = make_dirs,
                                      raw_data_dir = raw_data_dir,
                                      compressed_data_dir = compressed_data_dir)
        

def calculate_and_compress_psd(satellite: str,
                              field_model: model,
                              month: int, year: int,
                              make_dirs: bool = False,
                              raw_data_dir: str = "./../raw_data/",
                              compressed_data_dir: str = "./../compressed_data/",
                              debug_mode: bool = False,
                              verbose: bool = False) -> None:
    
    M_e = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]
    
    if(month == 12):
        start = datetime.datetime(year = year, month = month, day = 1)
        end = datetime.datetime(year = year + 1, month = 1, day = 1)
        
    else:
        start = datetime.datetime(year = year, month = month, day = 1)
        end = datetime.datetime(year = year, month = month + 1, day = 1)
        
    ect_data_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "ECT", "L3")
    
    os_helper.verify_input_dir_exists(ect_data_dir, hint="ECT DATA DIR")
    
    PSD = np.zeros((0, 35, 102), dtype=np.float64)
    EPOCH = np.zeros((0), dtype=datetime.datetime)
    JD = np.zeros((0), dtype=np.float64)
    ENERGIES = np.zeros((0, 102), dtype=np.float64)
    ALPHA = np.zeros((0, 35), dtype=np.float64)
    
    magephem_data_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "MAGEPHEM")
    
    os_helper.verify_input_dir_exists(magephem_data_dir, hint="MAGEPEHEM DATA DIR")

    K = np.zeros((0, 35), dtype=np.float64)
    L_STAR = np.zeros((0, 35), dtype=np.float64)
    L = np.zeros((0, 35), dtype=np.float64)

    IN_OUT = np.zeros((0), dtype=np.int32)
    ORBIT_NUMBER = np.zeros((0), dtype=np.int32)
    
    emfisis_data_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "EMFISIS", "L3")

    os_helper.verify_input_dir_exists(emfisis_data_dir, hint="EMFISIS DATA DIR")


    output_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "PSD")

    os_helper.verify_output_dir_exists(directory=output_dir,
                                       force_creation=make_dirs,
                                       hint="PSD DEPENDENCY OUTPUT DIR")
    
    if month < 10:
        output_file = os.path.join(output_dir, f"PSD_{year}0{month}_{satellite.upper()}_{field_model.name}.npz")
    else:
        output_file = os.path.join(output_dir, f"PSD_{year}{month}_{satellite.upper()}_{field_model.name}.npz")

    B = np.zeros((0), dtype=np.float64)
            
    curr = start
    while curr < end:
        
        _year = str(curr.year)
        _month = str(curr.month)
        _day = str(curr.day)
        
        if len(_month) < 2:
            _month = f"0{_month}"
        
        if len(_day) < 2:
            _day = f"0{_day}"
            
        ect_cdf = f"rbsp{satellite.lower()}_ect-elec-L3_{_year}{_month}{_day}*.cdf"
        ect_cdf_found = glob.glob(ect_cdf, root_dir=ect_data_dir)
        
        if len(ect_cdf_found) != 0:
            ect_cdf_path = os.path.join(ect_data_dir, ect_cdf_found[0])
        else:
            print(f"ECT CDF NOT FOUND: {os.path.join(ect_data_dir, ect_cdf)}. Skipping!")
            curr += datetime.timedelta(days = 1)   
            continue 
                
        match field_model:
            case field_model.TS04D:
                magephem_h5 = f"rbsp{satellite.lower()}_def_MagEphem_TS04D_{_year}{_month}{_day}*.h5"

            case field_model.T89D:
                magephem_h5 = f"rbsp{satellite.lower()}_def_MagEphem_T89D_{_year}{_month}{_day}*.h5"
                
        magephem_h5_found = glob.glob(magephem_h5, root_dir=magephem_data_dir)
        
        if len(magephem_h5_found) != 0:
            magphem_h5_path = os.path.join(magephem_data_dir, magephem_h5_found[0])
        else:
            print(f"MAGEPHEM H5 NOT FOUND: {os.path.join(magephem_data_dir, magephem_h5)}. Skipping!")
            curr += datetime.timedelta(days = 1)   
            continue
        
        emfisis_cdf = f"rbsp-{satellite.lower()}_magnetometer_1sec-gse_emfisis-l3_{_year}{_month}{_day}*.cdf"
        emfisis_cdf_found = glob.glob(emfisis_cdf, root_dir=emfisis_data_dir)
        
        if len(emfisis_cdf_found) != 0:
            emfisis_cdf_path = os.path.join(emfisis_data_dir, emfisis_cdf_found[0])
        else:
            print(f"EMFISIS CDF NOT FOUND: {os.path.join(emfisis_data_dir, emfisis_cdf)}. Skipping!")
            curr += datetime.timedelta(days = 1)   
            continue
        
        if debug_mode and verbose:
            print(f"Loading: {ect_cdf_path}")
            
        curr_ect = spacepy.pycdf.CDF(ect_cdf_path)
        
        curr_ect_fedu = curr_ect["FEDU"][...]
        curr_ect_fedu[curr_ect_fedu < 0] = np.NaN
        curr_ect_epoch = curr_ect["Epoch"][...]
        curr_ect_JD = spacepy.time.Ticktock(curr_ect_epoch, "UTC").getJD()
        
        curr_ect_fedu_energy = curr_ect["FEDU_Energy"][...]
        curr_ect_fedu_energy_delta_plus = curr_ect["FEDU_Energy_DELTA_plus"][...]
        curr_ect_fedu_energy_delta_minus = curr_ect["FEDU_Energy_DELTA_minus"][...]  
        
        valid_energy_channels = (0 < curr_ect_fedu_energy) & ((curr_ect_fedu_energy / 1000) < 10) & (curr_ect_fedu_energy_delta_plus > 0) & (curr_ect_fedu_energy_delta_minus > 0)    
            
        if not np.any(valid_energy_channels):
            
            print(f"All of the energy limits and energy channels were negative or zero, this file must be broken somehow! \nPath: {ect_cdf_path}. \nSkipping!")
            
            curr += datetime.timedelta(days = 1) 
            continue
        
        curr_valid_ect_fedu_energy = curr_ect_fedu_energy[valid_energy_channels]
        curr_valid_ect_fedu_energy_delta_plus = curr_ect_fedu_energy_delta_plus[valid_energy_channels]
        curr_valid_ect_fedu_energy_delta_minus = curr_ect_fedu_energy_delta_minus[valid_energy_channels]
        
        curr_energy_maximums = (curr_valid_ect_fedu_energy + curr_valid_ect_fedu_energy_delta_plus) / 1000.0
        curr_energy_minimums = (curr_valid_ect_fedu_energy - curr_valid_ect_fedu_energy_delta_minus) / 1000.0
        
        pc_squared = 0.5 * (curr_energy_minimums * (curr_energy_minimums + 2 * M_e) + curr_energy_maximums * (curr_energy_maximums + 2 * M_e)) 
        
        curr_PSD = np.full_like(curr_ect_fedu, np.NaN)
        curr_PSD[:, :, valid_energy_channels] = (curr_ect_fedu[:, :, valid_energy_channels] / pc_squared) * 1.66e-10 * 200.3 #CHEN 2005   
                
        curr_valid_energies = np.sqrt(curr_energy_maximums * curr_energy_minimums)

        curr_energies = np.full_like(curr_ect_fedu_energy, fill_value = np.NaN)
        curr_energies[valid_energy_channels] = curr_valid_energies
            
        curr_alpha = np.deg2rad(curr_ect["FEDU_Alpha"][...])
        
        sort_alpha = np.argsort(curr_alpha)  #indices to sort alpha
        sort_energies = np.argsort(curr_energies) #indices to sort energies
        
        curr_alpha = curr_alpha[sort_alpha]
        curr_energies = curr_energies[sort_energies]
        curr_PSD = curr_PSD[:, sort_alpha, :]
        curr_PSD = curr_PSD[:, :, sort_energies]
        
        if debug_mode and verbose:
            print(f"Loading: {emfisis_cdf_path}")
            
        curr_emfisis = spacepy.pycdf.CDF(emfisis_cdf_path)
        curr_emfisis_B = curr_emfisis["Magnitude"][...]
        if(curr_emfisis["Epoch"][...].shape[0] <= 0):
            print(f"Emfisis file had no data!, Skipping: {curr}")
            curr += datetime.timedelta(days = 1)
            continue
        
        curr_emfisis_JD = spacepy.time.Ticktock(curr_emfisis["Epoch"][...], "UTC").getJD()
        
        valid_B = (curr_emfisis["magInvalid"][...] == 0) & (curr_emfisis["magFill"][...] == 0) & (curr_emfisis["calState"][...] == 0)
        curr_emfisis_B = curr_emfisis_B[valid_B] / 100000.0 #Get B Field in Gauss
        curr_emfisis_JD = curr_emfisis_JD[valid_B]
        
        B = np.concatenate((B, np.interp(curr_ect_JD, curr_emfisis_JD, curr_emfisis_B, left=np.NAN, right=np.NaN)), axis = 0)
        
        ENERGIES = np.concatenate((ENERGIES, np.tile(curr_energies, (len(curr_ect_JD), 1))), axis = 0)
        ALPHA = np.concatenate((ALPHA, np.tile(curr_alpha, (len(curr_ect_JD), 1))), axis = 0)
                
        if not np.all(np.diff(curr_ect_JD) > 0):
            raise Exception("ect_JD needs to be strictly increasing to interpolate over time!")
        
        PSD = np.concatenate((PSD, curr_PSD), axis=0)
        EPOCH = np.concatenate((EPOCH, curr_ect_epoch), axis = 0)
        JD = np.concatenate((JD, curr_ect_JD), axis = 0)
                     
        if debug_mode and verbose:
            print(f"Loading: {magphem_h5_path}")
        
        curr_magephem = h5py.File(magphem_h5_path, "r")
        
        curr_magephem_k = curr_magephem["K"][...]
        curr_magephem_Lstar = curr_magephem["Lstar"][...]
        curr_magephem_L = curr_magephem["L"][...]
        curr_magephem_JD = curr_magephem["JulianDate"][...]
        curr_magephem_in_out = curr_magephem["InOut"][...]
        curr_magephem_orbit_number = curr_magephem["OrbitNumber"][...]
        curr_magephem_alpha = curr_magephem["Alpha"][...]
    
        curr_magephem_k[(curr_magephem_k < 0)] = np.NaN
        curr_magephem_Lstar[(curr_magephem_Lstar < 0)] = np.NaN
        curr_magephem_L[(curr_magephem_L < 0)] = np.NaN

        
        _, magephem_uniq = np.unique(curr_magephem_JD, return_index=True)
        curr_magephem_JD = curr_magephem_JD[magephem_uniq]
        curr_magephem_k = curr_magephem_k[magephem_uniq, :]
        curr_magephem_Lstar = curr_magephem_Lstar[magephem_uniq, :]
        curr_magephem_L = curr_magephem_L[magephem_uniq, :]
        curr_magephem_in_out = curr_magephem_in_out[magephem_uniq]
        curr_magephem_orbit_number = curr_magephem_orbit_number[magephem_uniq]
        
        curr_magephem_alpha = np.deg2rad(np.concatenate((np.flip(curr_magephem_alpha, axis=0), np.flip(curr_magephem_alpha, axis=0)[:-1] + 90), axis=0)) #We want alpha to go from 5 -> 90 -> 175 degrees        
        curr_magephem_k = np.concatenate((np.flip(curr_magephem_k, axis=1), curr_magephem_k[:, 1:]), axis=1)
        curr_magephem_Lstar = np.concatenate((np.flip(curr_magephem_Lstar, axis=1), curr_magephem_Lstar[:, 1:]), axis=1) 
        curr_magephem_L = np.concatenate((np.flip(curr_magephem_L, axis=1), curr_magephem_L[:, 1:]), axis=1) 

            
        K_interpolator = scipy.interpolate.RegularGridInterpolator(points = (curr_magephem_JD, curr_magephem_alpha), 
                                                                values = curr_magephem_k,
                                                                bounds_error = False, 
                                                                fill_value = np.NaN)
        
        Lstar_interpolator = scipy.interpolate.RegularGridInterpolator(points = (curr_magephem_JD, curr_magephem_alpha), 
                                                                    values = curr_magephem_Lstar,
                                                                    bounds_error = False, 
                                                                    fill_value = np.NaN)
        
        L_interpolator = scipy.interpolate.RegularGridInterpolator(points = (curr_magephem_JD, curr_magephem_alpha), 
                                                            values = curr_magephem_L,
                                                            bounds_error = False, 
                                                            fill_value = np.NaN)
        
        _x, _y = np.meshgrid(curr_ect_JD, curr_alpha, indexing="ij")
        
        K = np.concatenate((K, K_interpolator((_x, _y), method="linear")), axis = 0)
        L_STAR = np.concatenate((L_STAR, Lstar_interpolator((_x, _y), method="linear")), axis = 0)
        L = np.concatenate((L, L_interpolator((_x, _y), method="linear")), axis = 0)
        IN_OUT = np.concatenate((IN_OUT, np.int32(np.interp(curr_ect_JD, curr_magephem_JD, curr_magephem_in_out, left=np.NAN, right=np.NaN))), axis = 0)
        ORBIT_NUMBER = np.concatenate((ORBIT_NUMBER, np.int32(np.interp(curr_ect_JD, curr_magephem_JD, curr_magephem_orbit_number, left=np.NAN, right=np.NaN))), axis = 0)
                
        if(debug_mode):
            print(f"Successfully loaded all data for: {curr}")
        
        curr += datetime.timedelta(days = 1)          
        
    satisfies_timespan = (start < EPOCH) & (EPOCH < end)
    PSD = PSD[satisfies_timespan, :, :]
    EPOCH = EPOCH[satisfies_timespan]  
    JD = JD[satisfies_timespan]
    ENERGIES = ENERGIES[satisfies_timespan, :]
    ALPHA = ALPHA[satisfies_timespan, :]
    K = K[satisfies_timespan, :]
    L_STAR = L_STAR[satisfies_timespan, :]
    L = L[satisfies_timespan, :]
    IN_OUT = IN_OUT[satisfies_timespan]
    ORBIT_NUMBER = ORBIT_NUMBER[satisfies_timespan]
    B = B[satisfies_timespan]
    
    if not np.all(np.diff(JD) > 0):
        raise Exception("ect_JD needs to be strictly increasing to interpolate over time!")
        
    
    print(f"Saving: {output_file}")
    
    np.savez_compressed(output_file,
                        PSD = PSD,
                        EPOCH = EPOCH,
                        JD = JD,
                        ENERGIES = ENERGIES,
                        ALPHA = ALPHA,
                        K = K,
                        L_STAR = L_STAR,
                        L = L,
                        IN_OUT = IN_OUT,
                        ORBIT_NUMBER = ORBIT_NUMBER,
                        B = B)

def compress_emfisis_wna_survey_and_diagonal_spectral_matrix(satellite: str,
                                                             month: int, year: int,
                                                             make_dirs: bool = False,
                                                             raw_data_dir: str = "./../raw_data/",
                                                             compressed_data_dir: str = "./../compressed_data/"):
    
    input_dir_L2 = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "EMFISIS", "L2")
    input_dir_L4 = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "EMFISIS", "L4")

    os_helper.verify_input_dir_exists(directory = input_dir_L2,
                                      hint = "RAW EMFISIS L2 DIR")
    
    os_helper.verify_input_dir_exists(directory = input_dir_L4,
                                      hint = "RAW EMFISIS L4 DIR")
    
    output_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "EMFISIS", "CHORUS")

    os_helper.verify_output_dir_exists(directory = output_dir,
                                       force_creation = make_dirs,
                                       hint = "RBSP EMFISIS CHORUS DIR")
    
    if(month == 12):
        start = datetime.datetime(year = year, month = month, day = 1)
        end = datetime.datetime(year = year + 1, month = 1, day = 1)
        
    else:
        start = datetime.datetime(year = year, month = month, day = 1)
        end = datetime.datetime(year = year, month = month + 1, day = 1)
    
    _year = str(year)
    
    if len(str(month)) < 2:
        _month = f"0{month}"
    
    output_file = os.path.join(output_dir, f"EMFISIS_WNA_SURVEY_AND_DIAGONAL_SPECTRAL_MATRIX_{_year}{_month}_{satellite.upper()}.npz")
    
    WFR_bandwidths = np.zeros(shape=(0, 65), dtype=np.float32)
    
    curr = start
    while curr < end:
        
        _day = str(curr.day)
        
        if len(_day) < 2:
            _day = f"0{_day}"
                
        WFR_spectral_matrix_cdf_path_or_empty = glob.glob(pathname = f"rbsp-{satellite.lower()}_WFR-spectral-matrix-diagonal_emfisis-L2_{_year}{_month}{_day}*.cdf",
                                                          root_dir = input_dir_L2)
        
        if len(WFR_spectral_matrix_cdf_path_or_empty) != 0:
            WFR_spectral_matrix_cdf_path = os.path.join(input_dir_L2, WFR_spectral_matrix_cdf_path_or_empty[0])
        else:
            print(f"COULDN'T FIND WFR SPECTRAL MATRIX CDF FILE FOR DATE: {_month}/{_day}/{_year}. Skipping!")
            curr += datetime.timedelta(days = 1)
            continue
        
        WNA_survey_cdf_path_or_empty = glob.glob(pathname = f"rbsp-{satellite.lower()}_wna-survey_emfisis-L4_{_year}{_month}{_day}*.cdf",
                                                root_dir = input_dir_L4)
        
        if len(WNA_survey_cdf_path_or_empty) != 0:
            WNA_survey_cdf_path = os.path.join(input_dir_L4, WNA_survey_cdf_path_or_empty[0])
        else:
            print(f"COULDN'T FIND WNA SURVEY CDF FILE FOR DATE: {_month}/{_day}/{_year}. Skipping!")
            curr += datetime.timedelta(days = 1)
            continue
        
        WFR_spectral_matrix = pycdf.CDF(WFR_spectral_matrix_cdf_path)
        
        print(WFR_spectral_matrix["WFR_bandwidth"][...])

        
        print(WFR_spectral_matrix_cdf_path, WNA_survey_cdf_path)
    
        curr += datetime.timedelta(days = 1)
        
        
if __name__ == "__main__":

    #for month in range(1, 13):
    #    
    #    calculate_and_compress_psd(satellite="A", field_model=model.TS04D, month=month, year=2016, make_dirs=True, debug_mode=True)
        
    #calculate_and_compress_psd(satellite="B", field_model=model.TS04D, month=3, year=2015, make_dirs=True, debug_mode=True, verbose=False)
    
    #compress_emfisis_wna_survey_and_diagonal_spectral_matrix(satellite = "A",
    #                                                         month = 1,
    #                                                         year = 2013,
    #                                                         make_dirs = True)
    for sat in ["noaa17", "noaa18", "noaa19"]:

        compress_poes_after_2012(satellite = sat, make_dirs = True)