from spacepy import pycdf
import spacepy
import scipy
from field_models import model
import h5py
import glob
import os
import os_helper
import datetime
import numpy as np
import re


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
    
    
def compress_month_poes(satellite: str,
                        month: int, year: int,
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

    _year = str(year)
    if month < 10:
        _month = f"0{month}"
    else:
        _month = str(month)

    output_file = os.path.join(output_dir, f"POES_{_year}{_month}_{satellite.lower()}_DIRTY.npz")

    if satellite not in ["metop1", "metop2", "noaa15", "noaa16", "noaa17", "noaa18", "noaa19"]:

        raise Exception("The compressor for satellites without SEM-2 Instrument Package is not yet implemented!")

    poes_cdf_files = glob.glob(pathname=f"{satellite.lower()}_poes-sem2_fluxes-2sec_{_year}{_month}*.cdf",
                                root_dir=input_dir)

    if len(poes_cdf_files) == 0:
        print(f"No raw data files found to compress for: {_year}-{_month}")
        return

    epoch = np.zeros(shape=0, dtype=datetime.datetime)
    mep_ele_flux = np.zeros(shape=(0, 2, 4), dtype=np.float32)
    L = np.zeros(shape=0, dtype=np.float32)
    mlt = np.zeros(shape=0, dtype=np.float32)

    for poes_cdf_file in sorted(poes_cdf_files):

        print("Compressing file: ", poes_cdf_file)

        poes_cdf_path = os.path.join(input_dir, poes_cdf_file)
        poes = pycdf.CDF(poes_cdf_path)

        if len(poes["Epoch"]) == 0:
            continue

        if (len(epoch) > 0) and (poes["Epoch"][0] < epoch[-1]):
            raise Exception("Concatenating POES files would lead to out of order Epoch.")

        epoch = np.concatenate((epoch, poes["Epoch"][...]), axis=0)
        mep_ele_flux = np.concatenate((mep_ele_flux, poes["mep_ele_flux"][...]), axis=0)
        L = np.concatenate((L, poes["l_igrf"][...]), axis=0)
        mlt = np.concatenate((mlt, poes["mlt"][...]), axis=0)

    np.savez_compressed(output_file, EPOCH=epoch, MEP_ELE_FLUX=mep_ele_flux, L=L, MLT=mlt)

    print(f"Compressed POES Data for {satellite} during: {_year}-{_month}.\nSaved to: {output_file}\n")


def compress_poes(satellite: str,
                  make_dirs: bool = False,
                  raw_data_dir: str = "./../raw_data/",
                  compressed_data_dir: str = "./../compressed_data/") -> None:

    input_dir = os.path.join(os.path.abspath(raw_data_dir), "POES", satellite.lower())

    os_helper.verify_input_dir_exists(directory = input_dir,
                                      hint = f"RAW POES DIR: {satellite}")

    possible_cdfs_to_compress = glob.glob("*.cdf", root_dir=input_dir)

    year_month_pairs: set[tuple[int, int]] = set()

    for filename in possible_cdfs_to_compress:

        re_match = re.search(pattern=r"([0-9]{8})", string=filename)

        if not re_match:
            continue

        date = re_match.group()

        year_month_pairs.add((int(date[:4]), int(date[4:6])))

    for _year, _month in sorted(list(year_month_pairs)):

        compress_month_poes(satellite = satellite,
                            month=_month, year = _year,
                            make_dirs = make_dirs,
                            raw_data_dir = raw_data_dir,
                            compressed_data_dir = compressed_data_dir)


def compress_psd_dependencies(satellite: str,
                              field_model: model,
                              month: int, year: int,
                              make_dirs: bool = False,
                              raw_data_dir: str = "./../raw_data/",
                              compressed_data_dir: str = "./../compressed_data/",
                              debug_mode: bool = False) -> None:
    
    if(month == 12):
        start = datetime.datetime(year = year, month = month, day = 1)
        end = datetime.datetime(year = year + 1, month = 1, day = 1)
        
    else:
        start = datetime.datetime(year = year, month = month, day = 1)
        end = datetime.datetime(year = year, month = month + 1, day = 1)
        
    ect_data_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "ECT", "L3")
    
    os_helper.verify_input_dir_exists(ect_data_dir, hint="ECT DATA DIR")
    
    ect_fedu = np.zeros((0, 35, 102), dtype=np.float32)
    ect_epoch = np.zeros((0), dtype=datetime.datetime)
    ect_JD = np.zeros((0), dtype=np.float64)
    
    magephem_data_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "MAGEPHEM")
    
    os_helper.verify_input_dir_exists(magephem_data_dir, hint="MAGEPEHEM DATA DIR")

    magephem_k = np.zeros((0, 18), dtype=np.float64)
    magephem_Lstar = np.zeros((0, 18), dtype=np.float64)
    magephem_JD = np.zeros((0), dtype=np.float64)
    magephem_in_out = np.zeros((0), dtype=np.int32)
    magephem_orbit_number = np.zeros((0), dtype=np.int32)
    
    emfisis_data_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "EMFISIS")

    os_helper.verify_input_dir_exists(emfisis_data_dir, hint="EMFISIS DATA DIR")

    _year = str(year)
    if month < 10:
        _month = f"0{month}"
    else:
        _month = str(month)

    output_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "PSD_DEPENDENCIES")

    os_helper.verify_output_dir_exists(directory=output_dir,
                                       force_creation=make_dirs,
                                       hint="PSD DEPENDENCY OUTPUT DIR")

    output_file = os.path.join(output_dir, f"PSD_DEPENDENCIES_{_year}{_month}_{satellite.upper()}_{field_model.name}.npz")

    emfisis_B = np.zeros((0), dtype=np.float64)
    emfisis_B_invalid = np.zeros((0), dtype=np.int8)
    emfisis_B_filled = np.zeros((0), dtype=np.int8)
    emfisis_B_calibration = np.zeros((0), dtype=np.int8)
    emfisis_JD = np.zeros((0), dtype=np.float64)
            
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
            raise Exception(f"ECT CDF NOT FOUND: {os.path.join(ect_data_dir, ect_cdf)}")        
        
        ect = spacepy.pycdf.CDF(ect_cdf_path)
        ect_fedu = np.concatenate((ect_fedu, ect["FEDU"][...]), axis=0)
        ect_epoch = np.concatenate((ect_epoch, ect["Epoch"][...]), axis=0)     
                
        match field_model:
            case field_model.TS04D:
                magephem_h5 = f"rbsp{satellite.lower()}_def_MagEphem_TS04D_{_year}{_month}{_day}*.h5"

            case field_model.T89D:
                magephem_h5 = f"rbsp{satellite.lower()}_def_MagEphem_T89D_{_year}{_month}{_day}*.h5"
                
        magephem_h5_found = glob.glob(magephem_h5, root_dir=magephem_data_dir)
        
        if len(magephem_h5_found) != 0:
            magphem_h5_path = os.path.join(magephem_data_dir, magephem_h5_found[0])
        else:
            raise Exception(f"MAGEPHEM H5 NOT FOUND: {os.path.join(magephem_data_dir, magephem_h5)}")
                
        magephem = h5py.File(magphem_h5_path, "r")
        magephem_k = np.concatenate((magephem_k, magephem["K"][...]), axis=0)
        magephem_Lstar = np.concatenate((magephem_Lstar, magephem["Lstar"][...]), axis=0)
        magephem_JD = np.concatenate((magephem_JD, magephem["JulianDate"][...]), axis=0)
        magephem_in_out = np.concatenate((magephem_in_out, magephem["InOut"][...]), axis=0)
        magephem_orbit_number = np.concatenate((magephem_orbit_number, magephem["OrbitNumber"][...]), axis=0)
        
        emfisis_cdf = f"rbsp-{satellite.lower()}_magnetometer_1sec-gse_emfisis-l3_{_year}{_month}{_day}*.cdf"
        emfisis_cdf_found = glob.glob(emfisis_cdf, root_dir=emfisis_data_dir)
        
        if len(emfisis_cdf_found) != 0:
            emfisis_cdf_path = os.path.join(emfisis_data_dir, emfisis_cdf_found[0])
        else:
            raise Exception(f"EMFISIS CDF NOT FOUND: {os.path.join(emfisis_data_dir, emfisis_cdf)}")
        
        emfisis = spacepy.pycdf.CDF(emfisis_cdf_path)
        emfisis_B = np.concatenate((emfisis_B, emfisis["Magnitude"][...].astype(np.float64)), axis=0)
        emfisis_B_invalid = np.concatenate((emfisis_B_invalid, emfisis["magInvalid"][...]), axis=0)
        emfisis_B_filled = np.concatenate((emfisis_B_filled, emfisis["magFill"][...]), axis=0)
        emfisis_B_calibration = np.concatenate((emfisis_B_calibration, emfisis["calState"][...]), axis=0)
        emfisis_JD = np.concatenate((emfisis_JD, spacepy.time.Ticktock(emfisis["Epoch"][...], "UTC").getJD()), axis=0)
        
        if(debug_mode):
            print(f"Compressed data for: {curr}")
        
        curr += datetime.timedelta(days = 1)          
        
    satisfies_timespan = (start < ect_epoch) & (ect_epoch < end)
    ect_fedu = ect_fedu[satisfies_timespan, :, :]
    
    ect_epoch = ect_epoch[satisfies_timespan]  
    
    ect_JD = spacepy.time.Ticktock(ect_epoch, "UTC").getJD()
    
    ect_fedu_alpha = np.deg2rad(ect["FEDU_Alpha"])
    ect_fedu_energy = ect["FEDU_Energy"][...]
    ect_fedu_energy_delta_plus = ect["FEDU_Energy_DELTA_plus"][...]
    ect_fedu_energy_delta_minus = ect["FEDU_Energy_DELTA_minus"][...]    

    magephem_alpha = magephem["Alpha"][...]
    
    magephem_k[(magephem_k < 0)] = np.NaN
    magephem_Lstar[(magephem_Lstar < 0)] = np.NaN
    
    _, magephem_uniq = np.unique(magephem_JD, return_index=True)
    magephem_JD = magephem_JD[magephem_uniq]
    magephem_k = magephem_k[magephem_uniq, :]
    magephem_Lstar = magephem_Lstar[magephem_uniq, :]
    magephem_in_out = magephem_in_out[magephem_uniq]
    magephem_orbit_number = magephem_orbit_number[magephem_uniq]
    
    magephem_alpha = np.deg2rad(np.concatenate((np.flip(magephem_alpha, axis=0), np.flip(magephem_alpha, axis=0)[:-1] + 90), axis=0)) #We want alpha to go from 5 -> 90 -> 175 degrees
    magephem_k = np.concatenate((np.flip(magephem_k, axis=1), magephem_k[:, 1:]), axis=1)
    magephem_Lstar = np.concatenate((np.flip(magephem_Lstar, axis=1), magephem_Lstar[:, 1:]), axis=1) 
        
    K_interpolator = scipy.interpolate.RegularGridInterpolator((magephem_JD, magephem_alpha), magephem_k) #Might need to fill in the internal nans here idk..
    Lstar_interpolator = scipy.interpolate.RegularGridInterpolator((magephem_JD, magephem_alpha), magephem_Lstar)
    _x, _y = np.meshgrid(ect_JD, ect_fedu_alpha, indexing="ij")
    K = K_interpolator((_x, _y), method="linear")
    L_star = Lstar_interpolator((_x, _y), method="linear")
    in_out = np.int32(np.interp(ect_JD, magephem_JD, magephem_in_out, left=np.NAN, right=np.NaN))
    orbit_number = np.int32(np.interp(ect_JD, magephem_JD, magephem_orbit_number, left=np.NAN, right=np.NaN))
    
    valid_B = (emfisis_B_invalid == 0) & (emfisis_B_filled == 0) & (emfisis_B_calibration == 0)
    emfisis_JD = emfisis_JD[valid_B]
    emfisis_B = emfisis_B[valid_B] / 100000 #Get B Field in Gauss
    B = np.interp(ect_JD, emfisis_JD, emfisis_B, left=np.NAN, right=np.NaN)
    
    np.savez_compressed(output_file,
                        ECT_FEDU = ect_fedu,
                        ECT_JD = ect_JD,
                        ECT_EPOCH = ect_epoch,
                        ECT_FEDU_ENERGY = ect_fedu_energy,
                        ECT_FEDU_ENERGY_DELTA_PLUS = ect_fedu_energy_delta_plus,
                        ECT_FEDU_ENERGY_DELTA_MINUS = ect_fedu_energy_delta_minus,
                        ECT_FEDU_ALPHA = ect_fedu_alpha,
                        MAGEPHEM_ALPHA = magephem_alpha,
                        K = K,
                        LSTAR = L_star,
                        IN_OUT = in_out,
                        ORBIT_NUMBER = orbit_number,
                        B = B)



if __name__ == "__main__":

    compress_psd_dependencies(satellite="A", field_model=model.TS04D, month=12, year=2013, make_dirs=True, debug_mode=True)
