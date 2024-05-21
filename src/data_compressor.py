from spacepy import pycdf
import glob
import os
import os_helper
import datetime
import numpy as np
import re


def compress_month_rept_l2(satellite: str,
                           month: int, year: int,
                           make_dirs: bool = False,
                           raw_data_dir: str = "./../raw_data/REPT/",
                           compressed_data_dir: str = "./../compressed_data/REPT/") -> None:
    
    output_dir = os.path.join(os.path.abspath(compressed_data_dir), f"{year}")
    
    os_helper.verify_output_dir_exists(directory = output_dir,
                                       force_creation = make_dirs,
                                       hint="COMPRESSED REPT DIR")
    
    if month < 10:
        uncompressed_dir = os.path.join(os.path.abspath(raw_data_dir), f"{year}/0{month}")
        output_file = os.path.join(output_dir, f"REPT_{year}0{month}_{satellite.upper()}.npz")
    else:
        uncompressed_dir = os.path.join(os.path.abspath(raw_data_dir), f"{year}/{month}")
        output_file = os.path.join(output_dir, f"REPT_{year}{month}_{satellite.upper()}.npz")
        
    os_helper.verify_input_dir_exists(directory = uncompressed_dir,
                                      hint = "RAW REPT DIR")
        
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
    
    
def compress_month_poes(satellite: str,
                        month: int, year: int,
                        make_dirs: bool = False,
                        raw_data_dir: str ="./../raw_data/POES/",
                        compressed_data_dir: str ="./../compressed_data/POES/DIRTY/") -> None:

    input_dir = os.path.join(os.path.abspath(raw_data_dir), satellite)
    os_helper.verify_input_dir_exists(directory=input_dir,
                                      hint=f"POES RAW DIR: {satellite}")

    output_dir = os.path.join(os.path.abspath(compressed_data_dir), satellite)

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
                  raw_data_dir: str = "./../raw_data/POES/",
                  compressed_data_dir: str = "./../compressed_data/POES/DIRTY") -> None:

    input_dir = os.path.join(os.path.abspath(raw_data_dir), satellite)

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


if __name__ == "__main__":

    compress_poes(satellite="noaa19",
                  make_dirs=True)
