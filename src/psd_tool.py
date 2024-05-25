import scipy.interpolate
from field_models import model
import datetime
import glob
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
import scipy
import spacepy.pycdf

M_e = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]

def k_from_alpha(x, A, B, C):
    
    return A * (x - np.pi/2) ** 4 + B * (x - np.pi/2) ** 2 + C

def k_from_alpha_shifted(x, A, B, C, K):
    
    return k_from_alpha(x, A, B, C) - K

def k_from_alpha_prime(x, A, B, C, K):
    
    return 4 * A * ((x - np.pi/2) ** 3) + 2 * B * (x - np.pi/2)

def k_from_alpha_prime2(x, A, B, C, K):
    
    return 12 * A * ((x - np.pi/2) ** 2) + 2 * B

def energy_from_mu(E, mu, alpha, B):
    
    return E**2 + 2 * M_e * E - ((2 * M_e * B * mu) / (np.sin(alpha) ** 2))
    
def energy_from_mu_prime(E, mu, alpha, B):
    
    return 2 * E  + 2 * M_e

def energy_from_mu_prime2(E, mu, alpha, B):
    
    return 2

def calculate_psd(start: datetime.datetime,
                  end: datetime.datetime,
                  chosen_mu: float,
                  chosen_k: float,
                  sat: str = "A",
                  field_model: model = model.TS04D,
                  energy_channels_to_remove_from_end: int = 5,
                  raw_data_dir: str = "../raw_data/",
                  debug_mode: bool = False,
                  verbose: bool = False) -> None:
    
    '''This version does linear interpolation across time for every pitch angle and energy.'''
    
    ect_data_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "ECT", "L3")
    
    ect_fedu = np.zeros((0, 35, 102), dtype=np.float32)
    ect_epoch = np.zeros((0), dtype=datetime.datetime)
    
    magephem_data_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "MAGEPHEM")

    magephem_k = np.zeros((0, 18), dtype=np.float64)
    magephem_Lstar = np.zeros((0, 18), dtype=np.float64)
    magephem_JD = np.zeros((0), dtype=np.float64)
    magephem_in_out = np.zeros((0), dtype=np.int32)
    magephem_orbit_number = np.zeros((0), dtype=np.int32)
    
    emfisis_data_dir = os.path.join(os.path.abspath(raw_data_dir), "RBSP", "EMFISIS")

    emfisis_B = np.zeros((0), dtype=np.float64)
    emfisis_B_invalid = np.zeros((0), dtype=np.int8)
    emfisis_B_filled = np.zeros((0), dtype=np.int8)
    emfisis_B_calibration = np.zeros((0), dtype=np.int8)
    emfisis_epoch = np.zeros((0), dtype=datetime.datetime)
    
    curr = start
    while curr < end:
        
        year = str(curr.year)
        month = str(curr.month)
        day = str(curr.day)
        
        if len(month) < 2:
            month = f"0{month}"
        
        if len(day) < 2:
            day = f"0{day}"
            
        ect_cdf = f"rbsp{sat.lower()}_ect-elec-L3_{year}{month}{day}*.cdf"
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
                magephem_h5 = f"rbsp{sat.lower()}_def_MagEphem_TS04D_{year}{month}{day}*.h5"

            case field_model.T89D:
                magephem_h5 = f"rbsp{sat.lower()}_def_MagEphem_T89D_{year}{month}{day}*.h5"
                
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
        
        emfisis_cdf = f"rbsp-{sat.lower()}_magnetometer_1sec-gse_emfisis-l3_{year}{month}{day}*.cdf"
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
        emfisis_epoch = np.concatenate((emfisis_epoch, emfisis["Epoch"][...]), axis=0)
        
        if(debug_mode):
            print(f"Loaded data for: {curr}")
        
        curr += datetime.timedelta(days = 1)        
        
    ect_JD = spacepy.time.Ticktock(ect_epoch, "UTC").getJD()
    
    ect_fedu_alpha = np.deg2rad(ect["FEDU_Alpha"])
    ect_fedu_energy = ect["FEDU_Energy"][...]
    ect_fedu_energy_delta_plus = ect["FEDU_Energy_DELTA_plus"][...]
    ect_fedu_energy_delta_minus = ect["FEDU_Energy_DELTA_minus"][...]

    ect_fedu[(ect_fedu < 0)] = np.NaN
    ect_fedu = ect_fedu[:, :, :-1*energy_channels_to_remove_from_end]
    energy_maximums = (ect_fedu_energy + ect_fedu_energy_delta_plus)[:-1*energy_channels_to_remove_from_end] / 1000.0
    energy_minimums = (ect_fedu_energy - ect_fedu_energy_delta_minus)[:-1*energy_channels_to_remove_from_end] / 1000.0
    ect_energies = np.sqrt(energy_maximums * energy_minimums)

    pc_squared = 0.5 * (energy_minimums * (energy_minimums + 2 * M_e) + energy_maximums * (energy_maximums + 2 * M_e))        
    PSD = (ect_fedu / pc_squared) * 1.66e-10 * 200.3 #CHEN 2005
    
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
    in_out = np.interp(ect_JD, magephem_JD, magephem_in_out, left=np.NAN, right=np.NaN)
    orbit_number = np.interp(ect_JD, magephem_JD, magephem_orbit_number, left=np.NAN, right=np.NaN)
    
    emfisis_JD = spacepy.time.Ticktock(emfisis_epoch, dtype="UTC").getJD()
    valid_B = (emfisis_B_invalid == 0) & (emfisis_B_filled == 0) & (emfisis_B_calibration == 0)
    emfisis_JD = emfisis_JD[valid_B]
    emfisis_B = emfisis_B[valid_B] / 100000 #Get B Field in Gauss
    B = np.interp(ect_JD, emfisis_JD, emfisis_B, left=np.NAN, right=np.NaN)
    
    if(debug_mode):
        num_failed_from_k = 0
        num_failed_from_L_star = 0
        
    for A in range(len(ect_fedu_alpha)):
        for E in range(len(ect_energies)):
            not_nan = np.isfinite(PSD[:, A, E])
            if np.any(not_nan):
                PSD[:, A, E] = np.interp(ect_JD, ect_JD[not_nan], PSD[not_nan, A, E], left=np.NaN, right = np.NaN)
    
    extracted_JD = []
    extracted_Lstar = []
    extracted_PSD = []
    extracted_in_out = []
    extracted_orbit_number = []
    
    for T in range(len(ect_JD)):
        
        not_nan_and_less_than_max_k = np.isfinite(K[T, :]) & (K[T, :] < 0.6)
        
        if not np.any(not_nan_and_less_than_max_k):
            continue

        try:
            K_popt, K_pcov = scipy.optimize.curve_fit(k_from_alpha, 
                                                      xdata=magephem_alpha[not_nan_and_less_than_max_k], 
                                                      ydata=K[T, not_nan_and_less_than_max_k], 
                                                      p0 = [0.3, 0.25, 0.02])
        except Exception as e:
            
            if(debug_mode):
                raise Exception(f"Unable to fit K for one value! Error: {e}")
            else:
                continue
        
        if(K_popt[2] > chosen_k):
            
            if(debug_mode):
                num_failed_from_k += 1
                
            continue
        
        try:
            selected_alpha = scipy.optimize.newton(func=k_from_alpha_shifted, 
                                                   fprime=k_from_alpha_prime, 
                                                   fprime2=k_from_alpha_prime2,
                                                   x0=np.pi/2 - 0.5, 
                                                   args=(K_popt[0], K_popt[1], K_popt[2], chosen_k), 
                                                   tol=1e-3)
            
            if (selected_alpha  < 0) or (np.pi < selected_alpha):
                raise Exception("An alpha value was less than 0 or greater than pi. :(")
            
        except Exception as e:
            
            xfit = np.linspace(0, np.pi, 100)
            plt.plot(xfit, k_from_alpha(xfit, *K_popt))
            plt.show()
            
            if(debug_mode):
                raise Exception(f"Unable to invert K to alpha! K: {chosen_k}, Error: {e}")
            else:
                continue
            
        
        if(debug_mode and verbose):
            print(f"Chosen K: {chosen_k}, Selected Alpha: {selected_alpha}")
            
        try: 
            selected_E = scipy.optimize.newton(func=energy_from_mu,
                                               fprime=energy_from_mu_prime,
                                               fprime2=energy_from_mu_prime2,
                                               x0 = 3.4,
                                               args = (chosen_mu, selected_alpha, B[T]),
                                               tol=1e-3)
        except Exception as e:
            if(debug_mode):
                raise Exception(f"Unable to select E from chosen mu: {chosen_mu}. Error: {e}")
            else:
                continue
        
        if(debug_mode and verbose):
            print(f"Chosen Mu: {chosen_mu}, Selected E: {selected_E}")
            
        not_nan = np.isfinite(L_star[T, :])
    
        if not np.any(not_nan):
            continue
        
        selected_L_star = np.interp(selected_alpha, magephem_alpha[not_nan], L_star[T, not_nan], left=np.NaN, right = np.NaN)
        
        if(debug_mode and verbose):
            print(f"Selected L_star: {selected_L_star}")
            
        if np.isnan(selected_L_star):
            if(debug_mode):
                num_failed_from_L_star += 1
            continue
        
        psd_per_e_at_selected_alpha = []
        
        for E in range(PSD.shape[2]):
        
            NOT_NAN = np.isfinite(PSD[T, :, E])
            if np.any(NOT_NAN):
                
                psd_per_e_at_selected_alpha.append(np.interp(selected_alpha, ect_fedu_alpha[NOT_NAN], PSD[T, NOT_NAN, E], left=np.NaN, right = np.NaN))
                
            else:
                psd_per_e_at_selected_alpha.append(np.NaN)
                
        psd_per_e_at_selected_alpha = np.array(psd_per_e_at_selected_alpha)
        not_nan = np.isfinite(psd_per_e_at_selected_alpha)
        
        if not np.any(not_nan):
            continue
        
        selected_psd = np.interp(selected_E, ect_energies[not_nan], psd_per_e_at_selected_alpha[not_nan], left=np.NaN, right = np.NaN)
        
        extracted_JD.append(ect_JD[T])
        extracted_Lstar.append(selected_L_star)
        extracted_PSD.append(selected_psd)
        extracted_in_out.append(in_out[T])
        extracted_orbit_number.append(orbit_number[T])
        
    if(debug_mode):
        print(f"Number failed: From K: {num_failed_from_k}")
        print(f"Number failed: From L_star: {num_failed_from_L_star}")

    return np.array(extracted_JD), np.array(extracted_Lstar), np.array(extracted_PSD), np.array(extracted_in_out), np.array(extracted_orbit_number)
    
if __name__ == "__main__":
    
    #This is just for debug, please use the interface provided above. Example can be seen in psd_adiabatic_space.ipynb as of 05/24/2024. Subject to change.
    
    start = datetime.datetime(year = 2013,
                              month = 10,
                              day = 1)
    
    end = datetime.datetime(year = 2013, 
                            month = 10, 
                            day = 4)
    
    JD, Lstar, PSD, in_out, orbit_number = calculate_psd(start, end, chosen_mu=3000, chosen_k=0.05, sat="B", field_model= model.TS04D, debug_mode=True, verbose=False)