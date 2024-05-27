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
import data_loader
import time
import numba as nb
from numba import njit

M_e = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]

def quartic(x, a, b, c, d, e):

    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def quadratic(x, a, b, c):
    
    return a * x ** 2 + b * x + c

@njit(cache=True, inline="always")    
def interpolate_through_time(alpha, energies, JD, PSD):
    
    for A in range(len(alpha)):
        for E in range(len(energies)):
            not_nan = np.isfinite(PSD[:, A, E])
            
            if np.any(not_nan):
                
                indices_of_not_nan = np.nonzero(not_nan)[0]
                JD_interp_start = JD[indices_of_not_nan[0]]
                JD_interp_end = JD[indices_of_not_nan[-1]]
                
                JD_interp_region = (JD_interp_start < JD) & (JD <= JD_interp_end)
                
                PSD[JD_interp_region, A, E] = np.interp(JD[JD_interp_region], JD[not_nan], PSD[not_nan, A, E])
                
@njit(cache=True, inline="always")
def accelerated_interp(x, xp, fp):
    
    j = np.searchsorted(xp, x, side="right") - 1
        
    if (j < 0) or (j >= xp.size - 1):
        return np.NaN
    
    d = (x - xp[j]) / (xp[j+1] - xp[j])
    return (1 - d) * fp[j] + d * fp[j+1]
                

def calculate_psd(dependencies: dict,
                  chosen_mu: float,
                  chosen_k: float,
                  energy_channels_to_remove_from_end: int = 5,
                  debug_mode: bool = False,
                  verbose: bool = False) -> None:
        
    '''This version does linear interpolation across time for every pitch angle and energy.'''
    
    import warnings

    warnings.filterwarnings('error')
    
    ect_fedu = dependencies["ECT_FEDU"]
    ect_JD = dependencies["ECT_JD"]
    ect_epoch = dependencies["ECT_EPOCH"]
    ect_fedu_energy = dependencies["ECT_FEDU_ENERGY"]
    ect_fedu_energy_delta_plus = dependencies["ECT_FEDU_ENERGY_DELTA_PLUS"]
    ect_fedu_energy_delta_minus = dependencies["ECT_FEDU_ENERGY_DELTA_MINUS"]
    ect_fedu_alpha = dependencies["ECT_FEDU_ALPHA"]
    magephem_alpha = dependencies["MAGEPHEM_ALPHA"]
    K = dependencies["K"]
    L_star = dependencies["LSTAR"]
    in_out = dependencies["IN_OUT"]
    orbit_number = dependencies["ORBIT_NUMBER"]
    B = dependencies["B"]

    ect_fedu[(ect_fedu < 0)] = np.NaN
    ect_fedu = ect_fedu[:, :, :-1*energy_channels_to_remove_from_end]
    energy_maximums = (ect_fedu_energy + ect_fedu_energy_delta_plus)[:-1*energy_channels_to_remove_from_end] / 1000.0
    energy_minimums = (ect_fedu_energy - ect_fedu_energy_delta_minus)[:-1*energy_channels_to_remove_from_end] / 1000.0
    ect_energies = np.sqrt(energy_maximums * energy_minimums)

    
    pc_squared = 0.5 * (energy_minimums * (energy_minimums + 2 * M_e) + energy_maximums * (energy_maximums + 2 * M_e))        
    PSD = (ect_fedu / pc_squared) * 1.66e-10 * 200.3 #CHEN 2005
        
    if(debug_mode):
        num_failed_from_k = 0
        num_failed_from_L_star = 0
        
    t4 = time.time()    
    
    if not np.all(np.diff(ect_JD) > 0):
        raise Exception("ect_JD needs to be strictly increasing to interpolate over time!")
    
    interpolate_through_time(ect_fedu_alpha, ect_energies, ect_JD, PSD)
                
    print(f"Time taken to interpolate over time: {time.time() - t4}")
    
    extracted_epoch = []
    extracted_Lstar = []
    extracted_PSD = []
    extracted_in_out = []
    extracted_orbit_number = []
    
    t5 = time.time()
    
    for T in range(len(ect_JD)):
        
        not_nan_and_less_than_max_k = np.isfinite(K[T, :]) & (K[T, :] < 0.25)
        
        if not np.any(not_nan_and_less_than_max_k):
            continue
                
        alpha_to_fit = magephem_alpha[not_nan_and_less_than_max_k]
        k_to_fit = K[T, not_nan_and_less_than_max_k]
        alpha_of_min_k = alpha_to_fit[np.argmin(k_to_fit)]
        
        try:
                                                           
            K_coef = np.polynomial.polynomial.polyfit(x = (alpha_to_fit - alpha_of_min_k),
                                                    y = k_to_fit,
                                                    deg = 4,
                                                    full=False)
            
        except Exception:
            
            if(debug_mode and verbose):
                
                print(f"Failed to fit k over alpha properly! T: {T}")
            
            continue
                        
        if(K_coef[0] > chosen_k):
            
            if(debug_mode):
                num_failed_from_k += 1
                
            continue
                            
        K_coef[0] = K_coef[0] - chosen_k                                         
        roots = np.roots(((np.flip(K_coef, axis=0)))) + alpha_of_min_k
        real_roots = np.isreal(roots)
        
        if not np.any(real_roots):
            continue
        
        roots = roots[real_roots]
        
        selected_alpha = np.real(roots[0]) #Gets real part of roots only!
                
        if (selected_alpha < 0) or (np.pi < selected_alpha):
            continue
        
        if(debug_mode and verbose):
            print(f"Chosen K: {chosen_k}, Selected Alpha: {selected_alpha}")
            
                
        E_coefs = np.array([1, (2 * M_e), -1 * ((2 * M_e * B[T] * chosen_mu) / (np.sin(selected_alpha) ** 2))])
        roots = np.roots(E_coefs)
        positive_roots = (roots > 0)
        
        if not np.any(positive_roots):
            continue
        
        selected_E = roots[positive_roots][0]
                        
        if(debug_mode and verbose):
            print(f"Chosen Mu: {chosen_mu}, Selected E: {selected_E}")
        
        
        not_nan = np.isfinite(L_star[T, :])
    
        if not np.any(not_nan):
            continue
        
        selected_L_star = accelerated_interp(selected_alpha, magephem_alpha[not_nan], L_star[T, not_nan])

        if(debug_mode and verbose):
            print(f"Selected L_star: {selected_L_star}")
            
        if np.isnan(selected_L_star):
            if(debug_mode):
                num_failed_from_L_star += 1
            continue
        
        psd_per_alpha_at_selected_e = []
        
        for A in range(PSD.shape[1]):
            
            not_nan = np.isfinite(PSD[T, A, :])
            
            if np.any(not_nan):
                
                psd_per_alpha_at_selected_e.append(accelerated_interp(selected_E, ect_energies[not_nan], PSD[T, A, not_nan]))
                
            else:
                
                psd_per_alpha_at_selected_e.append(np.NaN)
                
        not_nan = np.isfinite(psd_per_alpha_at_selected_e)
        
        if not np.any(not_nan):
            continue
                        
        extracted_epoch.append(ect_epoch[T])
        extracted_Lstar.append(selected_L_star)
        extracted_PSD.append(accelerated_interp(selected_alpha, ect_fedu_alpha[not_nan], np.asarray(psd_per_alpha_at_selected_e)[not_nan]))
        extracted_in_out.append(in_out[T])
        extracted_orbit_number.append(orbit_number[T])
                
    if(debug_mode):
        print(f"Number failed: From K: {num_failed_from_k}")
        print(f"Number failed: From L_star: {num_failed_from_L_star}")

    print(f"Time taken for loop: {time.time() - t5}")
    
    warnings.filterwarnings("once")
    
    return np.asarray(extracted_epoch), np.asarray(extracted_Lstar), np.asarray(extracted_PSD), np.asarray(extracted_in_out), np.asarray(extracted_orbit_number)
    
    
if __name__ == "__main__":
    
    #This is just for debug, please use the interface provided above. Example can be seen in psd_adiabatic_space.ipynb as of 05/24/2024. Subject to change.
    
    start = datetime.datetime(year = 2013,
                              month = 1,
                              day = 1)
    
    end = datetime.datetime(year = 2013, 
                            month = 2, 
                            day = 1)
    
    t_total = time.time()
    
    dependencies = data_loader.load_psd_dependencies(satellite="A", field_model=model.TS04D, start=start, end=end)

    JD, Lstar, PSD, in_out, orbit_number = calculate_psd(dependencies=dependencies, chosen_mu=3000, chosen_k=0.18, debug_mode=False, verbose=False)


        
    print(f"Total time: {time.time() - t_total}")