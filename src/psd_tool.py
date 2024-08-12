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

def quartic(x, a, b, c, d, e):

    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def quadratic(x, a, b, c):
    
    return a * x ** 2 + b * x + c
                
@njit(cache=True, inline="always")
def accelerated_interp(x, xp, fp):
    
    j = np.searchsorted(xp, x, side="right") - 1
        
    if (j < 0) or (j >= xp.size - 1):
        return np.NaN
    
    if (np.log10(xp[j+1]) - np.log10(xp[j])) > 0.5:
        return np.NaN
    
    d = (x - xp[j]) / (xp[j+1] - xp[j])
    return (1 - d) * fp[j] + d * fp[j+1]
                

def select_mu_and_k_from_psd(refs: dict,
                             chosen_mu: float,
                             chosen_k: float) -> dict:
        
    '''This version does linear interpolation across time for every pitch angle and energy.'''
    
    import warnings

    warnings.filterwarnings('error')
    
    PSD = refs["PSD"]
    JD = refs["JD"]
    EPOCH = refs["EPOCH"]
    ENERGIES = refs["ENERGIES"]
    ALPHA = refs["ALPHA"]
    K = refs["K"]
    L_STAR = refs["L_STAR"]
    L = refs["L"]
    IN_OUT = refs["IN_OUT"]
    ORBIT_NUMBER = refs["ORBIT_NUMBER"]
    B = refs["B"]
    
    #print(PSD.shape, JD.shape, EPOCH.shape, ENERGIES.shape, ALPHA.shape, K.shape, L_STAR.shape, B.shape)
    
    M_e = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]
                    
    extracted_epoch = []
    extracted_Lstar = []
    extracted_L = []
    extracted_PSD = []
    extracted_in_out = []
    extracted_orbit_number = []
    
    t5 = time.time()
    
    for T in range(len(JD)):
        
        if not np.isfinite(B[T]):
            continue
        
        not_nan_and_less_than_max_k = np.isfinite(K[T, :]) & (K[T, :] < 0.25)
        
        if not np.any(not_nan_and_less_than_max_k):
            continue
                
        alpha_to_fit = ALPHA[T, not_nan_and_less_than_max_k]
        k_to_fit = K[T, not_nan_and_less_than_max_k]
        alpha_of_min_k = alpha_to_fit[np.argmin(k_to_fit)]
        
        try:
                                                           
            K_coef = np.polynomial.polynomial.polyfit(x = (alpha_to_fit - alpha_of_min_k),
                                                    y = k_to_fit,
                                                    deg = 4,
                                                    full=False)
            
        except Exception:
            
            continue
                        
        if(K_coef[0] > chosen_k):
                
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
                
        E_coefs = np.array([1, (2 * M_e), -1 * ((2 * M_e * B[T] * chosen_mu) / (np.sin(selected_alpha) ** 2))])
            
        roots = np.roots(E_coefs)
        positive_roots = (roots > 0)
        
        if not np.any(positive_roots):
            continue
        
        selected_E = roots[positive_roots][0]  
        
        valid_Lstar = np.isfinite(L_STAR[T, :])
        valid_L = np.isfinite(L[T, :])
    
        if not np.any(valid_Lstar):
            continue
        
        selected_L_star = accelerated_interp(selected_alpha, ALPHA[T, valid_Lstar], L_STAR[T, valid_Lstar])
        
        if not np.any(valid_L):
            continue
        
        selected_L = accelerated_interp(selected_alpha, ALPHA[T, valid_L], L[T, valid_L])
            
        if np.isnan(selected_L_star):
            continue
        
        psd_per_alpha_at_selected_e = []
        
        for A in range(PSD.shape[1]):
            
            valid_e = np.isfinite(PSD[T, A, :]) & np.isfinite(ENERGIES[T, :])
            
            if np.any(valid_e):
                
                psd_per_alpha_at_selected_e.append(np.exp(accelerated_interp(selected_E * 1000, ENERGIES[T, valid_e] * 1000, np.log(PSD[T, A, valid_e]))))
                
            else:
                
                psd_per_alpha_at_selected_e.append(np.NaN)
                
        
        valid_a = np.isfinite(psd_per_alpha_at_selected_e)
        
        if not np.any(valid_a):
            continue
        
        extracted_epoch.append(EPOCH[T])
        extracted_Lstar.append(selected_L_star)
        extracted_L.append(selected_L)
        extracted_PSD.append(accelerated_interp(selected_alpha, ALPHA[T, valid_a], np.asarray(psd_per_alpha_at_selected_e)[valid_a]))
        extracted_in_out.append(IN_OUT[T])
        extracted_orbit_number.append(ORBIT_NUMBER[T])

    print(f"Time taken for loop: {time.time() - t5}")
    
    warnings.filterwarnings("once")
    
    refs = {
        
        "EPOCH" : np.asarray(extracted_epoch),
        "L_STAR" : np.asarray(extracted_Lstar),
        "L" : np.asarray(extracted_L),
        "PSD" : np.asarray(extracted_PSD),
        "IN_OUT" : np.asarray(extracted_in_out),
        "ORBIT_NUMBER" : np.asarray(extracted_orbit_number)
        
    }
    
    return refs


def bin_radial_profile(LSTAR, PSD, LSTAR_MIN, LSTAR_MAX, dL = 0.10):
    
    binned_PSD = []
    binned_Lstar = []
    
    curr = LSTAR_MIN
    while curr < LSTAR_MAX:
        
        bin = ((curr - dL / 2.0) < LSTAR) & (LSTAR <= (curr + dL / 2.0))
        if (np.sum(bin) != 0):
            binned_PSD.append(np.nanmean(PSD[bin]))
        else:
            binned_PSD.append(np.NaN)
        binned_Lstar.append(curr)
        curr += dL
        
    return np.asarray(binned_Lstar), np.asarray(binned_PSD)
    
    
def plot_radial_profile(LSTAR, PSD, IS_INBOUND: bool, START_OF_ORBIT: datetime.datetime, AXIS=None):
    
    year = str(START_OF_ORBIT.year)
    month = str(START_OF_ORBIT.month)
    day = str(START_OF_ORBIT.day)
    if len(day) < 2:
        day = f"0{day}"
    hour = str(START_OF_ORBIT.hour)
    if len(hour) < 2:
        hour = f"0{hour}"
    minute = str(START_OF_ORBIT.minute)
    if len(minute) < 2:
        minute = f"0{minute}"
    
    if IS_INBOUND:
        if AXIS == None:
            plt.semilogy(LSTAR, PSD, marker="*", label=f"{month}/{day}/{year} {hour}:{minute}")
        else:
            AXIS.semilogy(LSTAR, PSD, marker="*", label=f"{month}/{day}/{year} {hour}:{minute}")
    else:
        if AXIS == None:
            plt.semilogy(LSTAR, PSD, marker=".", label=f"{month}/{day}/{year} {hour}:{minute}")
        else:
            AXIS.semilogy(LSTAR, PSD, marker="*", label=f"{month}/{day}/{year} {hour}:{minute}")
                
    
if __name__ == "__main__":
    
    #This is just for debug, please use the interface provided above. Example can be seen in psd_adiabatic_space.ipynb as of 05/24/2024. Subject to change.
    
    start = datetime.datetime(year = 2015,
                              month = 10,
                              day = 24)
    
    end = datetime.datetime(year = 2015, 
                            month = 11, 
                            day = 1)
    
    t_total = time.time()
    
    dependencies = data_loader.load_psd(satellite="B", field_model=model.TS04D, start=start, end=end)

    selected_refs = select_mu_and_k_from_psd(refs=dependencies, chosen_mu=1000, chosen_k=0.07)


        
    print(f"Total time: {time.time() - t_total}")