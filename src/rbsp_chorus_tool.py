
import numpy as np


def calculate_chorus_amplitudes_from_bsum(B_uvw, B_sum, WFR_bandwidths, WFR_frequencies):
    
    m_e = 9.10938356E-31 #Mass of electron in kg
    q = 1.60217662E-19 #Charge of electron
    
    B = np.sqrt(B_uvw[:, 0]**2 + B_uvw[:, 1]**2 + B_uvw[:, 2]**2)
    B = B / (1e9) #Magnetic field magnitude in Tesla
    
    w_ce = (B*q) / (m_e) 
    f_ce = w_ce / (2*np.pi) #Cyclotron frequency in Hz
    
    min_f = f_ce / 10.0 
    max_f = 8*(f_ce / 10.0)
    
    frequency_bin_minimums = WFR_frequencies - WFR_bandwidths / 2.0
    frequency_bin_maximums = WFR_frequencies + WFR_bandwidths / 2.0
    
    chorus = []
    
    for T in range(B_uvw.shape[0]):
        
    
        frequency_bins_to_include = (min_f[T] <= frequency_bin_minimums) & (frequency_bin_maximums < max_f[T])
        
        if np.any(frequency_bins_to_include):
        
            bandwidths = WFR_bandwidths[frequency_bins_to_include]
            power_hz = B_sum[T, frequency_bins_to_include]
            
            power = np.nansum(power_hz * bandwidths)
            amplitude = np.sqrt(power)
            chorus.append(amplitude * 1000.0) #Convert from nT to pT
        else:
            chorus.append(np.nan)
        
    return chorus


def iterate_through_days_and_calculate_chorus_amplitudes(WNA_survey, WFR_spectral_matrix):
    
    chorus = []

    #This makes sure that the same number of days were loaded
    assert(len(WNA_survey["timestamps_per_file"]) == WFR_spectral_matrix["WFR_bandwidth"].shape[0])

    index = 0
    for day in range(len(WNA_survey["timestamps_per_file"])):
        
        num_timesteps_for_day = WNA_survey["timestamps_per_file"][day]
        
        chorus.extend(calculate_chorus_amplitudes_from_bsum(WNA_survey["Buvw"][index:index + num_timesteps_for_day], 
                                                            WNA_survey["bsum"][index:index + num_timesteps_for_day],
                                                            WFR_spectral_matrix["WFR_bandwidth"][day, :],
                                                            WFR_spectral_matrix["WFR_frequencies"][day, :]))
                
        index = index + num_timesteps_for_day
        
    return np.asarray(chorus)