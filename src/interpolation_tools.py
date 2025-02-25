
import numpy as np
from typing import Iterable

def interpolate_between_big_distances(x : Iterable, xp : Iterable, fp : Iterable, big_distances : Iterable = []):
    
    '''big_distances is usually something like this: "np.nonzero((times[1:] - times[:-1]) > max_diff_in_times)[0] + 1" 
        i.e an array denoting right-most edges of interpolation chunks.. See files calculating L* for use-case.'''
    
    interpolated_data = np.zeros_like(x)
    interpolated_data[:] = np.nan
    
    if len(big_distances) > 0:
                            
        for m, d in enumerate(big_distances):
            
            if m == 0 :
                
                start_index = 0
                end_index = d
                
            else:
                
                start_index = big_distances[m - 1]
                end_index = d
                
            interpolated_chunk = np.interp(x, xp[start_index:end_index], fp[start_index:end_index], left=np.nan, right=np.nan)
            non_nan_values = np.isfinite(interpolated_chunk)
            interpolated_data[non_nan_values] = interpolated_chunk[non_nan_values]
        
        #Get the last chunk too
        
        start_index = big_distances[-1]
        
        interpolated_chunk = np.interp(x, xp[start_index:], fp[start_index:], left=np.nan, right=np.nan)
        non_nan_values = np.isfinite(interpolated_chunk)
        interpolated_data[non_nan_values] = interpolated_chunk[non_nan_values]

    else:
        
        #If big_distances is empty... just interpolate like normal!
        
        interpolated_data = np.interp(x, xp, fp, left=np.nan, right=np.nan)
        
    return interpolated_data