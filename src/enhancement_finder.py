import numpy as np
import pandas as pd
import datetime


def find_enchancements_based_on_del_ae(OMNI: pd.DataFrame, 
                                       slow_t: str, 
                                       low_threshold: float, 
                                       high_threshold: float, 
                                       minimum_duration: datetime.timedelta, 
                                       end_duration: datetime.timedelta):
    
    del_AE = (OMNI["AE"] - OMNI["AE"].rolling(slow_t, center=True).mean()).abs()
    time = del_AE.index
    
    current_enhancement = False
    start = None
    last_above = None
    
    for t, y in zip(time, del_AE):
        
        if y >= high_threshold:
                        
            if (current_enhancement == False):
                
                current_enhancement = True
                
                start = t
                                
        if y > low_threshold:
            
            last_above = t
            
        if y < low_threshold:
            
            if (current_enhancement == True) and ((t - last_above) > end_duration):
                
                current_enhancement = False
                
                if (last_above - start) > minimum_duration:
                    
                    print(f"Start: {start}, End: {last_above}, Duration: {last_above - start}")
                
            