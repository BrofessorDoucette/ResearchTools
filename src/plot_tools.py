from energy_channels import EnergyChannel
from matplotlib import colors
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import spacepy.time
from data_references import DataRefContainer, DataRef


OUTPUT_DIR = "./../saved_plots/"
    

def plot_L_CUT(refs: DataRefContainer, L_CUT: float, tol : float = 0.01, grid_t = "6h", avg_t = "12h", axis = None) -> pd.DataFrame:
    
    if axis == None:
        raise Exception("Axis cannot be None!")
    
    FESA, L, MLT, EPOCH, ENERGIES = refs.get_all_data()

    FESA = FESA[:, (ENERGIES < 6.5)]
    
    SATIFIES_L_CUT = ((L_CUT - tol) < L) & (L < (L_CUT + tol))
    EPOCH_TO_PLOT = EPOCH[SATIFIES_L_CUT]
    
    FESA_TO_PLOT = pd.DataFrame(FESA[SATIFIES_L_CUT, :], index=EPOCH_TO_PLOT).resample(grid_t).mean()
            
    NUM_ENERGIES = FESA_TO_PLOT.shape[1]
    colors = plt.cm.plasma(np.linspace(0, 1, NUM_ENERGIES))

    FESA_TO_PLOT = FESA_TO_PLOT.rolling(window=avg_t, center=True).mean()
    
    for E in range(NUM_ENERGIES):
        
        axis.semilogy(FESA_TO_PLOT.index, FESA_TO_PLOT[E], label=f"{ENERGIES[E]} MeV", color=colors[E], linewidth=2.5)
        
    axis.set_xlabel("Time")
    axis.set_ylabel("Flux ($cm^{-2}s^{-1}sr^{-1}MeV^{-1}$)")
    axis.legend()
    axis.yaxis.grid(color='gray', linestyle='dashed')
    axis.xaxis.grid(color='gray', linestyle='dashed')    
    
    return FESA_TO_PLOT

def plot_L_vs_Time(refs: DataRefContainer, energy_channel: EnergyChannel, l_min = 2, l_max = 6.5, flux_min = 5e0, flux_max = 1e2, axis=None):
              
    if axis == None:
        raise Exception("Axis cannot be None!")
                
    FESA, L, MLT, EPOCH, ENERGIES = refs.get_all_data()
    
    FESA = FESA[:, energy_channel.value] + 1e-9
    JD = spacepy.time.Ticktock(EPOCH, dtype="UTC").getJD()
    
    _, UNIQUE_JD = np.unique(JD, return_index=True)
    JD = JD[UNIQUE_JD]
    L = L[UNIQUE_JD]
    FESA = FESA[UNIQUE_JD]
        
    grid_x, grid_y = np.meshgrid(np.linspace(JD[0], JD[-1], len(JD)//300), np.linspace(l_min, l_max, len(JD)//300), indexing='ij')
    points = np.array([JD, L]).T
    
    interpolated_FESA = scipy.interpolate.griddata(points, FESA, (grid_x, grid_y), method="nearest")
    map = axis.imshow(interpolated_FESA.T, 
               cmap="magma", 
               extent=[JD[0], JD[-1], l_min, l_max], 
               origin='lower', 
               aspect="auto", 
               interpolation="none",
               norm=colors.LogNorm(vmin=flux_min, vmax=flux_max))
    
    axis.set_ylabel("L")
    cbar = plt.colorbar(map, ax=axis, pad=0.01)
    axis.tick_params('both', length = 3, width = 2)
    
    cbar.set_label("($cm^{-2}s^{-1}sr^{-1}MeV^{-1}$)\n", loc="center", labelpad=15, rotation=270)
    axis.set_title(f"{ENERGIES[energy_channel.value]} MeV", fontsize = 10)
    
    
#plot_L_CUT(3.5, "A", datetime.datetime(2013, 1, 1), datetime.datetime(2014, 1, 1), grid_t="24h", avg_t="1h")
#plot_L_vs_Time(EnergyChannel.MeV_6_3, "A", datetime.datetime(2013, 1, 1), datetime.datetime(2014, 1, 1))