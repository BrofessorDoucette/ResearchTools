from energy_channels import EnergyChannel
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import spacepy.time
from data_references import DataRefContainer


OUTPUT_DIR = "./../saved_plots/"
    

def plot_l_cut(refs: DataRefContainer,
               l_cut: float, tol: float = 0.01,
               grid_t ="6h", avg_t ="12h",
               axis = None) -> pd.DataFrame:
    
    if axis is None:
        raise Exception("Axis cannot be None!")
    
    fesa, L, mlt, epoch, energies = refs.get_all_data()

    fesa = fesa[:, (energies < 6.5)]
    
    satifies_l_cut = ((l_cut - tol) < L) & (L < (l_cut + tol))
    epoch_to_plot = epoch[satifies_l_cut]
    
    fesa_to_plot = pd.DataFrame(fesa[satifies_l_cut, :], index=epoch_to_plot).resample(grid_t).mean()
            
    num_energies = fesa_to_plot.shape[1]
    flux_colors = plt.cm.plasma(np.linspace(0, 1, num_energies))

    fesa_to_plot = fesa_to_plot.rolling(window=avg_t, center=True).mean()
    
    for E in range(num_energies):
        
        axis.semilogy(fesa_to_plot.index, fesa_to_plot[E],
                      color=flux_colors[E], linewidth=2.5,
                      label=f"{energies[E]} MeV")
        
    axis.set_xlabel("Time")
    axis.set_ylabel("Flux ($cm^{-2}s^{-1}sr^{-1}MeV^{-1}$)")
    axis.legend()
    axis.yaxis.grid(color='gray', linestyle='dashed')
    axis.xaxis.grid(color='gray', linestyle='dashed')    
    
    return fesa_to_plot


def plot_l_vs_time(refs: DataRefContainer, energy_channel: EnergyChannel,
                   l_min = 2, l_max = 6.5, flux_min = 5e0, flux_max = 1e2,
                   axis=None):
              
    if axis is None:
        raise Exception("Axis cannot be None!")
                
    fesa, L, mlt, epoch, energies = refs.get_all_data()
    
    fesa = fesa[:, energy_channel.value] + 1e-9
    JD = spacepy.time.Ticktock(epoch, dtype="UTC").getJD()
    
    _, unique_JD = np.unique(JD, return_index=True)
    JD = JD[unique_JD]
    L = L[unique_JD]
    fesa = fesa[unique_JD]
        
    grid_x, grid_y = np.meshgrid(np.linspace(JD[0], JD[-1], len(JD)//300), np.linspace(l_min, l_max, len(JD)//300),
                                 indexing='ij')

    points = np.array([JD, L]).T
    
    interpolated_fesa = scipy.interpolate.griddata(points, fesa, (grid_x, grid_y), method="nearest")

    image_cmap = axis.imshow(
            interpolated_fesa.T,
            cmap="magma",
            extent=[JD[0], JD[-1], l_min, l_max],
            origin='lower',
            aspect="auto",
            interpolation="none",
            norm=colors.LogNorm(vmin=flux_min, vmax=flux_max))
    
    axis.set_ylabel("L")
    cbar = plt.colorbar(image_cmap, ax=axis, pad=0.01)
    axis.tick_params('both', length = 3, width = 2)
    
    cbar.set_label("($cm^{-2}s^{-1}sr^{-1}MeV^{-1}$)\n", loc="center", labelpad=15, rotation=270)
    axis.set_title(f"{energies[energy_channel.value]} MeV", fontsize = 10)
