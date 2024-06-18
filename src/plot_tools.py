from data_references import REPTDataRefContainer
from data_references import POESDataRefContainer
from energy_channels import EnergyChannel
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import numpy.typing as npt
import pandas as pd
from collections.abc import Iterable
import typing


OUTPUT_DIR = "./../saved_plots/"
    

def plot_l_cut(refs: REPTDataRefContainer,
               l_cut: float, tol: float = 0.01,
               grid_t: str = "6h", avg_t: str = "12h",
               axis: typing.Any = None) -> pd.DataFrame:
    
    if axis is None:
        raise Exception("Axis cannot be None!")
    
    fesa, L, mlt, epoch, energies = refs.get_all_data()

    fesa = fesa[:, (energies < 6.5)]
    
    satifies_l_cut = ((l_cut - tol) < L) & (L < (l_cut + tol))
    epoch_to_plot = epoch[satifies_l_cut]
    
    fesa_to_plot = pd.DataFrame(fesa[satifies_l_cut, :], index=epoch_to_plot).resample(grid_t).mean()
            
    num_energies = fesa_to_plot.shape[1]
    flux_colors: typing.Any = plt.cm.get_cmap("plasma", num_energies)

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


def plot_l_vs_time_log_colors(x : Iterable, y : Iterable, c : Iterable, 
                              cmap = cm.get_cmap("viridis"),
                              cbar_label ="($cm^{-2}s^{-1}sr^{-1}MeV^{-1}$)\n",
                              axis: typing.Any = None) -> None:
              
    assert(len(x) == len(y) == len(c))          
    
    if axis is None:
        raise Exception("Axis cannot be None!")
                        
    _, unique_x = np.unique(x, return_index=True)
    x = x[unique_x]
    y = y[unique_x]
    c = c[unique_x]
    
    c_non_zero = (c > 0)
    x = x[c_non_zero]
    y = y[c_non_zero]
    c = c[c_non_zero]
    
    plot_cmap = axis.scatter(x, y, c = c, cmap=cmap, norm=colors.LogNorm(vmin = np.amin(c[np.isfinite(c)]), vmax = np.amax(c[np.isfinite(c)])), s = 8)
    
    axis.set_facecolor(cmap(0))

    cbar = plt.colorbar(plot_cmap, ax=axis, pad=0.01)
    axis.tick_params('both', length = 3, width = 2)
    
    cbar.set_label(cbar_label, loc="center", labelpad=15, rotation=270)

