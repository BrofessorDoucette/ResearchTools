import typing
from collections.abc import Iterable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_l_cut(
    refs: dict,
    l_cut: float,
    tol: float = 0.01,
    grid_t: str = "6h",
    avg_t: str = "12h",
    axis: typing.Any = None,
) -> pd.DataFrame:

    if axis is None:
        raise Exception("Axis cannot be None!")

    fesa = refs["FESA"]
    L = refs["L"]
    epoch = refs["EPOCH"]
    energies = refs["ENERGIES"]

    fesa = fesa[:, (energies < 6.5)]

    satifies_l_cut = ((l_cut - tol) < L) & (L < (l_cut + tol))
    epoch_to_plot = epoch[satifies_l_cut]

    fesa_to_plot = (
        pd.DataFrame(fesa[satifies_l_cut, :], index=epoch_to_plot).resample(grid_t).mean()
    )

    num_energies = fesa_to_plot.shape[1]
    flux_colors: typing.Any = plt.cm.get_cmap("plasma", num_energies)

    fesa_to_plot = fesa_to_plot.rolling(window=avg_t, center=True).mean()

    for E in range(num_energies):

        axis.semilogy(
            fesa_to_plot.index,
            fesa_to_plot[E],
            color=flux_colors[E],
            linewidth=2.5,
            label=f"{energies[E]} MeV",
        )

    axis.set_xlabel("Time")
    axis.set_ylabel("Flux ($cm^{-2}s^{-1}sr^{-1}MeV^{-1}$)")
    axis.legend()
    axis.yaxis.grid(color="gray", linestyle="dashed")
    axis.xaxis.grid(color="gray", linestyle="dashed")

    return fesa_to_plot


def plot_l_vs_time_log_colors(
    x: Iterable,
    y: Iterable,
    c: Iterable,
    cmap=cm.get_cmap("viridis"),
    cbar_label="($cm^{-2}s^{-1}sr^{-1}MeV^{-1}$)\n",
    axis: typing.Any = None,
    vmin=None,
    vmax=None,
) -> None:

    assert len(x) == len(y) == len(c)

    if axis is None:
        raise Exception("Axis cannot be None!")

    _, unique_x = np.unique(x, return_index=True)
    x = x[unique_x]
    y = y[unique_x]
    c = c[unique_x]

    c_non_zero = c > 0
    x = x[c_non_zero]
    y = y[c_non_zero]
    c = c[c_non_zero]

    if (vmin is None) or (vmax is None):
        plot_cmap = axis.scatter(
            x,
            y,
            c=c,
            cmap=cmap,
            norm=colors.LogNorm(vmin=np.amin(c[np.isfinite(c)]), vmax=np.amax(c[np.isfinite(c)])),
            s=8,
        )
    else:
        plot_cmap = axis.scatter(
            x, y, c=c, cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), s=8
        )

    axis.set_facecolor("black")
    axis.tick_params("both", length=3, width=2)

    # To be honest I have no idea how this actually works anymore. Matplotlib is bad and they make it difficult to position the colorbars.
    axins = inset_axes(
        axis,
        width="1%",  # width: 5% of parent_bbox width
        height="100%",  # height: 50%
        loc="lower left",
        bbox_to_anchor=(1.01, 0, 1, 1),
        bbox_transform=axis.transAxes,
        borderpad=0,
    )

    plot_cmap.cmap.set_under("black")
    cbar = plt.colorbar(plot_cmap, cax=axins)
    cbar.set_label(cbar_label, loc="center", labelpad=25, rotation=270)


def bin_3D_data(
    xdata, ydata, zdata, xstart, xend, xstep, ystart, yend, ystep, xprecision=3, yprecision=3
):
    """This is used when plotting the chorus proxy, chorus from VAP, fokker-planck simulation results, etc.
    This function is fast for high resolution bins, but slow for large input arrays.
    (xend - xstart)/xstep should be an integer, and (yend - ystart)/ystep should be an integer."""

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    zdata = np.array(zdata)

    x_bin_edges = [xstart + j * xstep for j in range((int((xend - xstart) / xstep) + 1))]
    y_bin_edges = [ystart + j * ystep for j in range((int((yend - ystart) / ystep) + 1))]

    indices_needed = (
        (x_bin_edges[0] < xdata)
        & (xdata < x_bin_edges[-1])
        & (y_bin_edges[0] < ydata)
        & (ydata < y_bin_edges[-1])
    )

    xdata = xdata[indices_needed]
    ydata = ydata[indices_needed]
    zdata = zdata[indices_needed]

    if type(x_bin_edges[0]) == float:
        x_bin_edges = np.round(x_bin_edges, decimals=xprecision)
    if type(y_bin_edges[0]) == float:
        y_bin_edges = np.round(y_bin_edges, decimals=yprecision)

    sum_of_z_in_each_x_y_bin = np.zeros(shape=(len(x_bin_edges) - 1, len(y_bin_edges) - 1))
    num_points_in_each_x_y_bin = np.zeros_like(sum_of_z_in_each_x_y_bin)

    x_mapping = np.searchsorted(x_bin_edges, xdata) - 1
    y_mapping = np.searchsorted(y_bin_edges, ydata) - 1

    for T in range(len(xdata)):

        x_bin = x_mapping[T]
        y_bin = y_mapping[T]

        num_points_in_each_x_y_bin[x_bin, y_bin] += 1
        sum_of_z_in_each_x_y_bin[x_bin, y_bin] += zdata[T]

    return sum_of_z_in_each_x_y_bin, num_points_in_each_x_y_bin


def create_conditional_probability_plot(list1, list2, bins=10):
    """
    Create a 2D conditional probability plot (heatmap) from two lists.

    Parameters:
    list1 (list): First input list
    list2 (list): Second input list, related to list1 by indices
    bins (int): Number of bins for the histogram (default: 10)
    """
    # Convert lists to numpy arrays
    x = np.array(list1)
    y = np.array(list2)

    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)

    # Normalize to get conditional probability P(y|x)
    hist = hist / np.sum(hist, axis=1, keepdims=True)

    # Create mesh for plotting
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(
        hist.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
        interpolation="nearest",
    )
    plt.colorbar(label="Conditional Probability P(y|x)")
    plt.xlabel("X (List 1)")
    plt.ylabel("Y (List 2)")
    plt.title("2D Conditional Probability Plot")

    # Save the plot
    plt.savefig("conditional_probability_plot.png")
    plt.close()

def plot_2d_heatmap(x, y, z, bins=50, xtitle="X", ytitle="Y", ztitle="Z", title="Heatmap", xlim =None, ylim=None, norm="linear", xscale="linear", yscale="linear", plot_density=False):
    """
    Create a 2D heatmap from three arrays where the color represents the average of z_values
    in bins defined by x_values and y_values.

    Parameters:
    x_values (array-like): Values for x-axis
    y_values (array-like): Values for y-axis
    z_values (array-like): Values to average for color
    bins (int or tuple): Number of bins for histogram (default: 50)

    Returns:
    None (displays a matplotlib plot)
    """
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    finite_x = np.isfinite(x)
    finite_y = np.isfinite(y)
    finite_z = np.isfinite(z)
    all_finite = finite_x & finite_y & finite_z

    x = x[all_finite]
    y = y[all_finite]
    z = z[all_finite]

    # Create 2D histogram with counts and sum of z_values
    hist_sum, x_edges, y_edges = np.histogram2d(
        x, y, bins=bins, weights=z, range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]]
    )
    hist_count, _, _ = np.histogram2d(x, y, bins=bins, range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]])

    hist_avg = np.zeros_like(hist_sum)

    hist_avg[hist_count == 0] = np.nan  # Set empty bins to NaN

    hist_avg[hist_count != 0] = hist_sum[hist_count != 0] / hist_count[hist_count != 0]
    
    # Create mesh for plotting
    _X, _Y = np.meshgrid(x_edges[:-1], y_edges[:-1])

    if plot_density:
        fig, axs = plt.subplots(2, 1, figsize=(10,8), sharex=True, sharey=True)
        axs[0].set_xscale(xscale)
        axs[0].set_yscale(yscale)
        axs[0].set_xlim(xlim)
        axs[0].set_ylim(ylim)
        coloredmesh = axs[0].pcolormesh(_X, _Y, hist_avg.T, cmap='viridis', norm=norm)
        cax = inset_axes(axs[0], width="2%", height="100%", loc='right', borderpad=-2)
        cbar = plt.colorbar(coloredmesh, cax=cax)
        cbar.set_label(ztitle, loc="center", labelpad=10, rotation=270)
        axs[0].set_xlabel(xtitle)
        axs[0].set_ylabel(ytitle)
        axs[0].set_title(title)


        hist_density, _, _ = np.histogram2d(x, y, bins=bins, density=True, range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]])
        
        axs[1].set_xscale(xscale)
        axs[1].set_yscale(yscale)
        axs[1].set_xlim(xlim)
        axs[1].set_ylim(ylim)
        coloredmesh = axs[1].pcolormesh(_X, _Y, hist_density.T, cmap='viridis')
        cax = inset_axes(axs[1], width="2%", height="100%", loc='right', borderpad=-2)
        cbar = plt.colorbar(coloredmesh, cax=cax)
        cbar.set_label(ztitle, loc="center", labelpad=10, rotation=270)
        axs[1].set_xlabel(xtitle)
        axs[1].set_ylabel(ytitle)
        axs[1].set_title("Density")
        cbar = plt.colorbar(coloredmesh, cax=cax)
        cbar.set_label("Joint Density", loc="center", labelpad=25, rotation=270)
        plt.show()
    else:
        
        plt.figure(figsize=(10, 8))        
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.pcolormesh(_X, _Y, hist_avg.T, cmap='viridis', norm=norm)
        cbar = plt.colorbar()
        cbar.set_label(ztitle, loc="center", labelpad=25, rotation=270)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.title(title)
        plt.show()
