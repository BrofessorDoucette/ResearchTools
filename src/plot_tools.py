import numpy as np
import datetime
from dateutil import rrule
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.interpolate
from matplotlib import colors
import pandas as pd
import enum
import spacepy.time

class EnergyChannel(enum.Enum):
    
    MeV_1_8 = 0
    MeV_2_1 = 1
    MeV_2_6 = 2
    MeV_3_4 = 3
    MeV_4_2 = 4
    MeV_5_2 = 5
    MeV_6_3 = 6
    MeV_7_7 = 7
    MeV_9_9 = 8
    MeV_12_3 = 9
    MeV_15_2 = 10
    MeV_20_0 = 11
    

COMPRESSED_DATA_DIR = "./../compressed_data/"
OUTPUT_DIR = "./../saved_plots/"

def load_compressed_data(satellite: str, start: datetime.datetime, end: datetime.datetime):
    
    FESA = np.zeros((0, 12), dtype=np.float64)
    L = np.zeros((0), dtype=np.float64)
    EPOCH = np.zeros((0), dtype=datetime.datetime)
    
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):
        
        _year = str(dt.year)
        _month = str(dt.month)
        
        if len(_month) < 2:
            _month = f"0{_month}"
        
        DATA_DIR = os.path.join(COMPRESSED_DATA_DIR, f"{_year}/")
        FILE_NAME = f"REPT_{_year}{_month}_{satellite.upper()}.npz"
        DATA_PATH = os.path.join(DATA_DIR, FILE_NAME)
        
        if not os.path.exists(DATA_PATH):
            raise Exception(f"\nData file not found: {DATA_PATH}")
        
        print(f"Loading : {FILE_NAME}")
        data = np.load(DATA_PATH, allow_pickle=True)
        
        FESA = np.concatenate((FESA, data["FESA"]), axis = 0)
        L = np.concatenate((L, data["L"]), axis = 0)
        EPOCH = np.concatenate((EPOCH, data["EPOCH"]), axis = 0)
        
        if i == 0:
            ENERGIES = data["ENERGIES"]
        
        data.close()
        
    return FESA, L, EPOCH, ENERGIES
    

def plot_L_CUT(L_CUT: float, satellite: str, start: datetime.datetime, end: datetime.datetime, tol : float = 0.01, grid_t = "6h", avg_t = "12h", save_plot=False):
    
    FESA, L, EPOCH, ENERGIES = load_compressed_data(satellite=satellite, start=start, end=end)
    
    FESA[FESA < 0] = np.NaN
    FESA = FESA[:, (ENERGIES < 6.5)]
    print(f"FESA: {FESA.shape}", f"L: {L.shape}", f"EPOCH: {EPOCH.shape}")
    
    SATIFIES_L_CUT = ((L_CUT - tol) < L) & (L < (L_CUT + tol))
    SATISFIES_DATE_EXTENT = (start < EPOCH) & (EPOCH < end)
    
    TIMES_TO_PLOT = SATIFIES_L_CUT & SATISFIES_DATE_EXTENT
        
    EPOCH_TO_PLOT = EPOCH[TIMES_TO_PLOT]
    
    FESA_TO_PLOT = pd.DataFrame(FESA[TIMES_TO_PLOT, :], index=EPOCH_TO_PLOT).resample(grid_t).mean()
        
    fig, ax = plt.subplots(1, 1, sharex=True)    
    
    NUM_ENERGIES = FESA_TO_PLOT.shape[1]
    colors = plt.cm.plasma(np.linspace(0, 1, NUM_ENERGIES))
        
    for E in range(NUM_ENERGIES):
        
        ax.semilogy(FESA_TO_PLOT.index, FESA_TO_PLOT[E].rolling(window=avg_t, center=True).mean(), label=f"{ENERGIES[E]} MeV", color=colors[E], linewidth=2.5)
        
    plt.xlabel("Time")
    plt.ylabel("Flux ($cm^{-2}s^{-1}sr^{-1}MeV^{-1}$)")
    plt.xlim((start, end))
    plt.title(rf"RBSP-{satellite.upper()}, Electron Flux (Spin Averaged), L = {L_CUT} $\pm$ {tol}")
    plt.legend()
    fig.set_size_inches(10, 6)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.tight_layout()
    plt.show()
    
    if save_plot:
        fig.savefig(OUTPUT_DIR + f"LCUT-{int(L_CUT*10)}-RBSP-{satellite.upper()}-{_month}-{_year}.pdf", format="pdf", dpi=100)
    

def plot_L_vs_Time(energy_channel: EnergyChannel, satellite: str, start: datetime.datetime, end: datetime.datetime):
    
    L_MIN = 2
    L_MAX = 6.5
    FLUX_MIN = 5e0
    FLUX_MAX = 1e2
    
    FESA, L, EPOCH, ENERGIES = load_compressed_data(satellite=satellite, start=start, end=end)
    
    FESA[FESA < 0] = np.NaN
    
    SATISFIES_DATE_EXTENT = (start < EPOCH) & (EPOCH < end)
    
    FESA = FESA[SATISFIES_DATE_EXTENT, energy_channel.value] + 1e-4
    EPOCH = EPOCH[SATISFIES_DATE_EXTENT]
    print(EPOCH)
    JD = spacepy.time.Ticktock(EPOCH, dtype="UTC").getJD()
    L = L[SATISFIES_DATE_EXTENT]
    
    _, UNIQUE_JD = np.unique(JD, return_index=True)
    JD = JD[UNIQUE_JD]
    L = L[UNIQUE_JD]
    FESA = FESA[UNIQUE_JD]
    
    fig, axs = plt.subplots(1, 1, sharex=True)
    
    grid_x, grid_y = np.meshgrid(np.linspace(JD[0], JD[-1], len(JD)//300), np.linspace(L_MIN, L_MAX, len(JD)//300), indexing='ij')
    print(grid_x, grid_y)
    
    points = np.array([JD, L]).T
    print(points.shape)
    print(FESA.shape)
    
    interpolated_FESA = scipy.interpolate.griddata(points, FESA, (grid_x, grid_y), method="nearest")
    map = axs.imshow(interpolated_FESA.T, 
               cmap="inferno", 
               extent=[JD[0], JD[-1], L_MIN, L_MAX], 
               origin='lower', 
               aspect="auto", 
               interpolation="gaussian",
               norm=colors.LogNorm(vmin=FLUX_MIN, vmax=FLUX_MAX))
    
    axs.set_ylabel("L")
    axs.set_xlabel("Time")
    cbar = fig.colorbar(map, ax=axs, pad=0.02)
    cbar.set_label("Flux ($cm^{-2}s^{-1}sr^{-1}MeV^{-1}$)\n", loc="center", labelpad=15, rotation=270)
    xticks = axs.get_xticks()
    print(xticks)
    dt_xticks = spacepy.time.Ticktock(xticks, "JD").getUTC()
    print(dt_xticks)
    labels = [date.strftime("%m-%d %H:%M") for date in dt_xticks]
    axs.set_xticks(xticks[1:-1], labels[1:-1])
    fig.set_size_inches(10, 6)
    plt.title(f"RBSP-{satellite.upper()}, Spin Averaged Electron Flux, E = {ENERGIES[energy_channel.value]} MeV")
    plt.tight_layout()
    plt.show()
    
    
#plot_L_CUT(3.5, "A", datetime.datetime(2013, 1, 1), datetime.datetime(2014, 1, 1), grid_t="24h", avg_t="1h")
plot_L_vs_Time(EnergyChannel.MeV_6_3, "A", datetime.datetime(2013, 1, 1), datetime.datetime(2014, 1, 1))