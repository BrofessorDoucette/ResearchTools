import numpy as np
import pandas as pd
import tqdm

def emfisis_noise_floor(f):

    x = [1, 10, 100, 1000, 10000]
    y = [1e-4, 1e-5, 1e-8, 1e-10, 1e-10]

    return np.exp(np.interp(np.log(f), np.log(x), np.log(y)))


def calculate_cyclotron_frequency(B):

    m_e = 9.10938356e-31  # Mass of electron in kg
    q = 1.60217662e-19  # Charge of electron

    w_ce = (q * np.asarray(B)) / (m_e)
    f_ce = w_ce / (2 * np.pi)  # Cyclotron frequency in Hz

    return f_ce


def calculate_chorus_amplitudes_from_bsum(B_uvw, B_sum, Plan_SVD, Ell_SVD, lower=True) -> tuple:

    B = np.sqrt(B_uvw[:, 0] ** 2 + B_uvw[:, 1] ** 2 + B_uvw[:, 2] ** 2)
    B = B / (1e9)  # Magnetic field magnitude in Tesla

    f_ce = calculate_cyclotron_frequency(B)

    if lower:
        min_f = 0.05 * f_ce
        max_f = 0.5 * f_ce
    else:
        min_f = 0.5 * f_ce
        max_f = f_ce


    WNA_frequencies_bandwidths = pd.read_csv("./../EMFISIS_WNA_frequencies_and_bandwidths.csv")
    center_frequencies = WNA_frequencies_bandwidths["f_l"]
    channel_bandwidths = WNA_frequencies_bandwidths["del_l"]

    noise_floor = emfisis_noise_floor(center_frequencies)
    planarity_floor = 0.6
    ellipticity_floor = 0.7

    chorus = np.zeros(shape=(B_uvw.shape[0]))

    for T in tqdm.tqdm(range(B_uvw.shape[0])):

        channels_selected = (min_f[T] <= center_frequencies) & (center_frequencies < max_f[T])

        if np.any(channels_selected):

            bandwidths = channel_bandwidths[channels_selected]
            power_hz = B_sum[T, channels_selected].flatten()
            planarity = Plan_SVD[T, channels_selected].flatten()
            ellipticity = Ell_SVD[T, channels_selected].flatten()

            greater_than_noise_floor = (power_hz > 10 * noise_floor[channels_selected])  # Should be an order of magnitude greater than noise floor
            greater_than_planarity_floor = (planarity >= planarity_floor)
            greater_than_ellipticity_floor = (ellipticity >= ellipticity_floor)

            satisfies_filters = greater_than_noise_floor & greater_than_planarity_floor & greater_than_ellipticity_floor

            power = np.nansum(power_hz[satisfies_filters] * bandwidths[satisfies_filters])
            chorus[T] = power * (1000.0**2)  # Convert from nT^2 to pT^2
        else:
            chorus[T] = np.nan

    return chorus


def calculate_chorus_power(WNA_survey, lower=True) -> dict:

    chorus_power = calculate_chorus_amplitudes_from_bsum(
        B_uvw=WNA_survey["Buvw"],
        B_sum=WNA_survey["bsum"],
        Plan_SVD=WNA_survey["plansvd"],
        Ell_SVD=WNA_survey["ellsvd"],
        lower=lower
    )

    return chorus_power