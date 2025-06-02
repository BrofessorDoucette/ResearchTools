import numpy as np
import pandas as pd

def emfisis_noise_floor(f):

    x = [1, 10, 100, 1000, 10000]
    y = [1e-4, 1e-5, 1e-8, 1e-10, 1e-10]

    return np.exp(np.interp(np.log(f), np.log(x), np.log(y)))


def emfisis_planarity_floor(f):

    x = [1, 10, 30, 100, 300, 1000, 3000, 10000]
    y = [1.0, 0.95, 0.90, 0.8, 0.6, 0.40, 0.25, 0.25]

    return np.interp(np.log(f), np.log(x), y)


def calculate_cyclotron_frequency(B):

    m_e = 9.10938356e-31  # Mass of electron in kg
    q = 1.60217662e-19  # Charge of electron

    w_ce = (q * np.asarray(B)) / (m_e)
    f_ce = w_ce / (2 * np.pi)  # Cyclotron frequency in Hz

    return f_ce


def calculate_chorus_amplitudes_from_bsum(B_uvw, B_sum, Magnetic_Planarity, Magnetic_Ellipticity, lower=True) -> tuple:

    B = np.sqrt(B_uvw[:, 0] ** 2 + B_uvw[:, 1] ** 2 + B_uvw[:, 2] ** 2)
    B = B / (1e9)  # Magnetic field magnitude in Tesla

    f_ce = calculate_cyclotron_frequency(B)

    if lower:
        min_f = 5 * (f_ce / 100.0)
        max_f = 5 * (f_ce / 10.0)
    else:
        min_f = 5 * (f_ce / 10.0)
        max_f = f_ce

    WNA_frequencies_bandwidths = pd.read_csv("./../EMFISIS_WNA_frequencies_and_bandwidths.csv")
    WNA_frequencies = WNA_frequencies_bandwidths["f_l"]
    WNA_bandwidths = WNA_frequencies_bandwidths["del_l"]

    print(WNA_frequencies)
    print(WNA_bandwidths)

    frequency_bin_minimums = WNA_frequencies - (WNA_bandwidths / 2)
    frequency_bin_maximums = WNA_frequencies + (WNA_bandwidths / 2)

    noise_floor = emfisis_noise_floor(WNA_frequencies)
    planarity_floor = 0.6
    ellipticity_floor = 0.7

    chorus = []

    for T in range(B_uvw.shape[0]):

        frequency_bins = (min_f[T] <= frequency_bin_minimums) & (frequency_bin_maximums < max_f[T])

        if np.any(frequency_bins):

            bandwidths = WNA_bandwidths[frequency_bins]
            power_hz = B_sum[T, frequency_bins].flatten()
            planarity = Magnetic_Planarity[T, frequency_bins].flatten()
            ellipticity = Magnetic_Ellipticity[T, frequency_bins].flatten()

            greater_than_noise_floor = (power_hz > 10 * noise_floor[frequency_bins])  # Should be an order of magnitude greater than noise floor
            greater_than_planarity_floor = (planarity >= planarity_floor)
            greater_than_ellipticity_floor = (ellipticity >= ellipticity_floor)

            satisfies_filters = greater_than_noise_floor & greater_than_planarity_floor & greater_than_ellipticity_floor

            if np.any(satisfies_filters):

                power = np.nansum(power_hz[satisfies_filters] * bandwidths[satisfies_filters])
                chorus.append((np.sqrt(power) * 1000.0)**2)  # Convert from nT to pT
            else:
                chorus.append(0.01)
        else:
            chorus.append(np.nan)

    return chorus


def calculate_chorus_power(WNA_survey, Magnetic_Planarity, Magnetic_Ellipticity, lower=True) -> dict:

    chorus_power = calculate_chorus_amplitudes_from_bsum(
        B_uvw=WNA_survey["Buvw"],
        B_sum=WNA_survey["bsum"],
        Magnetic_Planarity=Magnetic_Planarity,
        Magnetic_Ellipticity=Magnetic_Ellipticity,
        lower=lower
    )

    return chorus_power