import numpy as np


def emfisis_noise_floor(f):

    x = [1, 10, 100, 1000, 10000]
    y = [1e-4, 1e-5, 1e-8, 1e-10, 1e-10]

    return np.exp(np.interp(np.log(f), np.log(x), np.log(y), left=np.nan, right=np.nan))


def calculate_chorus_amplitudes_from_bsum(B_uvw, B_sum, WFR_bandwidths, WFR_frequencies) -> tuple:

    m_e = 9.10938356e-31  # Mass of electron in kg
    q = 1.60217662e-19  # Charge of electron

    B = np.sqrt(B_uvw[:, 0] ** 2 + B_uvw[:, 1] ** 2 + B_uvw[:, 2] ** 2)
    B = B / (1e9)  # Magnetic field magnitude in Tesla

    w_ce = (q * B) / (m_e)
    f_ce = w_ce / (2 * np.pi)  # Cyclotron frequency in Hz

    min_f_for_lower_chorus_band = 5 * (f_ce / 100.0)
    max_f_for_lower_chorus_band = 5 * (f_ce / 10.0)

    min_f_for_upper_chorus_band = 5 * (f_ce / 10.0)
    max_f_for_upper_chorus_band = f_ce

    frequency_bin_minimums = WFR_frequencies - (WFR_bandwidths / 2)
    frequency_bin_maximums = WFR_frequencies + (WFR_bandwidths / 2)

    noise_floor = emfisis_noise_floor(WFR_frequencies)

    lower_band_chorus = []
    upper_band_chorus = []

    for T in range(B_uvw.shape[0]):

        lower_band_frequency_bins = (min_f_for_lower_chorus_band[T] <= frequency_bin_minimums) & (frequency_bin_maximums < max_f_for_lower_chorus_band[T])
        upper_band_frequency_bins = (min_f_for_upper_chorus_band[T] <= frequency_bin_minimums) & (frequency_bin_maximums < max_f_for_upper_chorus_band[T])

        if np.any(lower_band_frequency_bins):

            bandwidths = WFR_bandwidths[lower_band_frequency_bins]
            power_hz = B_sum[T, lower_band_frequency_bins]

            greater_than_noise_floor = (power_hz > 10 * noise_floor[lower_band_frequency_bins])  # Should be an order of magnitude greater than noise floor

            if np.any(greater_than_noise_floor):

                power = np.nansum(power_hz[greater_than_noise_floor] * bandwidths[greater_than_noise_floor])
                amplitude = np.sqrt(power)
                lower_band_chorus.append(amplitude * 1000.0)  # Convert from nT to pT
            else:
                lower_band_chorus.append(0)
        else:
            lower_band_chorus.append(np.nan)

        if np.any(upper_band_frequency_bins):

            bandwidths = WFR_bandwidths[upper_band_frequency_bins]
            power_hz = B_sum[T, upper_band_frequency_bins]

            greater_than_noise_floor = (power_hz > 10 * noise_floor[upper_band_frequency_bins])  # Should be an order of magnitude greater than noise floor

            if np.any(greater_than_noise_floor):

                power = np.nansum(power_hz[greater_than_noise_floor] * bandwidths[greater_than_noise_floor])
                amplitude = np.sqrt(power)
                upper_band_chorus.append(amplitude * 1000.0)  # Convert from nT to pT
            else:
                upper_band_chorus.append(0)
        else:
            upper_band_chorus.append(np.nan)

    return lower_band_chorus, upper_band_chorus


def iterate_through_days_and_calculate_chorus_amplitudes(WNA_survey, WFR_spectral_matrix) -> dict:

    lower_band_chorus = []
    upper_band_chorus = []

    # This makes sure that the same number of days were loaded
    assert len(WNA_survey["timestamps_per_file"]) == WFR_spectral_matrix["WFR_bandwidth"].shape[0]

    index = 0
    for day in range(len(WNA_survey["timestamps_per_file"])):

        num_timesteps_for_day = WNA_survey["timestamps_per_file"][day]

        chorus_calculated = calculate_chorus_amplitudes_from_bsum(
            WNA_survey["Buvw"][index : index + num_timesteps_for_day],
            WNA_survey["bsum"][index : index + num_timesteps_for_day],
            WFR_spectral_matrix["WFR_bandwidth"][day, :],
            WFR_spectral_matrix["WFR_frequencies"][day, :],
        )

        lower_band_chorus.extend(chorus_calculated[0])
        upper_band_chorus.extend(chorus_calculated[1])

        index = index + num_timesteps_for_day

    return {
        "Lower_Band": np.asarray(lower_band_chorus),
        "Upper_Band": np.asarray(upper_band_chorus),
    }
