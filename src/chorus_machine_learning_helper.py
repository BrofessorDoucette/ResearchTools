import data_loader
import datetime
from cdflib.epochs_astropy import CDFAstropy as cdfepoch
import numpy as np
import astropy.time
import pandas as pd
import tqdm
import os


def load_MPE_year(year: int) -> list:

    POES = []

    for SAT in ["m01", "m02", "m03", "n15", "n16", "n17", "n18", "n19"]:

        POES_sat_refs = data_loader.load_raw_data_from_config(
            id=["POES", "SEM", "MPE"],
            satellite=SAT,
            start=datetime.datetime(year=year, month=1, day=1, hour=0, minute=0, second=0, tzinfo=datetime.UTC),
            end=datetime.datetime(year=year, month=12, day=31, hour=23, minute=59, second=59, tzinfo=datetime.UTC),
        )

        if POES_sat_refs:

            # This was done cause I wanted to scale the MLT before cleaning but Im lazy
            if year < 2014:
                POES_sat_refs["MLT"] = (POES_sat_refs["MLT"] / 360.0) * 24.0

            valid_times = np.isfinite(POES_sat_refs["time"]) & (0 < POES_sat_refs["time"])
            valid_BLC_Angle = np.isfinite(POES_sat_refs["BLC_Angle"]) & (
                0 < POES_sat_refs["BLC_Angle"]
            )
            valid_BLC_Flux = np.all(
                np.isfinite(POES_sat_refs["BLC_Flux"][:, :]), axis=1
            ) & np.all((0 < POES_sat_refs["BLC_Flux"][:, :]), axis=1)
            valid_MLT = (
                np.isfinite(POES_sat_refs["MLT"]) & (0 < POES_sat_refs["MLT"]) & (POES_sat_refs["MLT"] < 24)
            )
            valid_L = (
                np.isfinite(POES_sat_refs["lValue"]) & (0 < POES_sat_refs["lValue"]) & (POES_sat_refs["lValue"] < 20)
            )
            valid_geogLat = (
                np.isfinite(POES_sat_refs["geogLat"]) & (-90.0 <= POES_sat_refs["geogLat"]) & (POES_sat_refs["geogLat"] <= 90.0)
            )
            valid_geogLon = (
                np.isfinite(POES_sat_refs["geogLon"]) & (0.0 <= POES_sat_refs["geogLon"]) & (POES_sat_refs["geogLon"] <= 360.0)
            )

            valid_points = (
                valid_times & valid_BLC_Angle & valid_BLC_Flux & valid_MLT & valid_L & valid_geogLat & valid_geogLon
            )

            if np.any(valid_points):

                POES_sat_refs["time"] = POES_sat_refs["time"][valid_points]
                POES_sat_refs["BLC_Angle"] = POES_sat_refs["BLC_Angle"][valid_points]
                POES_sat_refs["BLC_Flux"] = POES_sat_refs["BLC_Flux"][valid_points, :]
                POES_sat_refs["MLT"] = POES_sat_refs["MLT"][valid_points]
                POES_sat_refs["lValue"] = POES_sat_refs["lValue"][valid_points]
                POES_sat_refs["geogLat"] = POES_sat_refs["geogLat"][valid_points]
                POES_sat_refs["geogLon"] = POES_sat_refs["geogLon"][valid_points]

                if year < 2014:

                    POES_sat_refs["UNIX_TIME"] = cdfepoch.unixtime(POES_sat_refs["time"])
                else:
                    POES_sat_refs["UNIX_TIME"] = POES_sat_refs["time"] / 1000

                # Sort them so assumptions for binary search are satisfied:
                order = np.argsort(POES_sat_refs["UNIX_TIME"])
                POES_sat_refs["time"] = POES_sat_refs["time"][order]
                POES_sat_refs["UNIX_TIME"] = POES_sat_refs["UNIX_TIME"][order]
                POES_sat_refs["BLC_Angle"] = POES_sat_refs["BLC_Angle"][order]
                POES_sat_refs["BLC_Flux"] = POES_sat_refs["BLC_Flux"][order, :]
                POES_sat_refs["MLT"] = POES_sat_refs["MLT"][order]
                POES_sat_refs["L"] = POES_sat_refs["lValue"][order]
                POES_sat_refs["geogLat"] = POES_sat_refs["geogLat"][order]
                POES_sat_refs["geogLon"] = POES_sat_refs["geogLon"][order]
                POES_sat_refs["SATID"] = SAT
                POES.append(POES_sat_refs)

    if not POES:
        print(f"No POES satellite coverage found for year : {year}")

    print(f"Finished loading POES data for year : {year}")

    return POES


def load_SUPERMAG_SME_year(year: int):

    print(f"Began loading SUPERMAG data for year : {year}")
    SUPERMAG_df = pd.read_csv(rf"./../processed_data/chorus_neural_network/SUPERMAG_SME/sme_{year}.csv")
    SUPERMAG = {}

    valid_SME = np.isfinite(SUPERMAG_df["SME"]) & (0 < SUPERMAG_df["SME"])

    if not np.any(valid_SME):
        print(f"No valid SME for year : {year}")

    SUPERMAG["SME"] = np.array(SUPERMAG_df["SME"][valid_SME])
    SUPERMAG["Date_UTC"] = np.array(SUPERMAG_df["Date_UTC"][valid_SME])
    SUPERMAG["UNIX_TIME"] = astropy.time.Time(
        SUPERMAG["Date_UTC"].astype(str), scale="utc", in_subfmt="date_hms"
    ).unix

    order = np.argsort(SUPERMAG["UNIX_TIME"])
    SUPERMAG["SME"] = SUPERMAG["SME"][order]
    SUPERMAG["UNIX_TIME"] = SUPERMAG["UNIX_TIME"][order]
    del SUPERMAG["Date_UTC"]
    print(f"Finished loading SUPERMAG data for year : {year}")

    return SUPERMAG


def load_OMNI_year(year: int) -> dict:

    print(f"Began loading OMNI data for year : {year}")
    OMNI_refs = data_loader.load_raw_data_from_config(
        id=["OMNI", "ONE_MIN_RESOLUTION"],
        start=datetime.datetime(year=year, month=1, day=1, hour=0, minute=0, second=0),
        end=datetime.datetime(year=year, month=12, day=31, hour=23, minute=59, second=59)
    )
    OMNI = {}

    valid_times = np.isfinite(OMNI_refs["Epoch"]) & (0 < OMNI_refs["Epoch"])

    OMNI["UNIX_TIME"] = cdfepoch.unixtime(OMNI_refs["Epoch"][valid_times])
    OMNI["AVG_B"] = OMNI_refs["F"][valid_times]
    OMNI["FLOW_SPEED"] = OMNI_refs["flow_speed"][valid_times]
    OMNI["PROTON_DENSITY"] = OMNI_refs["proton_density"][valid_times]
    OMNI["SYM_H"] = OMNI_refs["SYM_H"][valid_times]

    valid_AVG_B = np.isfinite(OMNI["AVG_B"]) & (0 <= OMNI["AVG_B"]) & (OMNI["AVG_B"] < 9990)
    valid_FLOW_SPEED = np.isfinite(OMNI["FLOW_SPEED"]) & (0 <= OMNI["FLOW_SPEED"]) & (OMNI["FLOW_SPEED"] < 99900)
    valid_PROTON_DENSITY = np.isfinite(OMNI["PROTON_DENSITY"]) & (-900 <= OMNI["PROTON_DENSITY"]) & (OMNI["PROTON_DENSITY"] < 900)
    valid_SYM_H = np.isfinite(OMNI["SYM_H"]) & (-99000 <= OMNI["SYM_H"]) & (OMNI["SYM_H"] < 99900)
    valid_points = valid_AVG_B & valid_FLOW_SPEED & valid_PROTON_DENSITY & valid_SYM_H

    if not np.any(valid_points):
        print(f"No valid OMNI DATA for year : {year}")
        print(f"SKIPPING YEAR : {year}")

    OMNI["AVG_B"][~valid_points] = np.nan
    OMNI["FLOW_SPEED"][~valid_points] = np.nan
    OMNI["PROTON_DENSITY"][~valid_points] = np.nan
    OMNI["SYM_H"] = OMNI["SYM_H"].astype(np.float32)
    OMNI["SYM_H"][~valid_points] = np.nan

    if (np.max(OMNI["UNIX_TIME"][1:] - OMNI["UNIX_TIME"][:-1])) > 300:
        raise Exception(
            "Tried to interpolate OMNI data but large gaps that are unexpected were present!"
        )

    start_interpolation_time = datetime.datetime(year=year, month=1, day=1).timestamp()
    end_interpolation_time = datetime.datetime(year=year + 1, month=1, day=1).timestamp()
    evenly_spaced_seconds = np.arange(
        start=start_interpolation_time, stop=end_interpolation_time + 1, step=1
    )

    OMNI["AVG_B"] = np.interp(x=evenly_spaced_seconds, xp=OMNI["UNIX_TIME"], fp=OMNI["AVG_B"])
    OMNI["FLOW_SPEED"] = np.interp(x=evenly_spaced_seconds, xp=OMNI["UNIX_TIME"], fp=OMNI["FLOW_SPEED"])
    OMNI["PROTON_DENSITY"] = np.interp(x=evenly_spaced_seconds, xp=OMNI["UNIX_TIME"], fp=OMNI["PROTON_DENSITY"])
    OMNI["SYM_H"] = np.interp(x=evenly_spaced_seconds, xp=OMNI["UNIX_TIME"], fp=OMNI["SYM_H"])

    order = np.argsort(evenly_spaced_seconds)
    OMNI["UNIX_TIME"] = evenly_spaced_seconds[order]
    OMNI["AVG_B"] = OMNI["AVG_B"][order]
    OMNI["FLOW_SPEED"] = OMNI["FLOW_SPEED"][order]
    OMNI["PROTON_DENSITY"] = OMNI["PROTON_DENSITY"][order]
    OMNI["SYM_H"] = OMNI["SYM_H"][order]

    not_nan = (
        np.isfinite(OMNI["AVG_B"]) & np.isfinite(OMNI["FLOW_SPEED"]) & np.isfinite(OMNI["PROTON_DENSITY"]) & np.isfinite(OMNI["SYM_H"])
    )
    OMNI["UNIX_TIME"] = OMNI["UNIX_TIME"][not_nan]
    OMNI["AVG_B"] = OMNI["AVG_B"][not_nan]
    OMNI["FLOW_SPEED"] = OMNI["FLOW_SPEED"][not_nan]
    OMNI["PROTON_DENSITY"] = OMNI["PROTON_DENSITY"][not_nan]
    OMNI["SYM_H"] = OMNI["SYM_H"][not_nan]

    print(f"Finished loading OMNI data for year : {year}")

    return OMNI

def find_max_in_last_12h(timestamps, values):
    """
    Find the maximum value in the preceding 24 hours for each timestamp.

    Args:
        timestamps: List of Unix timestamps (in seconds)
        values: List of corresponding values

    Returns:
        List of maximum values for each timestamp, considering the previous 24 hours
    """
    if len(timestamps) != len(values):
        raise ValueError("Timestamps and values arrays must have the same length")

    result = np.zeros_like(timestamps)
    result[:] = np.nan
    # 12 hours in seconds
    twenty_four_hours = 12 * 60 * 60

    for i, current_time in enumerate(timestamps):
        # Find all values within the last 24 hours
        within_window = ((current_time - twenty_four_hours) <= timestamps) & (timestamps <= current_time)
        window_values = values[within_window]

        if np.any(np.isnan(window_values)):
            continue

        # If no values in the window, append None or handle as needed
        result[i] = np.max(window_values)

    return result


def load_hp30(path):

    hp = pd.read_csv(os.path.abspath(path), delim_whitespace=True)

    start = datetime.datetime(year=1932, month=1, day=1, tzinfo=datetime.UTC)
    unix_times = []
    for t in range(len(hp["days_m"])):

        mid_of_interval = start + datetime.timedelta(days= hp["days_m"][t])
        unix_times.append(mid_of_interval.timestamp())

    order = np.argsort(unix_times)

    return np.asarray(unix_times)[order], np.asarray(hp["Hp30"])[order]

    
def load_plasmapause_filter(year, return_kp=False):

    start = datetime.datetime(year=year-1, month=1, day=1)
    end = datetime.datetime(year=year+2, month=1, day=1)

    OMNI = data_loader.load_raw_data_from_config(
        id=["OMNI", "ONE_HOUR_RESOLUTION"], start=start, end=end
    )

    UNIX_TIME = cdfepoch.unixtime(OMNI["Epoch"])
    KP = OMNI["KP"].astype(np.float64)

    invalid_omni_times = (UNIX_TIME < 0) | (KP < 0) | (KP >= 99)
    KP[invalid_omni_times] = np.nan

    kp_max = find_max_in_last_12h(UNIX_TIME, KP / 10.0)

    start_of_year = datetime.datetime(year=year, month=1, day=1, tzinfo=datetime.UTC).timestamp()
    end_of_year = datetime.datetime(year=year + 1, month=1, day=1, tzinfo=datetime.UTC).timestamp()
    within_year = (start_of_year <= UNIX_TIME) & (UNIX_TIME < end_of_year)

    L_ppi = 5.99 - 0.382 * kp_max # https://doi.org/10.1029/2001JA009211

    if return_kp:

        return UNIX_TIME[within_year], L_ppi[within_year], kp_max[within_year]

    else:

        return UNIX_TIME[within_year], L_ppi[within_year]

