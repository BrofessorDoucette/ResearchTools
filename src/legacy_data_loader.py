import datetime
import os

import numpy as np
import numpy.typing as npt
from dateutil import rrule

import os_helper
from field_models import model


def load_compressed_rept_data(
    satellite: str,
    start: datetime.datetime,
    end: datetime.datetime,
    compressed_data_dir: str = "./../compressed_data/",
) -> dict:

    rept_data_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "REPT")

    fesa = np.zeros(shape=(0, 12), dtype=np.float64)
    L = np.zeros(shape=0, dtype=np.float64)
    mlt = np.zeros(shape=0, dtype=np.float64)
    epoch: npt.NDArray[np.object_] = np.zeros(shape=0, dtype=datetime.datetime)

    print(f"Loading REPT data between: {start} and {end}.")

    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)):

        _year = str(dt.year)
        _month = str(dt.month)

        if len(_month) < 2:
            _month = f"0{_month}"

        rept_data_dir_year = os.path.join(os.path.abspath(rept_data_dir), f"{_year}")

        os_helper.verify_input_dir_exists(directory=rept_data_dir_year, hint="REPT DATA DIR")

        rept_file_name = f"REPT_{_year}{_month}_{satellite.upper()}.npz"
        rept_data_path = os.path.join(rept_data_dir_year, rept_file_name)

        if not os.path.exists(rept_data_path):
            raise Exception(f"\nData file not found: {rept_data_path}")

        print(f"Loading : {rept_file_name}")
        data = np.load(rept_data_path, allow_pickle=True)

        fesa = np.concatenate((fesa, data["FESA"]), axis=0)
        L = np.concatenate((L, data["L"]), axis=0)
        epoch = np.concatenate((epoch, data["EPOCH"]), axis=0)
        mlt = np.concatenate((mlt, data["MLT"]), axis=0)

        if i == 0:
            energies: npt.NDArray[np.float64] = data["ENERGIES"]

        data.close()

    satisfies_date_extent = (start < epoch) & (epoch < end)
    fesa = fesa[satisfies_date_extent, :]
    L = L[satisfies_date_extent]
    mlt = mlt[satisfies_date_extent]
    epoch = epoch[satisfies_date_extent]

    fesa[fesa < 0] = np.NaN

    refs = {"FESA": fesa, "L": L, "MLT": mlt, "EPOCH": epoch, "ENERGIES": energies}

    return refs


def load_psd(
    satellite: str,
    field_model: model,
    start: datetime.datetime,
    end: datetime.datetime,
    compressed_data_dir: str = "./../compressed_data/",
) -> dict:

    psd_dir = os.path.join(os.path.abspath(compressed_data_dir), "RBSP", "PSD")

    os_helper.verify_input_dir_exists(psd_dir, hint="PSD DIR")

    PSD = np.zeros((0, 35, 102), dtype=np.float64)
    JD = np.zeros((0), dtype=np.float64)
    EPOCH = np.zeros((0), dtype=datetime.datetime)

    ENERGIES = np.zeros((0, 102), dtype=np.float64)
    ALPHA = np.zeros((0, 35), dtype=np.float64)

    K = np.zeros((0, 35), dtype=np.float64)
    L_STAR = np.zeros((0, 35), dtype=np.float64)
    L = np.zeros((0, 35), np.float64)
    IN_OUT = np.zeros((0), dtype=np.int32)
    ORBIT_NUMBER = np.zeros((0), dtype=np.int32)

    B = np.zeros((0), dtype=np.float64)

    for i, dt in enumerate(
        rrule.rrule(
            rrule.MONTHLY,
            dtstart=datetime.datetime(year=start.year, month=start.month, day=1),
            until=end,
        )
    ):

        print(dt)

        _year = str(dt.year)
        _month = str(dt.month)

        if len(_month) < 2:
            _month = f"0{_month}"

        file_name = f"PSD_{_year}{_month}_{satellite.upper()}_{field_model.name}.npz"

        psd_path = os.path.join(psd_dir, file_name)

        if not os.path.exists(psd_path):
            raise Exception(f"\nData file not found: {psd_path}!")

        print(f"Loading : {file_name}")
        data = np.load(psd_path, allow_pickle=True)

        PSD = np.concatenate((PSD, data["PSD"]), axis=0)
        JD = np.concatenate((JD, data["JD"]), axis=0)
        EPOCH = np.concatenate((EPOCH, data["EPOCH"]), axis=0)

        ENERGIES = np.concatenate((ENERGIES, data["ENERGIES"]), axis=0)
        ALPHA = np.concatenate((ALPHA, data["ALPHA"]), axis=0)

        K = np.concatenate((K, data["K"]), axis=0)
        L_STAR = np.concatenate((L_STAR, data["L_STAR"]), axis=0)
        L = np.concatenate((L, data["L"]), axis=0)
        IN_OUT = np.concatenate((IN_OUT, data["IN_OUT"]), axis=0)
        ORBIT_NUMBER = np.concatenate((ORBIT_NUMBER, data["ORBIT_NUMBER"]), axis=0)

        B = np.concatenate((B, data["B"]), axis=0)

        data.close()

    satisfies_timespan = (start < EPOCH) & (EPOCH < end)
    PSD = PSD[satisfies_timespan, :, :]
    JD = JD[satisfies_timespan]
    EPOCH = EPOCH[satisfies_timespan]
    ENERGIES = ENERGIES[satisfies_timespan, :]
    ALPHA = ALPHA[satisfies_timespan, :]
    K = K[satisfies_timespan, :]
    L_STAR = L_STAR[satisfies_timespan, :]
    L = L[satisfies_timespan, :]
    IN_OUT = IN_OUT[satisfies_timespan]
    ORBIT_NUMBER = ORBIT_NUMBER[satisfies_timespan]
    B = B[satisfies_timespan]

    refs = {
        "PSD": PSD,
        "JD": JD,
        "EPOCH": EPOCH,
        "ENERGIES": ENERGIES,
        "ALPHA": ALPHA,
        "K": K,
        "L_STAR": L_STAR,
        "L": L,
        "IN_OUT": IN_OUT,
        "ORBIT_NUMBER": ORBIT_NUMBER,
        "B": B,
    }

    return refs
