import datetime
import multiprocessing as mp
import os

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import chorus_machine_learning_helper


def find_conjunctions(CHUNK_TIME, POES, RBSP, SUPERMAG, OMNI, T_SIZE, L_SIZE, MLT_SIZE, L_MIN, L_MAX):

    CONJUNCTIONS = []

    L_EDGES = np.arange(L_MIN, L_MAX, step=L_SIZE)
    MLT_EDGES = np.arange(0, 24, step=MLT_SIZE)

    CUM_FLUX_0 = np.zeros(shape=(L_EDGES.shape[0], MLT_EDGES.shape[0]))
    CUM_FLUX_1 = np.zeros_like(CUM_FLUX_0)
    CUM_FLUX_2 = np.zeros_like(CUM_FLUX_0)
    CUM_FLUX_3 = np.zeros_like(CUM_FLUX_0)
    CUM_FLUX_4 = np.zeros_like(CUM_FLUX_0)
    CUM_FLUX_5 = np.zeros_like(CUM_FLUX_0)
    CUM_FLUX_6 = np.zeros_like(CUM_FLUX_0)
    CUM_FLUX_7 = np.zeros_like(CUM_FLUX_0)
    CUM_L_POES = np.zeros_like(CUM_FLUX_0)
    CUM_MLT_POES = np.zeros_like(CUM_FLUX_0)
    NUM_IN_EACH_POES_BIN = np.zeros_like(CUM_FLUX_0)

    CUM_LOWER_BAND_CHORUS_RBSP = np.zeros(shape=(L_EDGES.shape[0], MLT_EDGES.shape[0]))
    CUM_UPPER_BAND_CHORUS_RBSP = np.zeros_like(CUM_LOWER_BAND_CHORUS_RBSP)
    CUM_L_RBSP = np.zeros_like(CUM_LOWER_BAND_CHORUS_RBSP)
    CUM_MLT_RBSP = np.zeros_like(CUM_LOWER_BAND_CHORUS_RBSP)
    NUM_IN_EACH_RBSP_BIN = np.zeros_like(CUM_LOWER_BAND_CHORUS_RBSP)

    for SATID in POES:

        L_mapping = np.digitize(POES[SATID]["LSTAR"], L_EDGES, right=False) - 1
        MLT_mapping = np.digitize(POES[SATID]["MLT"], MLT_EDGES, right=False) - 1

        for T in range(len(POES[SATID]["UNIX_TIME"])):

            x_bin = L_mapping[T]
            y_bin = MLT_mapping[T]

            CUM_FLUX_0[x_bin, y_bin] += POES[SATID]["BLC_Flux"][T, 0]
            CUM_FLUX_1[x_bin, y_bin] += POES[SATID]["BLC_Flux"][T, 1]
            CUM_FLUX_2[x_bin, y_bin] += POES[SATID]["BLC_Flux"][T, 2]
            CUM_FLUX_3[x_bin, y_bin] += POES[SATID]["BLC_Flux"][T, 3]
            CUM_FLUX_4[x_bin, y_bin] += POES[SATID]["BLC_Flux"][T, 4]
            CUM_FLUX_5[x_bin, y_bin] += POES[SATID]["BLC_Flux"][T, 5]
            CUM_FLUX_6[x_bin, y_bin] += POES[SATID]["BLC_Flux"][T, 6]
            CUM_FLUX_7[x_bin, y_bin] += POES[SATID]["BLC_Flux"][T, 7]
            CUM_L_POES[x_bin, y_bin] += POES[SATID]["LSTAR"][T]
            CUM_MLT_POES[x_bin, y_bin] += POES[SATID]["MLT"][T]
            NUM_IN_EACH_POES_BIN[x_bin, y_bin] += 1

    for PROBE in RBSP:

        L_mapping = np.digitize(PROBE["LSTAR"], L_EDGES, right=False) - 1
        MLT_mapping = np.digitize(PROBE["MLT"], MLT_EDGES, right=False) - 1

        for T in range(len(PROBE["UNIX_TIME"])):

            x_bin = L_mapping[T]
            y_bin = MLT_mapping[T]

            CUM_LOWER_BAND_CHORUS_RBSP[x_bin, y_bin] += PROBE["LOWER_BAND"][T]
            CUM_UPPER_BAND_CHORUS_RBSP[x_bin, y_bin] += PROBE["UPPER_BAND"][T]
            CUM_L_RBSP[x_bin, y_bin] += PROBE["LSTAR"][T]
            CUM_MLT_RBSP[x_bin, y_bin] += PROBE["MLT"][T]
            NUM_IN_EACH_RBSP_BIN[x_bin, y_bin] += 1

    AVG_FLUX_0 = CUM_FLUX_0 / NUM_IN_EACH_POES_BIN
    AVG_FLUX_1 = CUM_FLUX_1 / NUM_IN_EACH_POES_BIN
    AVG_FLUX_2 = CUM_FLUX_2 / NUM_IN_EACH_POES_BIN
    AVG_FLUX_3 = CUM_FLUX_3 / NUM_IN_EACH_POES_BIN
    AVG_FLUX_4 = CUM_FLUX_4 / NUM_IN_EACH_POES_BIN
    AVG_FLUX_5 = CUM_FLUX_5 / NUM_IN_EACH_POES_BIN
    AVG_FLUX_6 = CUM_FLUX_6 / NUM_IN_EACH_POES_BIN
    AVG_FLUX_7 = CUM_FLUX_7 / NUM_IN_EACH_POES_BIN
    AVG_L_POES = CUM_L_POES / NUM_IN_EACH_POES_BIN
    AVG_MLT_POES = CUM_MLT_POES / NUM_IN_EACH_POES_BIN

    AVG_LOWER_CHORUS = CUM_LOWER_BAND_CHORUS_RBSP / NUM_IN_EACH_RBSP_BIN
    AVG_UPPER_CHORUS = CUM_UPPER_BAND_CHORUS_RBSP / NUM_IN_EACH_RBSP_BIN
    AVG_L_RBSP = CUM_L_RBSP / NUM_IN_EACH_RBSP_BIN
    AVG_MLT_RBSP = CUM_MLT_RBSP / NUM_IN_EACH_RBSP_BIN

    if (np.nanmax(NUM_IN_EACH_POES_BIN) == 0) or (np.nanmax(NUM_IN_EACH_RBSP_BIN) == 0):
        return [[], np.nan, np.nan]

    for x_bin in range(L_EDGES.shape[0]):

        for y_bin in range(MLT_EDGES.shape[0]):

            if ((0 < NUM_IN_EACH_POES_BIN[x_bin, y_bin]) and (0 < NUM_IN_EACH_RBSP_BIN[x_bin, y_bin])):

                CONJUNCTION = [
                    CHUNK_TIME + (T_SIZE / 2.0),
                    AVG_L_POES[x_bin, y_bin],
                    AVG_MLT_POES[x_bin, y_bin],
                    AVG_FLUX_0[x_bin, y_bin],
                    AVG_FLUX_1[x_bin, y_bin],
                    AVG_FLUX_2[x_bin, y_bin],
                    AVG_FLUX_3[x_bin, y_bin],
                    AVG_FLUX_4[x_bin, y_bin],
                    AVG_FLUX_5[x_bin, y_bin],
                    AVG_FLUX_6[x_bin, y_bin],
                    AVG_FLUX_7[x_bin, y_bin],
                    CHUNK_TIME + (T_SIZE / 2.0),
                    AVG_L_RBSP[x_bin, y_bin],  # LSTAR OF RBSP POINT CHOSEN
                    AVG_MLT_RBSP[x_bin, y_bin],  # DIFFERENCE IN MLT FOUND
                    AVG_UPPER_CHORUS[x_bin, y_bin],  # UPPER BAND CHORUS OBSERVED
                    AVG_LOWER_CHORUS[x_bin, y_bin],  # LOWER BAND CHORUS OBSERVED
                    SUPERMAG["SME"],
                    OMNI["AVG_B"],
                    OMNI["FLOW_SPEED"],
                    OMNI["PROTON_DENSITY"],
                    OMNI["SYM_H"],
                ]

                CONJUNCTIONS.append(CONJUNCTION)

    return [CONJUNCTIONS, np.nanmedian(NUM_IN_EACH_POES_BIN[NUM_IN_EACH_POES_BIN > 0]), np.nanmedian(NUM_IN_EACH_RBSP_BIN[NUM_IN_EACH_RBSP_BIN > 0])]


if __name__ == "__main__":

    # Stage 2, clean then combine RBSP, OMNI, and POES Data and find conjunctions between RBSP and POES

    VERSION = "v2b"
    FIELD_MODEL = "T89"

    pdata_folder = os.path.abspath("./../processed_data/chorus_neural_network/")
    rbsp_chorus_folder = os.path.join(pdata_folder, "STAGE_1", "CHORUS")
    rbsp_lstar_folder = os.path.join(pdata_folder, "STAGE_1", "Lstar")
    mpe_folder = os.path.join(pdata_folder, "STAGE_0", "MPE_DATA_PREPROCESSED_WITH_LSTAR")
    output_folder = os.path.join(pdata_folder, "STAGE_2", VERSION)

    T_SIZE = 3600
    L_SIZE = 0.025
    MLT_SIZE = 2.0
    L_MIN = 3.5
    L_MAX = 9

    CONJUNCTIONS_TOTAL = []

    for _year in range(2012, 2020, 1):

        print(f"Began processing year : {_year}")

        # LOAD THE OBSERVED CHORUS
        print(f"Began loading RBSP Data for year: {_year}")
        refs = np.load(
            file=os.path.join(rbsp_chorus_folder, rf"RBSP_OBSERVED_CHORUS_{_year}.npz"),
            allow_pickle=True,
        )
        RBSP_A = {}
        RBSP_A["UNIX_TIME"] = refs["UNIX_TIME_A"]
        RBSP_A["MLT"] = refs["MLT_A"]
        RBSP_A["L"] = refs["L_A"]
        RBSP_A["LOWER_BAND"] = refs["LOWER_BAND_CHORUS_A"]
        RBSP_A["UPPER_BAND"] = refs["UPPER_BAND_CHORUS_A"]

        RBSP_B = {}
        RBSP_B["UNIX_TIME"] = refs["UNIX_TIME_B"]
        RBSP_B["MLT"] = refs["MLT_B"]
        RBSP_B["L"] = refs["L_B"]
        RBSP_B["LOWER_BAND"] = refs["LOWER_BAND_CHORUS_B"]
        RBSP_B["UPPER_BAND"] = refs["UPPER_BAND_CHORUS_B"]

        refs.close()

        # LOAD THE LSTAR AND INTERPOLATE
        refs_A = np.load(
            file=os.path.join(rbsp_lstar_folder, rf"RBSP_A_{FIELD_MODEL}_{_year}.npz"),
            allow_pickle=True,
        )

        MAGEPHEM_TIME_A = refs_A["UNIX_TIME"]
        MAGEPHEM_LSTAR_A = refs_A["Lstar"]
        MAGEPHEM_L_A = refs_A["L"]

        refs_A.close()

        refs_B = np.load(
            file=os.path.join(rbsp_lstar_folder, rf"RBSP_B_{FIELD_MODEL}_{_year}.npz"),
            allow_pickle=True,
        )

        MAGEPHEM_TIME_B = refs_B["UNIX_TIME"]
        MAGEPHEM_LSTAR_B = refs_B["Lstar"]
        MAGEPHEM_L_B = refs_B["L"]

        refs_B.close()

        # PREPROCESS DATA

        RBSP_A["LSTAR"] = np.interp(
            RBSP_A["UNIX_TIME"], MAGEPHEM_TIME_A, MAGEPHEM_LSTAR_A, left=np.nan, right=np.nan
        )
        RBSP_B["LSTAR"] = np.interp(
            RBSP_B["UNIX_TIME"], MAGEPHEM_TIME_B, MAGEPHEM_LSTAR_B, left=np.nan, right=np.nan
        )

        order_A = np.argsort(RBSP_A["UNIX_TIME"])
        order_B = np.argsort(RBSP_B["UNIX_TIME"])

        RBSP_A["UNIX_TIME"] = RBSP_A["UNIX_TIME"][order_A]
        RBSP_A["MLT"] = RBSP_A["MLT"][order_A]
        RBSP_A["L"] = RBSP_A["L"][order_A]
        RBSP_A["LSTAR"] = RBSP_A["LSTAR"][order_A]
        RBSP_A["LOWER_BAND"] = RBSP_A["LOWER_BAND"][order_A]
        RBSP_A["UPPER_BAND"] = RBSP_A["UPPER_BAND"][order_A]

        RBSP_B["UNIX_TIME"] = RBSP_B["UNIX_TIME"][order_B]
        RBSP_B["MLT"] = RBSP_B["MLT"][order_B]
        RBSP_B["L"] = RBSP_B["L"][order_B]
        RBSP_B["LSTAR"] = RBSP_B["LSTAR"][order_B]
        RBSP_B["LOWER_BAND"] = RBSP_B["LOWER_BAND"][order_B]
        RBSP_B["UPPER_BAND"] = RBSP_B["UPPER_BAND"][order_B]

        between_bins_A = (L_MIN <= RBSP_A["LSTAR"]) & (RBSP_A["LSTAR"] < L_MAX) & (0 <= RBSP_A["MLT"]) & (RBSP_A["MLT"] < 24)
        between_bins_B = (L_MIN <= RBSP_B["LSTAR"]) & (RBSP_B["LSTAR"] < L_MAX) & (0 <= RBSP_B["MLT"]) & (RBSP_B["MLT"] < 24)

        RBSP_A["UNIX_TIME"] = RBSP_A["UNIX_TIME"][between_bins_A]
        RBSP_A["MLT"] = RBSP_A["MLT"][between_bins_A]
        RBSP_A["L"] = RBSP_A["L"][between_bins_A]
        RBSP_A["LSTAR"] = RBSP_A["LSTAR"][between_bins_A]
        RBSP_A["LOWER_BAND"] = RBSP_A["LOWER_BAND"][between_bins_A]
        RBSP_A["UPPER_BAND"] = RBSP_A["UPPER_BAND"][between_bins_A]

        RBSP_B["UNIX_TIME"] = RBSP_B["UNIX_TIME"][between_bins_B]
        RBSP_B["MLT"] = RBSP_B["MLT"][between_bins_B]
        RBSP_B["L"] = RBSP_B["L"][between_bins_B]
        RBSP_B["LSTAR"] = RBSP_B["LSTAR"][between_bins_B]
        RBSP_B["LOWER_BAND"] = RBSP_B["LOWER_BAND"][between_bins_B]
        RBSP_B["UPPER_BAND"] = RBSP_B["UPPER_BAND"][between_bins_B]

        RBSP = [RBSP_A, RBSP_B]
        print(f"RBSP Data loaded for year : {_year}")

        print(f"Began loading POES Data for year : {_year}")

        POES_REFS = np.load(
            file=os.path.join(mpe_folder, rf"MPE_PREPROCESSED_DATA_{FIELD_MODEL}_{_year}_interpolated.npz"),
            allow_pickle=True,
        )
        POES = POES_REFS["DATA"].flatten()[0]

        for SATID in POES:

            between_bins_POES = (L_MIN <= POES[SATID]["LSTAR"]) & (POES[SATID]["LSTAR"] < L_MAX) & (0 <= POES[SATID]["MLT"]) & (POES[SATID]["MLT"] < 24)

            POES[SATID] = {
                "UNIX_TIME" : POES[SATID]["UNIX_TIME"][between_bins_POES],
                "LSTAR" : POES[SATID]["LSTAR"][between_bins_POES],
                "MLT" : POES[SATID]["MLT"][between_bins_POES],
                "BLC_Flux" : POES[SATID]["BLC_Flux"][between_bins_POES, :]
            }

        print(f"Finished loading POES data for year : {_year}")

        OMNI = chorus_machine_learning_helper.load_OMNI_year(_year)
        SUPERMAG = chorus_machine_learning_helper.load_SUPERMAG_SME_year(_year)

        # FINALLY FIND THE CONJUNCTIONS

        start_of_year = datetime.datetime(year=_year, month=1, day=1).timestamp()
        end_of_year = datetime.datetime(year=_year + 1, month=1, day=1).timestamp()

        print("Separating into chunks!")
        CONJUNCTIONS_YEAR = []

        num_processes = mp.cpu_count() - 2
        pool = mp.Pool(processes=num_processes)

        chunk_times = np.arange(start_of_year, end_of_year, step=T_SIZE)

        CHUNKS_TO_PROCESS = []
        for T in chunk_times:

            POES_CHUNK = {}

            for SATID in POES:

                TIME_RANGE_POES = np.searchsorted(
                    a=POES[SATID]["UNIX_TIME"],
                    v=[T, T + T_SIZE],
                )

                POES_CHUNK[SATID] = {
                    "UNIX_TIME" : POES[SATID]["UNIX_TIME"][TIME_RANGE_POES[0] : TIME_RANGE_POES[1]],
                    "LSTAR" : POES[SATID]["LSTAR"][TIME_RANGE_POES[0] : TIME_RANGE_POES[1]],
                    "MLT" : POES[SATID]["MLT"][TIME_RANGE_POES[0] : TIME_RANGE_POES[1]],
                    "BLC_Flux" : POES[SATID]["BLC_Flux"][TIME_RANGE_POES[0] : TIME_RANGE_POES[1], :]
                }

            RBSP_CHUNK = []

            for RBSP_PROBE in RBSP:

                TIME_RANGE_RBSP = np.searchsorted(
                    a=RBSP_PROBE["UNIX_TIME"],
                    v=[T, T + T_SIZE],
                )

                RBSP_PROBE_CHUNK = {
                    "UNIX_TIME" : RBSP_PROBE["UNIX_TIME"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]],
                    "MLT" : RBSP_PROBE["MLT"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]],
                    "L" : RBSP_PROBE["L"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]],
                    "LSTAR" : RBSP_PROBE["LSTAR"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]],
                    "LOWER_BAND" : RBSP_PROBE["LOWER_BAND"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]],
                    "UPPER_BAND" : RBSP_PROBE["UPPER_BAND"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]]
                }

                RBSP_CHUNK.append(RBSP_PROBE_CHUNK)

            TIME_RANGE_SUPERMAG = np.searchsorted(
                a=SUPERMAG["UNIX_TIME"],
                v=[T, T + T_SIZE],
            )

            SUPERMAG_CHUNK = {
                "UNIX_TIME" : np.nanmean(SUPERMAG["UNIX_TIME"][TIME_RANGE_SUPERMAG[0] : TIME_RANGE_SUPERMAG[1]]),
                "SME" : np.nanmean(SUPERMAG["SME"][TIME_RANGE_SUPERMAG[0] : TIME_RANGE_SUPERMAG[1]])
            }

            TIME_RANGE_OMNI = np.searchsorted(
                a=OMNI["UNIX_TIME"],
                v=[T, T + T_SIZE],
            )

            OMNI_CHUNK = {
                "UNIX_TIME" : np.nanmean(OMNI["UNIX_TIME"][TIME_RANGE_OMNI[0] : TIME_RANGE_OMNI[1]]),
                "AVG_B" : np.nanmean(OMNI["AVG_B"][TIME_RANGE_OMNI[0] : TIME_RANGE_OMNI[1]]),
                "FLOW_SPEED" : np.nanmean(OMNI["FLOW_SPEED"][TIME_RANGE_OMNI[0] : TIME_RANGE_OMNI[1]]),
                "PROTON_DENSITY" : np.nanmean(OMNI["PROTON_DENSITY"][TIME_RANGE_OMNI[0] : TIME_RANGE_OMNI[1]]),
                "SYM_H" : np.nanmean(OMNI["SYM_H"][TIME_RANGE_OMNI[0] : TIME_RANGE_OMNI[1]])
            }

            if not (np.isfinite(SUPERMAG_CHUNK["SME"]) & np.isfinite(OMNI_CHUNK["AVG_B"]) & np.isfinite(OMNI_CHUNK["FLOW_SPEED"]) & np.isfinite(OMNI_CHUNK["PROTON_DENSITY"]) & np.isfinite(OMNI_CHUNK["SYM_H"])):
                continue

            CHUNK = (T,
                     POES_CHUNK,
                     RBSP_CHUNK,
                     SUPERMAG_CHUNK,
                     OMNI_CHUNK,
                     T_SIZE,
                     L_SIZE,
                     MLT_SIZE,
                     L_MIN,
                     L_MAX)  # TIME

            CHUNKS_TO_PROCESS.append(CHUNK)

        print(f"Finding CONJUNCTIONS for year : {_year}")

        RESULTS = pool.starmap(find_conjunctions, CHUNKS_TO_PROCESS, chunksize=100)

        pool.close()
        pool.join()

        CONJUNCTIONS = []
        AVG_POES_POINTS = []
        AVG_RBSP_POINTS = []
        for r in RESULTS:

            CONJUNCTIONS.extend(r[0])
            AVG_POES_POINTS.append(r[1])
            AVG_RBSP_POINTS.append(r[2])

        print(
            f"Finished processing data for year : {_year}, Number of conjunctions: {len(CONJUNCTIONS)}"
        )

        print(f"Median of median number of POES points per bin: {np.nanmedian(AVG_POES_POINTS)}")
        print(f"Median of median number of RBSP points per bin: {np.nanmedian(AVG_RBSP_POINTS)}")
        CONJUNCTIONS_TOTAL.extend(CONJUNCTIONS)

        print(f"Finished processing year: {_year}")

        print(f"Total number of conjunctions so far: {len(CONJUNCTIONS_TOTAL)}")

    CONJUNCTIONS_TO_BE_SAVED = np.vstack(CONJUNCTIONS_TOTAL)

    print(f"Conjunctions to be saved: {CONJUNCTIONS_TO_BE_SAVED.shape}")

    np.savez(
        file=os.path.join(output_folder, f"CONJUNCTIONS_{VERSION}_{FIELD_MODEL}.npz"),
        CONJUNCTIONS=CONJUNCTIONS_TO_BE_SAVED,
    )