import datetime
import multiprocessing as mp
import os
import time
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import chorus_machine_learning_helper


def find_conjunctions(CHUNK_TIME, POES, RBSP, SME_CHUNK, HP, T_SIZE, L_SIZE, MLT_SIZE, MLAT_SIZE, L_MIN, L_MAX, MLAT_MIN, MLAT_MAX):

    CONJUNCTIONS = []

    L_EDGES = np.arange(L_MIN, L_MAX, step=L_SIZE)
    MLT_EDGES = np.arange(0, 24, step=MLT_SIZE)
    MLAT_EDGES = np.arange(MLAT_MIN, MLAT_MAX, step=MLAT_SIZE)

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

    CUM_CHORUS_RBSP = np.zeros(shape=(L_EDGES.shape[0], MLT_EDGES.shape[0], MLAT_EDGES.shape[0]))
    CUM_L_RBSP = np.zeros_like(CUM_CHORUS_RBSP)
    CUM_MLT_RBSP = np.zeros_like(CUM_CHORUS_RBSP)
    CUM_MLAT_RBSP = np.zeros_like(CUM_CHORUS_RBSP)
    NUM_IN_EACH_RBSP_BIN = np.zeros_like(CUM_CHORUS_RBSP)

    for SAT in POES:

        L_mapping = np.digitize(SAT["L"], L_EDGES, right=False) - 1
        MLT_mapping = np.digitize(SAT["MLT"], MLT_EDGES, right=False) - 1

        for T in range(len(SAT["UNIX_TIME"])):

            x_bin = L_mapping[T]
            y_bin = MLT_mapping[T]

            CUM_FLUX_0[x_bin, y_bin] += SAT["BLC_Flux"][T, 0]
            CUM_FLUX_1[x_bin, y_bin] += SAT["BLC_Flux"][T, 1]
            CUM_FLUX_2[x_bin, y_bin] += SAT["BLC_Flux"][T, 2]
            CUM_FLUX_3[x_bin, y_bin] += SAT["BLC_Flux"][T, 3]
            CUM_FLUX_4[x_bin, y_bin] += SAT["BLC_Flux"][T, 4]
            CUM_FLUX_5[x_bin, y_bin] += SAT["BLC_Flux"][T, 5]
            CUM_FLUX_6[x_bin, y_bin] += SAT["BLC_Flux"][T, 6]
            CUM_FLUX_7[x_bin, y_bin] += SAT["BLC_Flux"][T, 7]
            CUM_L_POES[x_bin, y_bin] += SAT["L"][T]
            CUM_MLT_POES[x_bin, y_bin] += SAT["MLT"][T]
            NUM_IN_EACH_POES_BIN[x_bin, y_bin] += 1

    for PROBE in RBSP:

        L_mapping = np.digitize(PROBE["L"], L_EDGES, right=False) - 1
        MLT_mapping = np.digitize(PROBE["MLT"], MLT_EDGES, right=False) - 1
        MLAT_mapping = np.digitize(PROBE["MLAT"], MLAT_EDGES, right=False) - 1

        for T in range(len(PROBE["UNIX_TIME"])):

            x_bin = L_mapping[T]
            y_bin = MLT_mapping[T]
            z_bin = MLAT_mapping[T]

            CUM_CHORUS_RBSP[x_bin, y_bin, z_bin] += PROBE["CHORUS"][T]
            CUM_L_RBSP[x_bin, y_bin, z_bin] += PROBE["L"][T]
            CUM_MLT_RBSP[x_bin, y_bin, z_bin] += PROBE["MLT"][T]
            CUM_MLAT_RBSP[x_bin, y_bin, z_bin] += PROBE["MLAT"][T]
            NUM_IN_EACH_RBSP_BIN[x_bin, y_bin, z_bin] += 1

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

    AVG_CHORUS = CUM_CHORUS_RBSP / NUM_IN_EACH_RBSP_BIN
    AVG_L_RBSP = CUM_L_RBSP / NUM_IN_EACH_RBSP_BIN
    AVG_MLT_RBSP = CUM_MLT_RBSP / NUM_IN_EACH_RBSP_BIN
    AVG_MLAT_RBSP = CUM_MLAT_RBSP / NUM_IN_EACH_RBSP_BIN

    if (np.nanmax(NUM_IN_EACH_POES_BIN) == 0) or (np.nanmax(NUM_IN_EACH_RBSP_BIN) == 0):
        return [[], [], np.nan, np.nan]

    SME_CHUNKS = []

    for x_bin in range(L_EDGES.shape[0]):

        for y_bin in range(MLT_EDGES.shape[0]):

            for z_bin in range(MLAT_EDGES.shape[0]):

                if ((0 < NUM_IN_EACH_POES_BIN[x_bin, y_bin]) and (0 < NUM_IN_EACH_RBSP_BIN[x_bin, y_bin, z_bin])):

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
                        AVG_L_RBSP[x_bin, y_bin, z_bin],  # LSTAR OF RBSP POINT CHOSEN
                        AVG_MLT_RBSP[x_bin, y_bin, z_bin],  # DIFFERENCE IN MLT FOUND
                        AVG_MLAT_RBSP[x_bin, y_bin, z_bin],
                        AVG_CHORUS[x_bin, y_bin, z_bin],  # CHORUS OBSERVED
                        HP
                    ]

                    CONJUNCTIONS.append(CONJUNCTION)
                    SME_CHUNKS.append(SME_CHUNK)

    return [CONJUNCTIONS, SME_CHUNKS, np.nanmedian(NUM_IN_EACH_POES_BIN[NUM_IN_EACH_POES_BIN > 0]), np.nanmedian(NUM_IN_EACH_RBSP_BIN[NUM_IN_EACH_RBSP_BIN > 0])]


if __name__ == "__main__":

    # Stage 2, clean then combine RBSP, OMNI, and POES Data and find conjunctions between RBSP and POES

    VERSION = "v1"
    MODEL_TYPE = "LOWER_BAND"

    pdata_folder = os.path.abspath("./../processed_data/chorus_neural_network/")
    rbsp_chorus_folder = os.path.join(pdata_folder, "observed_chorus")
    output_folder = os.path.join(pdata_folder, "models", VERSION)

    T_SIZE = 1800
    L_SIZE = 0.05
    MLT_SIZE = 2.0
    MLAT_SIZE = 0.5
    L_MIN = 2.0
    L_MAX = 9
    MLAT_MIN = -25
    MLAT_MAX = 25

    CONJUNCTIONS_TOTAL = []
    SME_CHUNKS_TOTAL = []

    HP_UNIX_TIME, HP = chorus_machine_learning_helper.load_hp30(os.path.join(pdata_folder, "Hp30_Ap30_1996_2025.txt"))

    for _year in range(2012, 2020, 1):

        print(f"Began processing year : {_year}")

        RBSP = []

        for SATID in ["A", "B"]:

            # LOAD THE OBSERVED CHORUS
            print(f"Began loading RBSP Data for year: {_year}")
            refs = np.load(
                file=os.path.join(rbsp_chorus_folder, rf"observed_chorus_{_year}_{SATID}_{MODEL_TYPE}.npz"),
                allow_pickle=True,
            )
            PROBE = {}
            PROBE["UNIX_TIME"] = refs["UNIX_TIME"]
            PROBE["MLT"] = refs["MLT"]
            PROBE["MLAT"] = refs["MLAT"]
            PROBE["L"] = refs["L"]
            PROBE["CHORUS"] = refs["CHORUS"]

            refs.close()

            print(f"\nRBSP-{SATID} SHAPES BEFORE PREPROCESSING:")
            print(PROBE["UNIX_TIME"].shape)
            print(PROBE["MLT"].shape)
            print(PROBE["L"].shape)
            print(PROBE["MLAT"].shape)
            print(PROBE["CHORUS"].shape)

            order = np.argsort(PROBE["UNIX_TIME"])
            PROBE["UNIX_TIME"] = PROBE["UNIX_TIME"][order]
            PROBE["MLT"] = PROBE["MLT"][order]
            PROBE["L"] = PROBE["L"][order]
            PROBE["MLAT"] = PROBE["MLAT"][order]
            PROBE["CHORUS"] = PROBE["CHORUS"][order]

            between_bins = (L_MIN <= PROBE["L"]) & (PROBE["L"] < L_MAX) & (0 <= PROBE["MLT"]) & (PROBE["MLT"] < 24) & (MLAT_MIN <= PROBE["MLAT"]) & (PROBE["MLAT"] < MLAT_MAX)
            t_Lpp, Lpp = chorus_machine_learning_helper.load_plasmapause_filter(year=_year)

            Lpp_i = np.interp(x=PROBE["UNIX_TIME"], xp=t_Lpp, fp=Lpp, left=None, right=None)

            outside_plasmasphere = PROBE["L"] > Lpp_i

            PROBE["UNIX_TIME"] = PROBE["UNIX_TIME"][between_bins & outside_plasmasphere]
            PROBE["MLT"] = PROBE["MLT"][between_bins & outside_plasmasphere]
            PROBE["L"] = PROBE["L"][between_bins & outside_plasmasphere]
            PROBE["MLAT"] = PROBE["MLAT"][between_bins & outside_plasmasphere]
            PROBE["CHORUS"] = PROBE["CHORUS"][between_bins & outside_plasmasphere]

            print(f"\nRBSP-{SATID} SHAPES AFTER REMOVING POINTS OUTSIDE BINS and INSIDE PLASMASPHERE:")
            print(PROBE["UNIX_TIME"].shape)
            print(PROBE["MLT"].shape)
            print(PROBE["L"].shape)
            print(PROBE["MLAT"].shape)
            print(PROBE["CHORUS"].shape)

            RBSP.append(PROBE)

        print(f"RBSP Data loaded for year : {_year}")

        print(f"Began loading POES Data for year : {_year}")

        POES_UNFILTERED = chorus_machine_learning_helper.load_MPE_year(year=_year)

        POES = []
        for SAT in POES_UNFILTERED:

            between_bins_POES = (L_MIN <= SAT["L"]) & (SAT["L"] < L_MAX) & (0 <= SAT["MLT"]) & (SAT["MLT"] < 24)

            t_Lpp, Lpp = chorus_machine_learning_helper.load_plasmapause_filter(year=_year)

            Lpp_i = np.interp(x=SAT["UNIX_TIME"], xp=t_Lpp, fp=Lpp, left=np.nan, right=np.nan)
            outside_plasmasphere = SAT["L"] > Lpp_i

            POES.append({
                "UNIX_TIME" : SAT["UNIX_TIME"][between_bins_POES & outside_plasmasphere],
                "L" : SAT["L"][between_bins_POES & outside_plasmasphere],
                "MLT" : SAT["MLT"][between_bins_POES & outside_plasmasphere],
                "BLC_Flux" : SAT["BLC_Flux"][between_bins_POES & outside_plasmasphere, :]
            })

        print(f"Finished loading POES data for year : {_year}")

        SUPERMAG = chorus_machine_learning_helper.load_SUPERMAG_SME_year(_year)
        # FINALLY FIND THE CONJUNCTIONS

        start_of_year = datetime.datetime(year=_year, month=1, day=1, tzinfo=datetime.UTC).timestamp()
        end_of_year = datetime.datetime(year=_year + 1, month=1, day=1, tzinfo=datetime.UTC).timestamp()

        print("Separating into chunks!")
        CONJUNCTIONS_YEAR = []

        num_processes = mp.cpu_count() - 2
        pool = mp.Pool(processes=num_processes)

        chunk_times = np.arange(start_of_year, end_of_year, step=T_SIZE)
        print(len(chunk_times))
        CHUNKS_TO_PROCESS = []

        for T in chunk_times:

            POES_CHUNK = []

            t0 = time.perf_counter()

            for SAT in POES:

                TIME_RANGE_POES = np.searchsorted(
                    a=SAT["UNIX_TIME"],
                    v=[T - T_SIZE, T],
                )

                POES_CHUNK.append({
                    "UNIX_TIME" : SAT["UNIX_TIME"][TIME_RANGE_POES[0] : TIME_RANGE_POES[1]],
                    "L" : SAT["L"][TIME_RANGE_POES[0] : TIME_RANGE_POES[1]],
                    "MLT" : SAT["MLT"][TIME_RANGE_POES[0] : TIME_RANGE_POES[1]],
                    "BLC_Flux" : SAT["BLC_Flux"][TIME_RANGE_POES[0] : TIME_RANGE_POES[1], :]
                })

            RBSP_CHUNK = []

            for RBSP_PROBE in RBSP:

                TIME_RANGE_RBSP = np.searchsorted(
                    a=RBSP_PROBE["UNIX_TIME"],
                    v=[T - T_SIZE, T],
                )

                RBSP_PROBE_CHUNK = {
                    "UNIX_TIME" : RBSP_PROBE["UNIX_TIME"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]],
                    "MLT" : RBSP_PROBE["MLT"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]],
                    "L" : RBSP_PROBE["L"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]],
                    "MLAT" : RBSP_PROBE["MLAT"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]],
                    "CHORUS" : RBSP_PROBE["CHORUS"][TIME_RANGE_RBSP[0] : TIME_RANGE_RBSP[1]],
                }

                RBSP_CHUNK.append(RBSP_PROBE_CHUNK)

            TIME_RANGE_SUPERMAG = np.searchsorted(
                a=SUPERMAG["UNIX_TIME"],
                v=[T],
            )

            HP_TIME = np.searchsorted(a=HP_UNIX_TIME, v=[T])

            CHUNK = (T,
                     POES_CHUNK,
                     RBSP_CHUNK,
                     SUPERMAG["SME"][TIME_RANGE_SUPERMAG[-1] - 240 : TIME_RANGE_SUPERMAG[-1]],
                     HP[HP_TIME[-1]],
                     T_SIZE,
                     L_SIZE,
                     MLT_SIZE,
                     MLAT_SIZE,
                     L_MIN,
                     L_MAX,
                     MLAT_MIN,
                     MLAT_MAX)  # TIME

            CHUNKS_TO_PROCESS.append(CHUNK)

        print(f"Finding CONJUNCTIONS for year : {_year}")

        RESULTS = pool.starmap(find_conjunctions, CHUNKS_TO_PROCESS, chunksize=100)

        pool.close()
        pool.join()

        CONJUNCTIONS = []
        SME_CHUNKS = []
        AVG_POES_POINTS = []
        AVG_RBSP_POINTS = []
        for r in RESULTS:

            CONJUNCTIONS.extend(r[0])
            SME_CHUNKS.extend(r[1])
            AVG_POES_POINTS.append(r[2])
            AVG_RBSP_POINTS.append(r[3])

        print(
            f"Finished processing data for year : {_year}, Number of conjunctions: {len(CONJUNCTIONS)}"
        )

        print(f"Median of median number of POES points per bin: {np.nanmedian(AVG_POES_POINTS)}")
        print(f"Median of median number of RBSP points per bin: {np.nanmedian(AVG_RBSP_POINTS)}")
        CONJUNCTIONS_TOTAL.extend(CONJUNCTIONS)
        SME_CHUNKS_TOTAL.extend(SME_CHUNKS)
        print(f"Finished processing year: {_year}")

        print(f"Total number of conjunctions so far: {len(CONJUNCTIONS_TOTAL)}")

    CONJUNCTIONS_TO_BE_SAVED = np.vstack(CONJUNCTIONS_TOTAL)
    SME_CHUNKS_TO_BE_SAVED = np.vstack(SME_CHUNKS)

    print(f"Conjunctions to be saved: {CONJUNCTIONS_TO_BE_SAVED.shape}")

    np.savez(
        file=os.path.join(output_folder, f"CONJUNCTIONS_{VERSION}_{MODEL_TYPE}.npz"),
        CONJUNCTIONS=CONJUNCTIONS_TO_BE_SAVED,
        SME_CHUNKS=SME_CHUNKS_TO_BE_SAVED
    )