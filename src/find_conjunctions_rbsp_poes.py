import numpy as np
import multiprocessing as mp
import chorus_machine_learning_helper
import os


def find_conjunctions(POES, RBSP, SUPERMAG, OMNI, MAX_T_DIFF_SEC, MAX_L_DIFF, MAX_MLT_DIFF):

    CONJUNCTIONS = []

    NUMBER_OF_RECORDS = len(POES["UNIX_TIME"])
    for T in range(NUMBER_OF_RECORDS):

        UNIX_TIME = POES["UNIX_TIME"][T]
        LSTAR = POES["LSTAR"][T]
        MLT = POES["MLT"][T]
        FLUX_SPECTRUM = POES["BLC_Flux"][T, :]

        for RBSP_PROBE in RBSP:

            TIME_RANGE = np.searchsorted(
                a=RBSP_PROBE["UNIX_TIME"],
                v=[(UNIX_TIME - MAX_T_DIFF_SEC), (UNIX_TIME + MAX_T_DIFF_SEC)],
            )

            CANDIDATE_TIMES = []
            CANDIDATE_LSTAR = []
            CANDIDATE_DEL_MLT = []
            CANDIDATE_UPPER_BAND = []
            CANDIDATE_LOWER_BAND = []

            for POINT in range(TIME_RANGE[0], TIME_RANGE[1], 1):

                DEL_LSTAR = LSTAR - RBSP_PROBE["LSTAR"][POINT]
                DEL_MLT = np.min(
                    [
                        (
                            max(MLT, RBSP_PROBE["MLT"][POINT]) - min(MLT, RBSP_PROBE["MLT"][POINT])
                        ),
                        (
                            (24 - max(MLT, RBSP_PROBE["MLT"][POINT])) + (min(MLT, RBSP_PROBE["MLT"][POINT]) - 0)
                        ),
                    ]
                )

                if (DEL_LSTAR**2 < MAX_L_DIFF**2) and (DEL_MLT**2 < MAX_MLT_DIFF**2):

                    CANDIDATE_TIMES.append(RBSP_PROBE["UNIX_TIME"][POINT])
                    CANDIDATE_LSTAR.append(RBSP_PROBE["LSTAR"][POINT])
                    CANDIDATE_DEL_MLT.append(DEL_MLT)
                    CANDIDATE_UPPER_BAND.append(RBSP_PROBE["UPPER_BAND"][POINT])
                    CANDIDATE_LOWER_BAND.append(RBSP_PROBE["LOWER_BAND"][POINT])

            if len(CANDIDATE_TIMES) == 0:
                continue

            TIME_RANGE = np.searchsorted(
                a=SUPERMAG["UNIX_TIME"],
                v=[(UNIX_TIME - MAX_T_DIFF_SEC), (UNIX_TIME + MAX_T_DIFF_SEC)],
            )
            AVG_SME = np.nanmean(SUPERMAG["SME"][TIME_RANGE[0] : TIME_RANGE[1]])

            TIME_RANGE = np.searchsorted(
                a=OMNI["UNIX_TIME"],
                v=[(UNIX_TIME - MAX_T_DIFF_SEC), (UNIX_TIME + MAX_T_DIFF_SEC)],
            )
            AVG_AVG_B = np.nanmean(OMNI["AVG_B"][TIME_RANGE[0] : TIME_RANGE[1]])
            AVG_FLOW_SPEED = np.nanmean(OMNI["FLOW_SPEED"][TIME_RANGE[0] : TIME_RANGE[1]])
            AVG_PROTON_DENSITY = np.nanmean(
                OMNI["PROTON_DENSITY"][TIME_RANGE[0] : TIME_RANGE[1]]
            )
            AVG_SYM_H = np.nanmean(OMNI["SYM_H"][TIME_RANGE[0] : TIME_RANGE[1]])

            if (
                np.isfinite(AVG_SME) & np.isfinite(AVG_AVG_B) & np.isfinite(AVG_FLOW_SPEED) & np.isfinite(AVG_PROTON_DENSITY) & np.isfinite(AVG_SYM_H)
            ):

                CONJUNCTION = [
                    UNIX_TIME,
                    LSTAR,
                    MLT,
                    *FLUX_SPECTRUM,
                    np.nanmean(CANDIDATE_TIMES),  # TIME OF RBSP POINT CHOSEN
                    np.nanmean(CANDIDATE_LSTAR),  # LSTAR OF RBSP POINT CHOSEN
                    np.nanmean(CANDIDATE_DEL_MLT),  # DIFFERENCE IN MLT FOUND
                    np.nanmean(CANDIDATE_UPPER_BAND),  # UPPER BAND CHORUS OBSERVED
                    np.nanmean(CANDIDATE_LOWER_BAND),  # LOWER BAND CHORUS OBSERVED
                    AVG_SME,
                    AVG_AVG_B,
                    AVG_FLOW_SPEED,
                    AVG_PROTON_DENSITY,
                    AVG_SYM_H,
                ]

                CONJUNCTIONS.append(CONJUNCTION)

    return CONJUNCTIONS


if __name__ == "__main__":

    # Stage 2, clean then combine RBSP, OMNI, and POES Data and find conjunctions between RBSP and POES

    VERSION = "v1d"
    FIELD_MODEL = "T89"

    pdata_folder = os.path.abspath("./../processed_data/chorus_neural_network/")
    rbsp_chorus_folder = os.path.join(pdata_folder, "STAGE_1", "CHORUS")
    rbsp_lstar_folder = os.path.join(pdata_folder, "STAGE_1", "Lstar")
    mpe_folder = os.path.join(pdata_folder, "STAGE_0", "MPE_DATA_PREPROCESSED_WITH_LSTAR")
    output_folder = os.path.join(pdata_folder, "STAGE_2", VERSION)

    MAX_L_DIFF = 0.10
    MAX_MLT_DIFF = 0.75
    MAX_T_DIFF_SEC = 3

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

        RBSP = [RBSP_A, RBSP_B]
        print(f"RBSP Data loaded for year : {_year}")

        print(f"Began loading POES Data for year : {_year}")

        POES_REFS = np.load(
            file=os.path.join(mpe_folder, rf"MPE_PREPROCESSED_DATA_{FIELD_MODEL}_{_year}_interpolated.npz"),
            allow_pickle=True,
        )
        POES = POES_REFS["DATA"].flatten()[0]

        print(f"Finished loading POES data for year : {_year}")

        OMNI = chorus_machine_learning_helper.load_OMNI_year(_year)
        SUPERMAG = chorus_machine_learning_helper.load_SUPERMAG_SME_year(_year)

        # FINALLY FIND THE CONJUNCTIONS

        print(f"Finding CONJUNCTIONS for year : {_year}")
        CONJUNCTIONS_YEAR = []
        for SATID in POES:

            NUMBER_OF_RECORDS = len(POES[SATID]["UNIX_TIME"])

            print(f"Number of records: {NUMBER_OF_RECORDS} for POES SATELLITE: {SATID}")

            queued_work = []
            num_processes = mp.cpu_count() - 2
            pool = mp.Pool(processes=num_processes)

            CUT_SIZE = NUMBER_OF_RECORDS // num_processes

            for p in range(num_processes):

                if p < num_processes - 1:
                    queued_work.append((p * CUT_SIZE, (p + 1) * CUT_SIZE))
                else:
                    queued_work.append((p * CUT_SIZE, NUMBER_OF_RECORDS))

            CHUNKS_TO_PROCESS = []
            for work in queued_work:

                if work[1] >= NUMBER_OF_RECORDS:
                    CHUNK_START_T = POES[SATID]["UNIX_TIME"][work[0]] - MAX_T_DIFF_SEC
                    CHUNK_END_T = POES[SATID]["UNIX_TIME"][-1] + MAX_T_DIFF_SEC
                else:
                    CHUNK_START_T = POES[SATID]["UNIX_TIME"][work[0]] - MAX_T_DIFF_SEC
                    CHUNK_END_T = POES[SATID]["UNIX_TIME"][work[1]] + MAX_T_DIFF_SEC

                POES_CHUNK = {
                    "UNIX_TIME" : POES[SATID]["UNIX_TIME"][work[0] : work[1]],
                    "LSTAR" : POES[SATID]["LSTAR"][work[0] : work[1]],
                    "MLT" : POES[SATID]["MLT"][work[0] : work[1]],
                    "BLC_Flux" : POES[SATID]["BLC_Flux"][work[0] : work[1], :]
                }

                TIME_RANGE_A = np.searchsorted(
                    a=RBSP[0]["UNIX_TIME"],
                    v=[CHUNK_START_T, CHUNK_END_T],
                )

                RBSP_A_CHUNK = {
                    "UNIX_TIME" : RBSP[0]["UNIX_TIME"][TIME_RANGE_A[0] : TIME_RANGE_A[1]],
                    "MLT" : RBSP[0]["MLT"][TIME_RANGE_A[0] : TIME_RANGE_A[1]],
                    "L" : RBSP[0]["L"][TIME_RANGE_A[0] : TIME_RANGE_A[1]],
                    "LSTAR" : RBSP[0]["LSTAR"][TIME_RANGE_A[0] : TIME_RANGE_A[1]],
                    "LOWER_BAND" : RBSP[0]["LOWER_BAND"][TIME_RANGE_A[0] : TIME_RANGE_A[1]],
                    "UPPER_BAND" : RBSP[0]["UPPER_BAND"][TIME_RANGE_A[0] : TIME_RANGE_A[1]]
                }

                TIME_RANGE_B = np.searchsorted(
                    a=RBSP[1]["UNIX_TIME"],
                    v=[CHUNK_START_T, CHUNK_END_T],
                )

                RBSP_B_CHUNK = {
                    "UNIX_TIME" : RBSP[1]["UNIX_TIME"][TIME_RANGE_B[0] : TIME_RANGE_B[1]],
                    "MLT" : RBSP[1]["MLT"][TIME_RANGE_B[0] : TIME_RANGE_B[1]],
                    "L" : RBSP[1]["L"][TIME_RANGE_B[0] : TIME_RANGE_B[1]],
                    "LSTAR" : RBSP[1]["LSTAR"][TIME_RANGE_B[0] : TIME_RANGE_B[1]],
                    "LOWER_BAND" : RBSP[1]["LOWER_BAND"][TIME_RANGE_B[0] : TIME_RANGE_B[1]],
                    "UPPER_BAND" : RBSP[1]["UPPER_BAND"][TIME_RANGE_B[0] : TIME_RANGE_B[1]]
                }

                RBSP_CHUNK = [RBSP_A_CHUNK, RBSP_B_CHUNK]

                TIME_RANGE_SUPERMAG = np.searchsorted(
                    a=SUPERMAG["UNIX_TIME"],
                    v=[CHUNK_START_T, CHUNK_END_T],
                )

                SUPERMAG_CHUNK = {
                    "UNIX_TIME" : SUPERMAG["UNIX_TIME"][TIME_RANGE_SUPERMAG[0] : TIME_RANGE_SUPERMAG[1]],
                    "SME" : SUPERMAG["SME"][TIME_RANGE_SUPERMAG[0] : TIME_RANGE_SUPERMAG[1]]
                }

                TIME_RANGE_OMNI = np.searchsorted(
                    a=OMNI["UNIX_TIME"],
                    v=[CHUNK_START_T, CHUNK_END_T],
                )

                OMNI_CHUNK = {
                    "UNIX_TIME" : OMNI["UNIX_TIME"][TIME_RANGE_OMNI[0] : TIME_RANGE_OMNI[1]],
                    "AVG_B" : OMNI["AVG_B"][TIME_RANGE_OMNI[0] : TIME_RANGE_OMNI[1]],
                    "FLOW_SPEED" : OMNI["FLOW_SPEED"][TIME_RANGE_OMNI[0] : TIME_RANGE_OMNI[1]],
                    "PROTON_DENSITY" : OMNI["PROTON_DENSITY"][TIME_RANGE_OMNI[0] : TIME_RANGE_OMNI[1]],
                    "SYM_H" : OMNI["SYM_H"][TIME_RANGE_OMNI[0] : TIME_RANGE_OMNI[1]]
                }

                CHUNK = (POES_CHUNK,
                         RBSP_CHUNK,
                         SUPERMAG_CHUNK,
                         OMNI_CHUNK,
                         MAX_T_DIFF_SEC,
                         MAX_L_DIFF,
                         MAX_MLT_DIFF)  # TIME

                CHUNKS_TO_PROCESS.append(CHUNK)

            RESULTS = pool.starmap(find_conjunctions, CHUNKS_TO_PROCESS)

            pool.close()
            pool.join()

            CONJUNCTIONS = []

            for r in RESULTS:

                CONJUNCTIONS.extend(r)

            print(
                f"Finished processing data for: {SATID}, Number of conjunctions: {len(CONJUNCTIONS)}"
            )

            CONJUNCTIONS_YEAR.extend(CONJUNCTIONS)

        CONJUNCTIONS_TOTAL.extend(CONJUNCTIONS_YEAR)

        print(f"Finished processing year: {_year}")

        print(f"Total number of conjunctions so far: {len(CONJUNCTIONS_TOTAL)}")

    CONJUNCTIONS_TO_BE_SAVED = np.vstack(CONJUNCTIONS_TOTAL)

    print(f"Conjunctions to be saved: {CONJUNCTIONS_TO_BE_SAVED.shape}")

    np.savez(
        file=os.path.join(output_folder, f"CONJUNCTIONS_{VERSION}_{FIELD_MODEL}.npz"),
        CONJUNCTIONS=CONJUNCTIONS_TO_BE_SAVED,
    )