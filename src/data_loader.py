import datetime
import glob
import os
import warnings

import cdflib
import h5py
import numpy as np
from dateutil import rrule
from netCDF4 import Dataset

import date_helper
import global_configuration


def get_file_names_between_start_and_end(
    start: datetime.datetime,
    end: datetime.datetime,
    file_glob: str = "",
    input_dir_structure: str = "",
    debug: bool = False,
    verbose: bool = False,
) -> list[str]:

    # Maybe this code should be adapted in the future to work for more cases, but currently it works for all needed instruments as of 08/29/2024

    if not file_glob:
        raise Exception(
            "Tried to load cdfs, but no file glob was given. I have no idea what the file names look like!"
        )

    if not input_dir_structure:
        raise Exception(
            "Tried to load cdfs but no input_dir was given. I have no idea where to look for the requested file names!"
        )

    if debug:
        print(f"Start date: {start}")
        print(f"End date: {end}")
        print(f"Unprocessed file_glob : {file_glob}")
        print(f"Unprocessed input_dir_structure : {input_dir_structure}")

    paths = []

    if ("{$DAY}" in file_glob) or ("{$DAY}" in input_dir_structure):

        for dt in rrule.rrule(freq=rrule.DAILY, dtstart=start, until=end):

            formatted_file_glob = global_configuration.replace_all_keys_in_string_with_values(
                file_glob,
                {
                    "{$YEAR}": dt.year,
                    "{$MONTH}": date_helper.month_str_from_int(dt.month),
                    "{$DAY}": date_helper.day_str_from_int(dt.day),
                },
            )

            input_dir = global_configuration.replace_all_keys_in_string_with_values(
                input_dir_structure,
                {
                    "{$YEAR}": dt.year,
                    "{$MONTH}": date_helper.month_str_from_int(dt.month),
                    "{$DAY}": date_helper.day_str_from_int(dt.day),
                },
            )
            if debug:
                print(formatted_file_glob, input_dir)

            list_of_file_names_or_empty = glob.glob(
                pathname=formatted_file_glob, root_dir=input_dir
            )

            if len(list_of_file_names_or_empty) > 0:

                sorted_list_of_file_names = sorted(list_of_file_names_or_empty)
                sorted_list_of_paths = [
                    os.path.join(os.path.abspath(input_dir), name)
                    for name in sorted_list_of_file_names
                ]

                # Grab the newest file... Some of the servers have more than one version of the same file for the same day. We need to deal with that somehow.
                paths.append(sorted_list_of_paths[-1])

            else:
                if debug and verbose:
                    warnings.warn(
                        f"No file on disk matches the following glob: {os.path.join(os.path.abspath(input_dir), formatted_file_glob)}"
                    )

    elif ("{$MONTH}" in file_glob) or ("{$MONTH}" in input_dir_structure):

        for dt in rrule.rrule(
            freq=rrule.MONTHLY,
            dtstart=datetime.datetime(year=start.year, month=start.month, day=1),
            until=end,
            bymonthday=(1),
        ):

            formatted_file_glob = global_configuration.replace_all_keys_in_string_with_values(
                file_glob,
                {
                    "{$YEAR}": dt.year,
                    "{$MONTH}": date_helper.month_str_from_int(dt.month),
                    "{$DAY}": date_helper.day_str_from_int(dt.day),
                },
            )

            input_dir = global_configuration.replace_all_keys_in_string_with_values(
                input_dir_structure,
                {
                    "{$YEAR}": dt.year,
                    "{$MONTH}": date_helper.month_str_from_int(dt.month),
                    "{$DAY}": date_helper.day_str_from_int(dt.day),
                },
            )
            if debug:
                print(formatted_file_glob, input_dir)

            list_of_file_names_or_empty = glob.glob(
                pathname=formatted_file_glob, root_dir=input_dir
            )

            if len(list_of_file_names_or_empty) > 0:

                sorted_list_of_file_names = sorted(list_of_file_names_or_empty)
                sorted_list_of_paths = [
                    os.path.join(os.path.abspath(input_dir), name)
                    for name in sorted_list_of_file_names
                ]
                paths.extend(sorted_list_of_paths)
            else:
                if debug and verbose:
                    warnings.warn(
                        f"No file on disk matches the following glob: {os.path.join(os.path.abspath(input_dir), formatted_file_glob)}"
                    )

    elif ("{$YEAR}" in file_glob) or ("{$YEAR}" in input_dir_structure):

        for dt in rrule.rrule(
            freq=rrule.YEARLY,
            dtstart=datetime.datetime(year=start.year, month=1, day=1),
            until=end,
            byyearday=(1),
        ):

            formatted_file_glob = global_configuration.replace_all_keys_in_string_with_values(
                file_glob,
                {
                    "{$YEAR}": dt.year,
                    "{$MONTH}": date_helper.month_str_from_int(dt.month),
                    "{$DAY}": date_helper.day_str_from_int(dt.day),
                },
            )

            input_dir = global_configuration.replace_all_keys_in_string_with_values(
                input_dir_structure,
                {
                    "{$YEAR}": dt.year,
                    "{$MONTH}": date_helper.month_str_from_int(dt.month),
                    "{$DAY}": date_helper.day_str_from_int(dt.day),
                },
            )
            if debug:
                print(formatted_file_glob, input_dir)

            list_of_file_names_or_empty = glob.glob(
                pathname=formatted_file_glob, root_dir=input_dir
            )

            if len(list_of_file_names_or_empty) > 0:

                sorted_list_of_file_names = sorted(list_of_file_names_or_empty)
                sorted_list_of_paths = [
                    os.path.join(os.path.abspath(input_dir), name)
                    for name in sorted_list_of_file_names
                ]
                paths.extend(sorted_list_of_paths)
            else:
                if debug and verbose:
                    warnings.warn(
                        f"No file on disk matches the following glob: {os.path.join(os.path.abspath(input_dir), formatted_file_glob)}"
                    )
    else:

        raise Exception(
            "Neither the file_glob nor the input_dir_structure had {$YEAR} or {$MONTH} or {$DAY} in it. So how can I tell what the ordering of the files should be?"
        )

    return paths


def load_data_files(paths: list[str], extension: str, variable_config: dict, debug: bool = False) -> dict:

    if debug:
        print(f"Paths: {paths}")
        print(f"Variable config: {variable_config}")

    if "time_variables" not in variable_config:
        variable_config["time_variables"] = {}
    if "time_dependent" not in variable_config:
        variable_config["time_dependent"] = {}
    if "file_dependent" not in variable_config:
        variable_config["file_dependent"] = {}

    # Concatenating all the arrays at once is much faster than copying every time?
    unconcatenated_arrays = {}

    for var_type in variable_config:
        for var in variable_config[var_type]:
            unconcatenated_arrays[var] = []

    timestamps_per_file = []

    if extension == ".cdf":

        for i, path in enumerate(paths):

            try:

                with cdflib.CDF(path=path) as cdf_file:

                    head_is_zero = (cdf_file.vdr_info(list(variable_config["time_variables"])[0]).head_vxr == 0)
                    tail_is_zero = (cdf_file.vdr_info(list(variable_config["time_variables"])[0]).last_vxr == 0)

                    if (head_is_zero and tail_is_zero):
                        continue

                    for j, var in enumerate(variable_config["time_variables"]):
                        t_array = np.squeeze(cdf_file.varget(variable=var))
                        unconcatenated_arrays[var].append(t_array)
                        if j == 0:
                            timestamps_per_file.append(len(t_array))

                    for var in variable_config["time_dependent"]:
                        unconcatenated_arrays[var].append(cdf_file.varget(variable=var))

                    for var in variable_config["file_dependent"]:
                        unconcatenated_arrays[var].append(
                            np.expand_dims(cdf_file.varget(variable=var), axis=0)
                        )
            except:
                with cdflib.CDF(path=path) as cdf_file:
                    print(cdf_file.vdr_info("Epoch"))
                print(path)
                break

    if extension == ".h5":

        for i, path in enumerate(paths):

            with h5py.File(path, "r") as h5_file:

                for j, var in enumerate(variable_config["time_variables"]):
                    t_array = np.squeeze(h5_file[var][...])
                    unconcatenated_arrays[var].append(t_array)
                    if j == 0:
                        timestamps_per_file.append(len(t_array))

                for var in variable_config["time_dependent"]:
                    unconcatenated_arrays[var].append(h5_file[var][...])

                for var in variable_config["file_dependent"]:
                    unconcatenated_arrays[var].append(np.expand_dims(h5_file[var][...], axis=0))

    if extension == ".nc":

        for i, path in enumerate(paths):

            with Dataset(path, "r") as nc_file:

                for j, var in enumerate(variable_config["time_variables"]):
                    t_array = np.squeeze(
                        np.ma.MaskedArray.filled(nc_file.variables[var][...], fill_value=np.nan)
                    )
                    unconcatenated_arrays[var].append(t_array)
                    if j == 0:
                        timestamps_per_file.append(len(t_array))

                for var in variable_config["time_dependent"]:
                    unconcatenated_arrays[var].append(
                        np.ma.MaskedArray.filled(nc_file.variables[var][...], fill_value=np.nan)
                    )

                for var in variable_config["file_dependent"]:
                    unconcatenated_arrays[var].append(
                        np.expand_dims(
                            np.ma.MaskedArray.filled(
                                nc_file.variables[var][...], fill_value=np.nan
                            ),
                            axis=0,
                        )
                    )

    refs = {}
    for var in variable_config["time_variables"]:
        refs[var] = np.concatenate(
            unconcatenated_arrays[var], axis=variable_config["time_variables"][var]
        )
    for var in variable_config["time_dependent"]:
        refs[var] = np.concatenate(
            unconcatenated_arrays[var], axis=variable_config["time_dependent"][var]
        )
    for var in variable_config["file_dependent"]:
        refs[var] = np.concatenate(unconcatenated_arrays[var], axis=0)

    if timestamps_per_file:
        refs["timestamps_per_file"] = timestamps_per_file

    return refs


def load_raw_data_from_config(
    id: list[str],
    start: datetime.datetime,
    end: datetime.datetime,
    satellite: str = "",
    config_path: str = "",
    root_data_dir: str = "",
    use_config_keys_in_subdir: bool = True,
    debug: bool = False,
    verbose: bool = False,
) -> dict:
    """This function loads at least all the data between start and end for the variables specified in the config.yaml file for this id.
    However, this function is not meant to be intelligent, essentially it doesn't know what type of data the time variable is..
    so it doesn't crop the data loaded to provide only data between start and end. Instead, the user of the function is expected to crop the data manually.
    """

    config, config_path = global_configuration.Config(config_path).load()

    id_config = config
    for level in id:
        id_config = id_config[level]

    if use_config_keys_in_subdir:

        if (root_data_dir):
            input_dir_structure = os.path.join(os.path.abspath(os.path.dirname(root_data_dir)), *id)
        elif os.environ.get("RESEARCH_RAW_DATA_DIR"):
            input_dir_structure = os.path.join(os.environ["RESEARCH_RAW_DATA_DIR"], *id)
        else:
            input_dir_structure = os.path.join(os.path.abspath(os.path.dirname(config_path)), *id)

    else:

        if (root_data_dir):
            input_dir_structure = os.path.abspath(os.path.dirname(root_data_dir))
        elif os.environ.get("RESEARCH_RAW_DATA_DIR"):
            input_dir_structure = os.environ["RESEARCH_RAW_DATA_DIR"]
        else:
            input_dir_structure = os.path.abspath(os.path.dirname(config_path))
    
    if "subdir" in id_config:
        input_dir_structure = os.path.join(input_dir_structure, *id_config["subdir"].split("/"))

    if "file_glob" not in id_config.keys():
        raise Exception(
            "Tried to load an ID with no file_glob set in the global config. I have no idea what the filename looks like or what type it is!"
        )

    file_name, file_extension = os.path.splitext(id_config["file_glob"])

    # More functionality can be added here before the loading routines if absolutely needed
    if "{$SATELLITE}" in file_name or "{$SATELLITE}" in input_dir_structure:
        if not satellite:
            raise Exception(
                "File name or input_dir for this ID requires a satellite but no satellite was specified!"
            )
        else:
            file_name = global_configuration.replace_all_keys_in_string_with_values(
                file_name, {"{$SATELLITE}": satellite}
            )
            input_dir_structure = global_configuration.replace_all_keys_in_string_with_values(
                input_dir_structure, {"{$SATELLITE}": satellite}
            )

    if "{$SATELLITE_UPPER}" in file_name or "{$SATELLITE_UPPER}" in input_dir_structure:
        if not satellite:
            raise Exception(
                "File name or input_dir for this ID requires a satellite but no satellite was specified!"
            )
        else:
            file_name = global_configuration.replace_all_keys_in_string_with_values(
                file_name, {"{$SATELLITE_UPPER}": satellite.upper()}
            )
            input_dir_structure = global_configuration.replace_all_keys_in_string_with_values(
                input_dir_structure, {"{$SATELLITE_UPPER}": satellite.upper()}
            )
    # -------------------------------------------------------------------------------

    paths_of_files_within_timeperiod = get_file_names_between_start_and_end(
        start=start,
        end=end,
        file_glob=(file_name + file_extension),
        input_dir_structure=input_dir_structure,
        debug=debug,
        verbose=verbose,
    )

    if len(paths_of_files_within_timeperiod) == 0:
        return {}

    return load_data_files(
        paths=paths_of_files_within_timeperiod,
        extension=file_extension,
        variable_config=id_config["variables"],
        debug=debug,
    )
