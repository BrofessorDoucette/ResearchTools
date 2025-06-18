import numpy as np
from typing import Iterable, Dict, Callable, Set, Union, Tuple
from numbers import Number


def get_dataset_stats(
    x: Iterable[Number],
    stats: Set[str],
    less_than: Union[Number, None] = None,
    between: Union[Tuple[Number, Number], None] = None,
    greater_than: Union[Number, None] = None
) -> Dict[str, float]:
    """
    Calculate various statistical measures for a given dataset, including subsets based on value thresholds.

    Args:
        x: Iterable of numbers (int or float)
        stats: Set of strings specifying which statistics to compute
        less_than: Upper bound for values to include (optional)
        between: Tuple of (lower, upper) bounds for values to include (optional)
        greater_than: Lower bound for values to include (optional)

    Returns:
        Dictionary mapping requested statistics to their computed values

    Raises:
        ValueError: If invalid stat is requested, input is invalid, or conflicting bounds are provided
    """
    # Convert input to numpy array and validate
    try:
        arr = np.array(x, dtype=float)
        if arr.size == 0:
            raise ValueError("Input array is empty")
    except (TypeError, ValueError):
        raise ValueError("Input must be an iterable of numbers")

    # Validate bounds parameters
    bounds_count = sum(1 for bound in [less_than, between, greater_than] if bound is not None)
    if bounds_count > 1:
        raise ValueError("Only one of less_than, between, or greater_than can be specified")
    
    # Apply bounds filtering
    if less_than is not None:
        arr = arr[arr < less_than]
    elif between is not None:
        if not isinstance(between, tuple) or len(between) != 2:
            raise ValueError("between must be a tuple of (lower, upper) bounds")
        lower, upper = between
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound")
        arr = arr[(arr >= lower) & (arr <= upper)]
    elif greater_than is not None:
        arr = arr[arr > greater_than]

    # If filtered array is empty, return empty dict for stats
    if arr.size == 0:
        return {stat: np.nan for stat in stats}

    # Define available statistics
    stat_functions: Dict[str, Callable[[np.ndarray], float]] = {
        "mean": lambda x: np.nanmean(x),
        "avg": lambda x: np.nanmean(x),  # Alias for mean
        "median": lambda x: np.nanmedian(x),
        "std": lambda x: np.nanstd(x),
        "deviation": lambda x: np.nanstd(x),  # Alias for std
        "var": lambda x: np.nanvar(x),
        "variance": lambda x: np.nanvar(x),  # Alias for var
        "min": lambda x: np.nanmin(x),
        "max": lambda x: np.nanmax(x),
        "sum": lambda x: np.nansum(x),
        "num_zeros": lambda x: np.sum(x == 0),
        "== 0": lambda x: np.sum(x == 0),  # Alias for num_zeros
        "num_nonzero": lambda x: np.sum(x != 0),
        "!= 0": lambda x: np.sum(x != 0),  # Alias for num_nonzero
        "% 0": lambda x: np.sum(x == 0) / len(x) if len(x) > 0 else 0.0,
        "percent_zeros": lambda x: np.sum(x == 0) / len(x) if len(x) > 0 else 0.0,  # Alias for % 0
        "% != 0": lambda x: np.sum(x != 0) / len(x) if len(x) > 0 else 0.0,
        "percent_nonzero": lambda x: np.sum(x != 0) / len(x) if len(x) > 0 else 0.0,  # Alias for % != 0
        "count": lambda x: np.sum(~np.isnan(x)),
        "num_nan": lambda x: np.sum(np.isnan(x)),
        "range": lambda x: np.nanmax(x) - np.nanmin(x),
        "q1": lambda x: np.nanpercentile(x, 25),
        "q3": lambda x: np.nanpercentile(x, 75),
        "iqr": lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25),
        "skew": lambda x: (
            np.nanmean((x - np.nanmean(x)) ** 3) / (np.nanstd(x) ** 3) if np.nanstd(x) != 0 else 0.0
        ),
        "kurtosis": lambda x: (
            np.nanmean((x - np.nanmean(x)) ** 4) / (np.nanstd(x) ** 4) if np.nanstd(x) != 0 else 0.0
        ),
    }

    # Validate requested stats
    invalid_stats = stats - set(stat_functions.keys())
    if invalid_stats:
        raise ValueError(f"Invalid statistics requested: {invalid_stats}")

    # Compute requested statistics
    return {stat: stat_functions[stat](arr) for stat in stats}


def print_dataset_stats(
    x: Iterable[Number],
    stats: Set[str],
    less_than: Union[Number, None] = None,
    between: Union[Tuple[Number, Number], None] = None,
    greater_than: Union[Number, None] = None
) -> None:
    """
    Print statistical measures for a dataset in a formatted manner, with optional value thresholds.

    Args:
        x: Iterable of numbers (int or float)
        stats: Set of strings specifying which statistics to compute
        less_than: Upper bound for values to include (optional)
        between: Tuple of (lower, upper) bounds for values to include (optional)
        greater_than: Lower bound for values to include (optional)
    """
    try:
        # Get the statistics
        stats_dict = get_dataset_stats(x, stats, less_than, between, greater_than)

        # Print header with bounds information
        print("\nDataset Statistics:")
        if less_than is not None:
            print(f"(Values less than {less_than})")
        elif between is not None:
            print(f"(Values between {between[0]} and {between[1]})")
        elif greater_than is not None:
            print(f"(Values greater than {greater_than})")
        print("-" * 50)

        # Define descriptive names for stats
        stat_names = {
            "mean": "Mean",
            "avg": "Average",
            "median": "Median",
            "std": "Standard Deviation",
            "deviation": "Standard Deviation",
            "variance": "Variance",
            "var": "Variance",
            "min": "Minimum",
            "max": "Maximum",
            "sum": "Sum",
            "num_zeros": "Number of Zeros",
            "== 0": "Number of Zeros",
            "num_nonzero": "Number of Non-Zeros",
            "!= 0": "Number of Non-Zeros",
            "% 0": "Percentage of Zeros",
            "percent_zeros": "Percentage of Zeros",
            "% != 0": "Percentage of Non-Zeros",
            "percent_nonzero": "Percentage of Non-Zeros",
            "count": "Count (non-NaN)",
            "num_nan": "Number of NaN Values",
            "range": "Range",
            "q1": "First Quartile (Q1)",
            "q3": "Third Quartile (Q3)",
            "iqr": "Interquartile Range",
            "skew": "Skewness",
            "kurtosis": "Kurtosis",
        }

        # Print each statistic
        for stat, value in stats_dict.items():
            # Format percentage for % 0, percent_zeros, % != 0, and percent_nonzero
            if stat in ["% 0", "percent_zeros", "% != 0", "percent_nonzero"]:
                print(f"{stat_names[stat]:<25}: {value * 100:.2f}%")
            else:
                print(f"{stat_names[stat]:<25}: {value:.4f}")

        print("-" * 50)

    except ValueError as e:
        print(f"Error: {e}")