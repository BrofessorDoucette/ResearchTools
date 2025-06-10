import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
import sklearn

def calculate_values(combined_array, values, k=3, min_neighbors=2, max_dist=0.1):

    indices = combined_array[:k]
    distances = combined_array[k:]

    valid_neighbors = indices != len(values)
    valid_mask = (distances < max_dist) & valid_neighbors
    num_valid = np.sum(valid_mask)

    if num_valid < min_neighbors:
        return np.nan

    valid_indices = indices[valid_mask].astype(np.int32)
    valid_distances = distances[valid_mask]

    weights = 1 / (valid_distances + 1e-9)  # Avoid division by zero
    weights /= np.sum(weights)  # Normalize weights


    returned_values = np.full_like()
    
    valid_values = values[valid_indices]

    return np.nansum(valid_values * weights), np.nansum()


def knn_interpolate_polar(r, theta, values, r_query, theta_query, k=3, max_dist=0.1, min_neighbors=2):
    """
    Perform 2D k-nearest neighbors interpolation in polar coordinates (r, theta).

    Parameters:
    - r: 1D array of radial coordinates (non-circular).
    - theta: 1D array of angular coordinates (circular, in radians).
    - values: 1D array of values to interpolate.
    - r_query: 1D array of query radial coordinates.
    - theta_query: 1D array of query angular coordinates.
    - k: Number of nearest neighbors to consider (default: 3).
    - max_dist: Maximum distance in Cartesian coordinates for interpolation (default: 1.0).

    Returns:
    - interpolated_values: 1D array of interpolated values at query points.
    """
    # Ensure inputs are numpy arrays
    r = np.asarray(r)
    theta = np.asarray(theta)
    r_query = np.asarray(r_query)
    theta_query = np.asarray(theta_query)

    # Convert polar coordinates to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x_query = r_query * np.cos(theta_query)
    y_query = r_query * np.sin(theta_query)

    # Create KD-tree for efficient neighbor search
    points = np.column_stack((x, y))
    query_points = np.column_stack((x_query, y_query))
    tree = KDTree(points)

    # Find k-nearest neighbors and their distances
    distances, indices = tree.query(query_points, k=k, distance_upper_bound=max_dist)

    # Initialize output array
    combined_array = np.hstack([indices, distances])

    for 
    
    values = np.apply_along_axis(
        calculate_values, 1, combined_array, values, k, min_neighbors, max_dist
    )

    return values


def interpolate_around_points(r, theta, values, r_query, theta_query, k=3, max_dist=0.1):

    return knn_interpolate_polar(r, theta, values, r_query, theta_query, k=k, max_dist=max_dist)
