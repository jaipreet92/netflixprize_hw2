import data_loader
from scipy.spatial import KDTree
import numpy as np

import main

# Map of user_id to its KDTree
userid_to_kdtree_map = {}

# Map of the user_id to its K nearest neighbours' indexes
userid_to_knnidxs_map = {}


def build_kdtree_for_user(user_to_movies_matrix, user_id):
    if user_id in userid_to_kdtree_map and user_id in userid_to_knnidxs_map:
        return userid_to_kdtree_map[user_id]

    user_idx = data_loader.userid_to_idx[user_id]

    # Find all movies current user has rated.
    user_rated_movies = user_to_movies_matrix[:, np.where(user_to_movies_matrix[user_idx, :] != 0.0)[0]]

    # Create a KDTree with the movies the user has rated.
    user_kd_tree = KDTree(user_rated_movies)
    userid_to_kdtree_map[user_id] = user_kd_tree

    # Get the indexes of the K nearest neighbours
    k_nn_idxs = user_kd_tree.query(user_rated_movies[user_idx], k=main.K + 1)[1][1:]
    userid_to_knnidxs_map[user_id] = k_nn_idxs

    return user_kd_tree
