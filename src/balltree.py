import data_loader
from sklearn.neighbors import BallTree
import numpy as np
import math

import kdtree_main
import balltree_main
import similarity


def predict_using_balltree(raw_testing_data, absolute_deviation_matrix):
    """
    Iterates through the test set, and predicts the rating for a user based on the ratings of the "K"
    nearest or most similar users for the required movie. The RMSE and MAE are aggregated as the predictions
    are made.
    This is Lazy - A ball tree is constructed on demand and queried for the K nearest neighbours
    :param raw_testing_data:
    :param absolute_deviation_matrix:
    """
    not_predicted_count = 0.0
    predicted_count = 0.0

    mae_sum = 0.0
    rmse_sum = 0.0
    num_test_records = raw_testing_data.shape[0]
    for i in range(num_test_records):
        movie_id = raw_testing_data[i, 0]
        user_id = raw_testing_data[i, 1]
        actual_rating = raw_testing_data[i, 2]

        predicted_rating = _predict_rating_using_balltree(movie_id, user_id, absolute_deviation_matrix)
        if predicted_rating is None:
            not_predicted_count += 1
            print('Could not predict rating for userID {} and movieID {}'.format(user_id, movie_id))
        else:
            predicted_count += 1
            rmse_sum += math.pow(predicted_rating - actual_rating, 2)
            mae_sum += math.fabs(predicted_rating - actual_rating)
            print(
                'Actual rating: {} Predicted rating: {} Test Record numner: {}'.format(actual_rating, predicted_rating,
                                                                                       i))

    print('FINAL RESULTS')
    print('Not predicted values: {}'.format(not_predicted_count))
    print('Predicted values: {}'.format(predicted_count))
    print('MAE: {}'.format(mae_sum / predicted_count))
    print('RMSE: {}'.format(rmse_sum / predicted_count))


def _predict_rating_using_balltree(movie_id, user_id, user_to_movies_deviations,
                                   movieid_to_idx=data_loader.movieid_to_idx,
                                   userid_to_idx=data_loader.userid_to_idx):
    user_idx = userid_to_idx[user_id]
    movie_idx = movieid_to_idx[movie_id]

    # Build a ball tree for the user
    build_balltree_for_user(user_to_movies_deviations, user_id)
    knn_idxs = userid_to_knnidxs_map[user_id]

    neighbour_movie_deviations = user_to_movies_deviations[knn_idxs, movie_idx]

    if len(neighbour_movie_deviations) == 0:
        return None
    else:
        mean_deviation = np.mean(neighbour_movie_deviations[np.where(neighbour_movie_deviations[:] != -10.0)])
        avg_rating = similarity._get_avg_rating(user_idx, kdtree_main.user_to_movies_matrix_global)
        return (avg_rating + mean_deviation)


# Map of the user_id to its K nearest neighbours' indexes
userid_to_knnidxs_map = {}


def build_balltree_for_user(user_to_movies_matrix, user_id):
    if user_id in userid_to_knnidxs_map:
        return None

    user_idx = data_loader.userid_to_idx[user_id]

    # Find all movies current user has rated. We ONLY use the ratings the user under test has computed
    # If another user has not rated a movie, it would be farther away in the hypersphere
    user_rated_movies = user_to_movies_matrix[:, np.where(user_to_movies_matrix[user_idx, :] != -10.0)[0]]

    # Create a BallTree with the movies the user has rated.
    user_ball_tree = BallTree(user_rated_movies)

    # Get the indexes & distances of the K nearest neighbours
    k_nn_distances, k_nn_idxs = user_ball_tree.query([user_rated_movies[user_idx]], k=balltree_main.K + 1)

    # Cache the user's k nearest neighbors
    userid_to_knnidxs_map[user_id] = k_nn_idxs[0][1:]

    return None
