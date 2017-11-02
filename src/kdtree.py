import data_loader
from scipy.spatial import KDTree
import numpy as np
import math

import kdtree_main
import similarity

# Map of the user_id to its K nearest neighbours' indexes
userid_to_knnidxs_map = {}


def build_kdtree_for_user(user_to_movies_matrix, user_id):
    if user_id in userid_to_knnidxs_map:
        return None

    user_idx = data_loader.userid_to_idx[user_id]

    # Find all movies current user has rated.
    user_rated_movies = user_to_movies_matrix[:, np.where(user_to_movies_matrix[user_idx, :] != -10.0)[0]]

    # Create a KDTree with the movies the user has rated.
    user_kd_tree = KDTree(user_rated_movies)

    # Get the indexes of the K nearest neighbours
    k_nn_distances, k_nn_idxs = user_kd_tree.query(user_rated_movies[user_idx], k=kdtree_main.K + 1)

    # Cache the user's k nearest neighbors
    userid_to_knnidxs_map[user_id] = k_nn_idxs[1:]

    return None


def get_deviation_from_mean_matrix(user_to_movies_matrix):
    """
    Here we are transforming our raw user_to_movies matrix into one where instead of the raw rating value we calculate
    how the movie's rating deviates from the user's average rating. For example, if User A's average rating is 1.0 and
    she rates movie as 2.0, and User B's average rating is 4.0 and she rates the movie as 5.0, then they are considered
    more "similar" rather than just comparing the raw 2.0 and 4.0 ratings. Similarly, this implies that users that have
    rated movies on opposite sides of their average rating are less similar.

    We assign a value of -10.0 to movies not rated by the user, since the maximum deviation is +/- 5.0. This has the
    effect that the KD-Tree we build will consider it less similar to the ones that both users have rated.
    :param user_to_movies_matrix:
    :return:
    """
    num_users = user_to_movies_matrix.shape[0]
    num_movies = user_to_movies_matrix.shape[1]

    deviation_matrix = np.full((num_users, num_movies), -10.0)
    for i in range(num_users):
        user_avg_rating = similarity._get_avg_rating(i, user_to_movies_matrix)
        for j in range(num_movies):
            if user_to_movies_matrix[i, j] != 0.0:
                # Only compute deviation if the user has rated the movie.
                deviation_matrix[i, j] = user_to_movies_matrix[i, j] - user_avg_rating
    return deviation_matrix

def predict_using_kdtree(raw_testing_data, user_to_movies_matrix):
    """
    Iterates through the test set, and predicts the rating for a user based on the ratings of the "K"
    nearest or most similar users for the required movie. The RMSE and MAE are aggregated as the predictions
    are made.
    :param raw_testing_data:
    :param user_to_movies_matrix:
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

        predicted_rating = _predict_rating_using_kdtree(movie_id, user_id, user_to_movies_matrix)
        if predicted_rating is None:
            not_predicted_count += 1
            print('Could not predict rating for userID {} and movieID {}'.format(user_id, movie_id))
        else:
            predicted_count += 1
            rmse_sum += math.pow(predicted_rating - actual_rating, 2)
            mae_sum += math.fabs(predicted_rating - actual_rating)
            print('Actual rating: {} Predicted rating: {} Test Record numner: {}'.format(actual_rating, predicted_rating, i))

    print('FINAL RESULTS')
    print('Not predicted values: {}'.format(not_predicted_count))
    print('Predicted values: {}'.format(predicted_count))
    print('MAE SUM: {}'.format(mae_sum))
    print('RMSE SUM: {}'.format(rmse_sum))
    print('MAE: {}'.format(mae_sum / predicted_count))
    print('RMSE: {}'.format(rmse_sum / predicted_count))


def _predict_rating_using_kdtree(movie_id, user_id, user_to_movies_deviations, movieid_to_idx=data_loader.movieid_to_idx,
                                 userid_to_idx=data_loader.userid_to_idx):
    user_idx = userid_to_idx[user_id]
    movie_idx = movieid_to_idx[movie_id]

    build_kdtree_for_user(user_to_movies_deviations, user_id)
    knn_idxs = userid_to_knnidxs_map[user_id]

    neighbour_movie_deviations = user_to_movies_deviations[knn_idxs, movie_idx]

    if len(neighbour_movie_deviations) == 0:
        return None
    else:
        mean_deviation = np.mean(neighbour_movie_deviations[np.where(neighbour_movie_deviations[:]!=-10.0)])
        avg_rating = similarity._get_avg_rating(user_idx, kdtree_main.user_to_movies_matrix_global)
        if np.isnan(mean_deviation):
            return avg_rating
        else:
            return (avg_rating + mean_deviation)
