import numpy as np
import data_loader
import similarity
import math
import kdtree

### THE K in our K-NN algorithm ###
K = 20


def _predict_rating(movie_id, user_id, coefficients, user_to_movies_matrix, movieid_to_idx=data_loader.movieid_to_idx,
                    userid_to_idx=data_loader.userid_to_idx):
    """
    For a given movie_id and user_id, gets the ratings of the K closest/most similar neighbouts and returns
    the average of their ratings. If no nearest neighbours have ratings for the movie, then average rating is returned.
    pa;j = va + Xni=1w(a; i)(v(i,j) - v(i, avg))
    :return:
    """
    movie_idx = int(movieid_to_idx[movie_id])
    user_idx = int(userid_to_idx[user_id])

    closest_neighbors_idxs = np.argsort(coefficients[user_idx, :])

    k_nearest_ratings = []
    sum_of_weights = 0.0
    for neighbor_idx in reversed(closest_neighbors_idxs):
        neighbor_rating = user_to_movies_matrix[neighbor_idx, movie_idx]
        neighbor_weight = coefficients[user_idx, neighbor_idx]
        if neighbor_rating != 0.0:
            k_nearest_ratings.append((neighbor_rating, neighbor_idx, neighbor_weight))
            sum_of_weights += neighbor_weight
        if len(k_nearest_ratings) == K:
            break

    if len(k_nearest_ratings) != 0:
        self_average = similarity._get_avg_rating(user_idx, user_to_movies_matrix)
        predicted_rating_change = 0.0
        for near_rating_tuple in k_nearest_ratings:
            neighbor_rating = near_rating_tuple[0]
            neighbor_idx = near_rating_tuple[1]
            neighbor_average = similarity._get_avg_rating(neighbor_idx, user_to_movies_matrix)
            neighbour_similarity = near_rating_tuple[2]

            predicted_rating_change += (neighbour_similarity * (neighbor_rating - neighbor_average))

        if sum_of_weights == 0.0:
            print('Got 0 sum of weights from user idx {}'.format(user_idx))
            predicted_rating = self_average
        else:
            predicted_rating = (self_average + (predicted_rating_change / sum_of_weights))
        return predicted_rating
    else:
        return 0.0


def test_predictions(coefficients, raw_testing_data, user_to_movies_matrix):
    """
    Iterates through the test set, and predicts the rating for a user based on the ratings of the "K"
    nearest or most similar users for the required movie. The RMSE and MAE are aggregated as the predictions
    are made.
    :param coefficients:
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

        predicted_rating = _predict_rating(movie_id, user_id, coefficients, user_to_movies_matrix)

        if predicted_rating is None:
            continue
        elif predicted_rating == 0.0:
            not_predicted_count += 1
            print('Could not predict rating for userID {} and movieID {} and test record {}'.format(user_id, movie_id, i))
        else:
            predicted_count += 1
            rmse_sum += math.pow(predicted_rating - actual_rating, 2)
            mae_sum += math.fabs(predicted_rating - actual_rating)
            print('Actual rating: {} Predicted rating: {} test record {}'.format(actual_rating, predicted_rating, i))

    print('FINAL RESULTS')
    print('Not predicted values: {}'.format(not_predicted_count))
    print('Predicted values: {}'.format(predicted_count))
    print('MAE: {}'.format(mae_sum / predicted_count))
    print('RMSE: {}'.format(rmse_sum / predicted_count))
