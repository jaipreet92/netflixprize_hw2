import numpy as np
import data_loader
import math

### THE K in our K-NN algorithm ###
K = 5


def _predict_rating(movie_id, user_id, coefficients, user_to_ratings_matrix, movieid_to_idx=data_loader.movieid_to_idx,
                    userid_to_idx=data_loader.userid_to_idx):
    """
    For a given movie_id and user_id, gets the ratings of the K closest/most similar neighbouts and returns
    the average of their ratings. If no nearest neighbours have ratings for the movie, then 0.0 is returned.
    :param movie_id:
    :param user_id:
    :param coefficients:
    :param user_to_ratings_matrix:
    :param movieid_to_idx:
    :param userid_to_idx:
    :return:
    """
    movie_idx = movieid_to_idx[movie_id]
    user_idx = userid_to_idx[user_id]

    if user_idx >= 2000 or movie_idx >= 100:
        print('We removed this user id or movie id from training set')
        return None

    closest_neighbours = np.sort(coefficients[user_idx, :])

    k_nearest_ratings = []
    for neighbor_idx, neighbor in enumerate(closest_neighbours):
        rating = user_to_ratings_matrix[neighbor_idx, movie_idx]
        if rating != 0.0:
            k_nearest_ratings.append(rating)
        if len(k_nearest_ratings) == K:
            break

    if len(k_nearest_ratings) != 0:
        return np.mean(k_nearest_ratings)
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
    num_test_records = raw_testing_data.shape[0]
    for i in range(num_test_records):
        movie_id = raw_testing_data[i, 0]
        user_id = raw_testing_data[i, 1]
        actual_rating = raw_testing_data[i, 2]

        predicted_rating = _predict_rating(movie_id, user_id, coefficients, user_to_movies_matrix)

        not_predicted_count = 0.0
        predicted_count = 0.0

        mae_sum = 0.0
        rmse_sum = 0.0
        if predicted_rating is None:
            continue
        elif predicted_rating == 0.0:
            not_predicted_count += 1
            print('Could not predict rating for userID {} and movieID {}'.format(user_id, movie_id))
        else:
            predicted_count += 1
            rmse_sum += math.pow(predicted_rating - actual_rating, 2)
            mae_sum += math.fabs(predicted_rating - actual_rating)
            print('Actual rating: {} Predicted rating: {}'.format(actual_rating, predicted_rating))

    print('FINAL RESULTS')
    print('Not predicted values: {}'.format(not_predicted_count))
    print('Predicted values: {}'.format(predicted_count))
    print('MAE: {}'.format(mae_sum / predicted_count))
    print('RMSE: {}'.format(rmse_sum / predicted_count))
