import numpy as np
import math
import time

user_idx_to_avg_rating = {}


def _get_avg_rating(user_idx, user_to_movies_matrix):
    """
    Utility method to get the avg ratings for a user. Movies for which user did not submit ratings
    are not considered in the total number. Caches values in user_idx_to_avg_rating
    :param user_idx:
    :param user_to_movies_matrix:
    :return:
    """
    if user_idx in user_idx_to_avg_rating:
        return user_idx_to_avg_rating[user_idx]

    user_non_zero_ratings = user_to_movies_matrix[user_idx,
                                                  np.where(user_to_movies_matrix[user_idx, :] != 0.0)]

    avg = np.average(user_non_zero_ratings)
    user_idx_to_avg_rating[user_idx] = avg
    return avg


def calculate_pearson_coefficient(user_to_movies_matrix):
    """
    Calculates the Pearson Coefficient as described in the 2.1.1 section. For movies
    where there is no recorded rating, they correlation between them does not contribute to
    the overall coefficient.
    In the representation of the user_user_to_movies_matrix, 0.0 indicates that is no rating for
    the user
    :param user_to_movies_matrix:
    """
    num_users, num_movies = user_to_movies_matrix.shape

    # Change this to be lower than the lowest possible value of the coeffient
    # Currently, this is initialized as 0.0
    coefficients = np.empty((num_users, num_users))

    for i in range(num_users):
        start_time = time.time()
        for j in range(num_users):
            if i == j or coefficients[j, i] != 0.0:
                continue

            similarity_i_j_top = 0.0
            similarity_i_j_bottom = 0.0

            for k in range(num_movies):
                rating_user_i = user_to_movies_matrix[i, k]
                rating_user_j = user_to_movies_matrix[j, k]

                if rating_user_i == 0.0 or rating_user_j == 0.0:
                    continue

                avg_rating_i = _get_avg_rating(i, user_to_movies_matrix)
                avg_rating_j = _get_avg_rating(j, user_to_movies_matrix)

                similarity_i_j_bottom, similarity_i_j_top = _get_similarity(avg_rating_i, avg_rating_j, rating_user_i,
                                                                            rating_user_j, similarity_i_j_bottom,
                                                                            similarity_i_j_top)

            if similarity_i_j_bottom == 0.0:
                #print('Denominator when calculating similarity was 0 for users {}x{}'.format(i, j))
                continue
            coefficients[i, j] = similarity_i_j_top / math.sqrt(similarity_i_j_bottom)
        print('Calculating similarity for {}x{} took {} seconds'.format(i, j, time.time() - start_time))
    return coefficients


def _get_similarity(avg_rating_i, avg_rating_j, rating_user_i, rating_user_j, similarity_i_j_bottom,
                    similarity_i_j_top):
    similarity_i_j_top += (rating_user_i - avg_rating_i) * (rating_user_j - avg_rating_j)
    similarity_i_j_bottom += math.pow(rating_user_i - avg_rating_i, 2) * math.pow(
        rating_user_j - avg_rating_j, 2)
    return similarity_i_j_bottom, similarity_i_j_top
