import data_loader
import similarity
import predict
import heapq
import numpy as np

K = 20


def _get_n_most_rated_movies(user_to_movies_matrix, n=21):
    n_most_rated_movies = []
    num_movies = user_to_movies_matrix.shape[1]

    for j in range(num_movies):
        num_ratings = np.where(user_to_movies_matrix[:, j] != 0.0)[0].shape[0]
        if len(n_most_rated_movies) < n:
            heapq.heappush(n_most_rated_movies, (num_ratings, j))
        elif num_ratings > n_most_rated_movies[0][0]:
            heapq.heappop(n_most_rated_movies)
            heapq.heappush(n_most_rated_movies, (num_ratings, j))

    assert len(n_most_rated_movies) == n

    n_most_rated_movies_idxs = []
    for movie in n_most_rated_movies:
        n_most_rated_movies_idxs.append(movie[1])
    return n_most_rated_movies_idxs


if __name__ == "__main__":
    raw_training_data = data_loader.load_data_from_saved_py()
    raw_testing_data = data_loader.load_testing_data_from_saved_npy()
    user_to_movies_matrix = data_loader.build_user_x_movie_matrix(raw_training_data)

    n_most_rated_movie_idxs = _get_n_most_rated_movies(user_to_movies_matrix)

    partial_user_to_movies_matrix = user_to_movies_matrix[:, n_most_rated_movie_idxs]

    pearson_coefficients = similarity.calculate_pearson_coefficient(partial_user_to_movies_matrix)

    predict.test_predictions(pearson_coefficients, raw_testing_data, partial_user_to_movies_matrix)