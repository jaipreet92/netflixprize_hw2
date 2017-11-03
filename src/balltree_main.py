import balltree
import data_loader
import kdtree as deviation_util
import mostrated_main as mostrated_util

K = 100

user_to_movies_matrix_global = None

if __name__ == "__main__":
    # Load training and testing data
    raw_training_data = data_loader.load_data_from_saved_py()
    raw_testing_data = data_loader.load_testing_data_from_saved_npy()

    # Form a user*movie_ratings matrix
    user_to_movies_matrix = data_loader.build_user_x_movie_matrix(raw_training_data)
    user_to_movies_matrix_global = user_to_movies_matrix

    # Create an absolute deviation matrix
    absolute_deviation_matrix = deviation_util.get_deviation_from_mean_matrix(user_to_movies_matrix)

    # Sample the N most commonly rated movies
    mostrated_movies = mostrated_util._get_n_most_rated_movies(user_to_movies_matrix, n=100)
    users_to_most_rated_movies_matrix = user_to_movies_matrix[:, mostrated_movies]
    most_rated_movies_deviation_matrix = absolute_deviation_matrix[:, mostrated_movies]

    balltree.predict_using_balltree(raw_testing_data=raw_testing_data,
                                    absolute_deviation_matrix=most_rated_movies_deviation_matrix,
                                    full_deviation_matrix=absolute_deviation_matrix)
