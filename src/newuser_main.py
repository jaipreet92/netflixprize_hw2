import balltree
import data_loader
import kdtree as deviation_util
import mostrated_main as mostrated_util

K = 100

user_to_movies_matrix_global = None

if __name__ == "__main__":
    # Load training data with self/new user ratings
    raw_training_data = data_loader.load_newuser_training_data_from_text()

    # Form a user*movie_ratings matrix
    user_to_movies_matrix = data_loader.build_user_x_movie_matrix(raw_training_data)
    user_to_movies_matrix_global = user_to_movies_matrix

    # Create an absolute deviation matrix
    absolute_deviation_matrix = deviation_util.get_deviation_from_mean_matrix(user_to_movies_matrix)

    # Sample the N most commonly rated movies for our Ball tree
    mostrated_movies = mostrated_util._get_n_most_rated_movies(user_to_movies_matrix, n=100)
    users_to_most_rated_movies_matrix = user_to_movies_matrix[:, mostrated_movies]
    most_rated_movies_deviation_matrix = absolute_deviation_matrix[:, mostrated_movies]

    # Do predictions for all movies that are not rated
    balltree.predict_new_user_rating(most_rated_movies_deviation_matrix, absolute_deviation_matrix, raw_training_data)
