import data_loader
import kdtree

K = 100

user_to_movies_matrix_global = None

if __name__ == "__main__":
    raw_training_data = data_loader.load_data_from_saved_py()
    raw_testing_data = data_loader.load_testing_data_from_saved_npy()
    user_to_movies_matrix = data_loader.build_user_x_movie_matrix(raw_training_data)
    user_to_movies_matrix_global = user_to_movies_matrix

    deviation_matrix = kdtree.get_deviation_from_mean_matrix(user_to_movies_matrix)

    kdtree.predict_using_kdtree(raw_testing_data=raw_testing_data, user_to_movies_matrix=deviation_matrix)
