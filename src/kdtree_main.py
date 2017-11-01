import data_loader
import predict

K = 20

if __name__ == "__main__":
    raw_training_data = data_loader.load_data_from_saved_py()
    raw_testing_data = data_loader.load_testing_data_from_saved_npy()
    user_to_movies_matrix = data_loader.build_user_x_movie_matrix(raw_training_data)

    predict.predict_using_kdtree(raw_testing_data=raw_testing_data, user_to_movies_matrix=user_to_movies_matrix)
