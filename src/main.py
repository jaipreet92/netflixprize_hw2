import data_loader
import similarity
import predict

K=20

if __name__ == "__main__":
    raw_training_data = data_loader.load_data_from_saved_py()
    raw_testing_data = data_loader.load_testing_data_from_saved_npy()
    user_to_movies_matrix = data_loader.build_user_x_movie_matrix(raw_training_data)

    partial_user_to_movies_matrix = user_to_movies_matrix[0:2000, 0:100]

    pearson_coefficients = similarity.calculate_pearson_coefficient(partial_user_to_movies_matrix)

    predict.test_predictions(pearson_coefficients, raw_testing_data, partial_user_to_movies_matrix)
