import data_loader
import naive
import similarity

K = 20

if __name__ == "__main__":
    # Load testing and training data
    raw_training_data = data_loader.load_data_from_saved_py()
    raw_testing_data = data_loader.load_testing_data_from_saved_npy()

    # Form a user x movies matrix
    user_to_movies_matrix = data_loader.build_user_x_movie_matrix(raw_training_data)

    # Sample a portion of the data
    partial_user_to_movies_matrix = user_to_movies_matrix[0:10000, 0:100]

    # Calculate similarity between pairs of users
    pearson_coefficients = similarity.calculate_pearson_coefficient(partial_user_to_movies_matrix)

    # Predict the testing data
    naive.test_predictions(pearson_coefficients, raw_testing_data, partial_user_to_movies_matrix, user_to_movies_matrix)
