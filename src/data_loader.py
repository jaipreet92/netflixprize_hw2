import numpy as np


def load_data_from_txt():
    raw_training_data = np.loadtxt('../data/TrainingRatings.txt', delimiter=',', skiprows=1)
    print(raw_training_data.shape)
    #np.save('../data/modified/raw_data_np.npy', raw_training_data)
    return raw_training_data

def load_testing_data_from_txt():
    raw_testing_data = np.loadtxt('../data/TestingRatings.txt', delimiter=',', skiprows=1)
    print(raw_testing_data.shape)
    np.save('../data/modified/raw_testing_data_np.npy', raw_testing_data)
    return raw_testing_data

def load_data_from_saved_py():
    raw_training_data = np.load('../data/modified/raw_data_np.npy')
    print(raw_training_data.shape)
    return raw_training_data

def load_testing_data_from_saved_npy():
    raw_testing_data = np.load('../data/modified/raw_testing_data_np.npy')
    print(raw_testing_data.shape)
    return raw_testing_data


def load_newuser_training_data_from_text():
    training_data = np.loadtxt('../data/modified/TrainingRatings_NewUser.txt', delimiter=',', skiprows=1)
    np.save('../data/modified/newuser_training_data_np.npy', training_data)
    print(training_data.shape)
    return training_data

def load_movie_titles_from_txt():
    movie_titles = np.loadtxt('../data/movie_titles.txt', delimiter=',', skiprows=1, dtype=str)
    print(movie_titles.shape)
    return movie_titles

def load_newuser_training_data_from_np():
    raw_training_data = np.load('../data/modified/newuser_training_data_np.npy')
    print(raw_training_data.shape)
    return raw_training_data


userid_to_idx = {}
movieid_to_idx = {}


def build_user_x_movie_matrix(raw_training_data):
    """
    Constructs a user x movies matrix with all the ratings as the values. If there is no rating, the matrix
    element is 0 valued. This is handled by the code further down calculating the coefficients
    :param raw_training_data:
    :return:
    """
    row, column = raw_training_data.shape

    num_users = len(np.unique(raw_training_data[:,1], return_counts=True)[0])
    num_movies = len(np.unique(raw_training_data[:,0], return_counts=True)[0])
    user_to_ratings = np.empty((num_users, num_movies))

    curr_x = 0
    curr_y = 0
    for i in range(row):
        curr_training_sample = raw_training_data[i]
        movie_id = curr_training_sample[0]
        user_id = curr_training_sample[1]
        rating = curr_training_sample[2]

        if user_id not in userid_to_idx:
            userid_to_idx[user_id] = curr_x
            curr_x += 1

        if movie_id not in movieid_to_idx:
            movieid_to_idx[movie_id] = curr_y
            curr_y += 1

        user_idx = userid_to_idx[user_id]
        movie_idx = movieid_to_idx[movie_id]

        user_to_ratings[user_idx][movie_idx] = rating

    return user_to_ratings
