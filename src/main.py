import data_loader

if __name__ == "__main__":
    raw_training_data= data_loader.load_data_from_saved_py()

    user_to_ratings = data_loader.build_user_x_movie_matrix(raw_training_data)