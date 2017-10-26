import numpy as np

def load_data_as_np():
    raw_training_data = np.loadtxt('../data/TrainingRatings.txt', delimiter=',', skiprows=1)
    print(raw_training_data.shape)