import numpy as np


def generate_random_sequence(N):
    return np.random.randint(1, 11, size=N)


def one_hot_encode_sequence(numbers):
    one_hot_encoded = np.zeros((len(numbers), 10))
    one_hot_encoded[np.arange(len(numbers)), numbers - 1] = 1

    return one_hot_encoded