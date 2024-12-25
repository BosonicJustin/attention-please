import numpy as np

# input is (N, M) matrix output is (N, M) matrix, which yields probabilities along dim=1
def softmax(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, "Input must be a 2D array"

    # To improve numerical stability, subtract the max value in each row, standard trick
    x_max = np.max(x, axis=1, keepdims=True)  # (N, 1), max for each row
    x_stable = x - x_max  # Stabilize values to prevent overflow, this will be invariant since exp(x+y= exp(x) * exp(y)

    exp_x = np.exp(x_stable)  # Compute exps
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)

    return exp_x / sum_exp_x  # Normalize to get probabilities

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return x * (x > 0)