import numpy as np

class EmbeddingMatrix:
    def __init__(self, input_dim, output_dim=256):
        self.M = np.random.randn(input_dim, output_dim)

    def __call__(self, x):
        return x.dot(self.M)


# input is (N, M) matrix output is (N, M) matrix, which yields probabilities along dim=1
def softmax(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, "Input must be a 2D array"

    # To improve numerical stability, subtract the max value in each row, standard trick
    x_max = np.max(x, axis=1, keepdims=True)  # (N, 1), max for each row
    x_stable = x - x_max  # Stabilize values to prevent overflow, this will be invariant since exp(x+y= exp(x) * exp(y)

    exp_x = np.exp(x_stable)  # Compute exps
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)

    return exp_x / sum_exp_x  # Normalize to get probabilities

class SelfAttention:
    def __init__(self, d_model, q_dim, k_dim, v_dim):
        self.q_matrix = EmbeddingMatrix(d_model, q_dim)
        self.k_matrix = EmbeddingMatrix(d_model, k_dim)
        self.v_matrix = EmbeddingMatrix(d_model, v_dim)

        self.norm_constant = 1 / np.sqrt(d_model)

        self.sm = lambda x : softmax(x * self.norm_constant)

    def __call__(self, x):
        Q = self.q_matrix(x).M
        K = self.k_matrix(x).M
        V = self.v_matrix(x).M

        return self.sm(Q.dot(K.T)).dot(V)


class SumClassificationModel:
    def __init__(self):
        # We'll use one-hot encoding for numbers between 1 and 10
        self.input_embedding = EmbeddingMatrix(input_dim=10, output_dim=256)
