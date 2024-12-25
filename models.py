import numpy as np

from data_utils import generate_random_sequence, one_hot_encode_sequence
from activations import sigmoid, relu, softmax

class EmbeddingMatrix:
    def __init__(self, input_dim, output_dim=256):
        self.M = np.random.randn(input_dim, output_dim)

    # Expected input is (N, d_model)
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x.dot(self.M)


class SelfAttention:
    def __init__(self, d_model, q_dim, v_dim):
        self.q_matrix = EmbeddingMatrix(d_model, q_dim)
        self.k_matrix = EmbeddingMatrix(d_model, q_dim)
        self.v_matrix = EmbeddingMatrix(d_model, v_dim)

        self.norm_constant = 1 / np.sqrt(d_model)

        self.sm = lambda x : softmax(x * self.norm_constant)

    def __call__(self, x):
        Q = self.q_matrix(x)
        K = self.k_matrix(x)
        V = self.v_matrix(x)

        return self.sm(Q.dot(K.T)).dot(V)


class ReLULayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(output_dim)
        self.activation = relu

    def __call__(self, x):
        return self.activation(x.dot(self.W) + self.b)


class SigmoidLayer:
    def __init__(self, input_dim):
        self.W = np.random.randn(1, input_dim)
        self.b = np.random.randn(1)
        self.activation = sigmoid

    def __call__(self, x):
        return self.activation(self.W.dot(x) + self.b)


# 2 layer MLP
class MLP:
    def __init__(self, input_dim, output_dim1):
        self.l1 = ReLULayer(input_dim, output_dim1)
        self.l2 = SigmoidLayer(output_dim1 * 20) # TODO: for now limiting to sequences of 20 members

    def __call__(self, x):
        out1 = self.l1(x)

        return self.l2(np.expand_dims(out1.flatten(), axis=1))


class SumClassificationModel:
    def __init__(self, d_model, q_dim, v_dim):
        # We'll use one-hot encoding for numbers between 1 and 10
        self.input_embedding = EmbeddingMatrix(input_dim=10, output_dim=d_model)

        self.A1 = SelfAttention(d_model, q_dim, v_dim)
        self.mlp = MLP(v_dim, 5)

    # Expected input is one-hot encoded number between 1 and 10: (N, 10)
    def __call__(self, x):
        embedding = self.input_embedding(x)
        attention = self.A1(embedding)

        return self.mlp(attention).item()


seq = generate_random_sequence(20)
encoded_seq = one_hot_encode_sequence(seq)

model = SumClassificationModel(30, 10, 10)
