import numpy as np
def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


if __name__ == "__main__":
    m = np.random.randn(2, 2, 2) + 2
    m = softmax(m)
    m = m.sum(axis=-2)
    print(m)