import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    from functools import reduce

    u = reduce(
        lambda x, _: (t := data @ x) / (t @ t) ** 0.5,
        range(num_steps),
        np.random.rand(data.shape[0]),
    )
    return float((v := u @ data @ u / (u @ u) ** 0.5)), u
