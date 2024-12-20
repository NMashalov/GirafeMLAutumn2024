import numpy as np


class LaplaceDistribution:
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        """
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        """
        ####
        # Do not change the class outside of this block
        # Your code here
        return np.mean(np.abs(x - np.quantile(x, q=0.5, axis=0)), axis=0)
        ####

    def __init__(self, features):
        """
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        """
        ####
        # Do not change the class outside of this block
        features = features.reshape(features.shape[0], -1)  # helps with 1d case
        self.loc = np.quantile(features, 0.5, axis=0)  # YOUR CODE HERE
        self.scale = self.mean_abs_deviation_from_median(features)  # YOUR CODE HERE
        ####

    def logpdf(self, values):
        """
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        """
        ####
        # Do not change the class outside of this block
        return -np.abs(self.loc - values) / self.scale - np.log(2 * self.scale)
        ####

    def pdf(self, values):
        """
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        """
        return np.exp(self.logpdf(values))
