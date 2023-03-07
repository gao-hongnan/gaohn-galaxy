import numpy as np
from src.base.types import T

from src.base.criterion import Criterion

# TODO: Check sanity of this implementation, might be slightly wrong.
# TODO: Add grad call


class BinaryCrossEntropy(Criterion):
    def __init__(self, epsilon: float = 1e-10) -> None:
        self.epsilon = epsilon

    def __call__(self, y_trues: T, y_preds: T) -> float:
        """
        Computes the cross entropy loss between predicted and true class labels.
        Assume that y_true is one-hot encoded.

        See: https://stackoverflow.com/questions/47377222/what-is-the-problem-with-my-implementation-of-the-cross-entropy-function

        Parameters:
            y_preds (numpy array): predicted class probabilities with shape (batch_size, num_classes)
            y_trues (numpy array): true class labels with shape (batch_size, )

        Returns:
            loss (float): the cross entropy loss
        """
        # Consider to check y_true is a one-hot encoded matrix

        num_samples = y_trues.shape[0]

        y_preds = np.clip(y_preds, self.epsilon, 1.0 - self.epsilon)

        # loss_matrix is same shape as y_true and y_pred since we are
        # just performing element wise operations on both of them.
        loss_matrix = y_trues * np.log(y_preds) + (1 - y_trues) * np.log(1 - y_preds)

        # we sum up all the loss for each individual sample
        total_loss = -np.sum(loss_matrix, axis=None)

        # we then average out the total loss across m samples, but we squeeze it to
        # make it a scalar; squeeze along axis = None since there is no column axix
        mean_loss = np.squeeze(total_loss / num_samples, axis=None)

        return mean_loss
