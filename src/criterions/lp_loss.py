class l2_loss:
    """L2 Loss (total l2 loss, to get mean l2_loss, please divide by the number of samples)"""

    # def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray = None) -> None:
    #     self.y_true = y_true
    #     self.y_pred = y_pred
    #     self.X = X

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray = None):
        """Implements L2 Loss with $l2_loss(y_true, y_pred) = \sum_{i=1}^{m} (y_true-y_pred)^2$
        Args:
            y_true (np.ndarray): [description]
            y_pred (np.ndarray): [description]
        Returns:
            [type]: [description]
        """
        l2_loss = np.sum(np.square(y_true - y_pred))
        return l2_loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray = None):
        """
        Compute the gradient of the l2 loss function.
        """

        gradient_vector: np.ndarray = -np.matmul((y_true - y_pred).T, X)
        # rename it to reflect its gradient of beta
        dl2_dB = gradient_vector[:]
        return dl2_dB