"""
A loss function shows how the result of prediction model is bad.

"""
import numpy as np
from minsoonet.tensor import Tensor

class Loss:
    """
    Simple mold of Loss functions
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    Loss Function 1 - Mean Squared Error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual)**2) / predicted.shape[-1]

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
