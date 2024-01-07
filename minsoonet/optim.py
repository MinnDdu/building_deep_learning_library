"""
Use optimizer to adjust the parameters of the network
based on the gradients computed during backprop
"""
from nn import NeuralNetwork

class Optimizer:
    def step(self, nn: NeuralNetwork) -> None:
        raise NotImplementedError
    
class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    """
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, nn: NeuralNetwork) -> None:
        for param, grad in nn.params_and_grads():
            param -= self.lr * grad