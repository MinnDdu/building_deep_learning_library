"""
A Simple Neural Network is a simple collection of layers.
"""

from tensor import Tensor
from layers import Layer

class NeuralNetwork:
    def __init__(self, layers: [Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
