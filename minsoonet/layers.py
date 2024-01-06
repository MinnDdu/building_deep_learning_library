"""

"""
import numpy as np
from minsoonet.tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.params = {}
        self.grads = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward Propagation

        """
        raise NotImplementedError
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward Propagation
        """
        raise NotImplementedError
    
class Linear(Layer):
    """
    A simple linear layer doing linear regression
    outputs = weights X inputs + bias
    inputs - (batch_size, input_size)
    outputs - (batch_size, output_size)
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__() # to get params dict initialization
        self.params["w"] = np.random.randn(output_size, input_size)
        self.params["b"] = np.random.randn(output_size, 1)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward Propagation of Linear Layer
        outputs = w @ inputs + b
        """
        self.inputs = inputs # save a copy of the inputs - we need when we do backprop
        return np.matmul(self.params['w'], self.inputs) + self.params['b']
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward Propagation of Linear Layer
        If y = f(x) and x = a @ b + c
        dy/da = f'(x) @ b.T
        dy/db = a.T @ f'(x)
        dy/dc = f'(x)
        f'(x) is regarded as 1 (or identity matrix)
        """
        self.grads['b'] = np.sum(grad, axis=0) # Since we are using batch dimension
        self.grads['w'] = np.matmul(grad, self.inputs)
        return np.matmul(self.params['w'], grad)