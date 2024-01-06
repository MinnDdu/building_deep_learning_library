"""

"""
import numpy as np
from tensor import Tensor

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
    Subclass of Layer class
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
    
class Activation(Layer):
    """
    Subclass of Layer class
    Activation Layer gives a non-linearity to the inputs
    """
    def __init__(self, f, f_prime):
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs # save copy of the inputs for later backprop
        return self.f(inputs)
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        If y = f(x) and f(x) = g(z)
        dy/dz = f'(x) * g'(z) - The chain rule
        """
        return grad * self.f_prime(self.inputs)

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    """
    f(x) = tanh(x)
    f'(x) = sech^2(x) = 1 - tanh^2(x)
    """
    y = tanh(x)
    return 1 - y**2

class Tanh(Activation):
    """
    Subclass of Activation class
    Hyperbolic Tangent Activation Function
    """
    def __init__(self):
        super().__init__(tanh, tanh_prime)