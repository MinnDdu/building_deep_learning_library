"""
Here's a function that can train a Neural Network
"""

from tensor import Tensor
from nn import NeuralNetwork
from loss import MSE, Loss
from optim import SGD, Optimizer
from data import DataIterator, BatchIterator

def train(net: NeuralNetwork, 
          inputs: Tensor, 
          targets: Tensor, 
          num_ephochs: int = 5000, 
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:

    for epoch in range(num_ephochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(f'epoch {epoch} - loss: {epoch_loss}')
