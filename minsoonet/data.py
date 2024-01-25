"""
The input data will be fed into network in batches.
This file provides some tools for iterating over data in batches.
"""

import numpy as np
from tensor import Tensor

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> [tuple]:
        raise NotImplementedError
    
class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> [tuple]:
        starts = np.arange(0, len(inputs), self.batch_size) # it will be [0, 32, 64, ...]
        if self.shuffle:
            np.random.shuffle(starts)
        
        for i, start in enumerate(starts):
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield (batch_inputs, batch_targets)
