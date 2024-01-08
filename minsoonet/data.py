"""
The input data will be fed into network in batches.
This file provides some tools for iterating over data in batches.
"""

from typing import Any
import numpy as np
from tensor import Tensor

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> [tuple]:
        raise NotImplementedError
    
class BatchIterator:
    def __init__(self, batch_size: int = 32, suffle: bool = True) -> None:
        self.batch_size = batch_size
        self.suffle = suffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> [tuple]:
        starts = np.arange(0, len(inputs), self.batch_size) # it will be [0, 32, 64, ...]
        if self.suffle:
            np.random.shuffle(starts)
        
        for i, start in enumerate(starts):
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield (batch_inputs, batch_targets)
