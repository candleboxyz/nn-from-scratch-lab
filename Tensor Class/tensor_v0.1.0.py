"""
[ tensor_v0.1.0.py ]
"""

import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        # TODO: save the data
        # TODO: gradient starts with None
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad

        self.shape = self.data.shape
        self.grad = None

    def __add__(self, other):
        # TODO: element-wise summation of the two tensors
        # TODO: returns new `Tensor`
        return self.data + other.data


# --- goal: make this work ---

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = a + b
print(c)
print(type(c))
