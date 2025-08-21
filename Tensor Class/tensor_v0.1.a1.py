"""
[ tensor_v0.1.a1.py ]
"""

import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad

        self.shape = self.data.shape
        self.grad = None

    def __str__(self):  # to print out the data
        return f"{self.data}"

    def __add__(self, other):
        return Tensor(
            data=self.data + other.data,
            requires_grad=self.requires_grad,  # ? check the logic
        )


# --- goal: make this work ---

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = a + b
print(c)
print(type(c))
