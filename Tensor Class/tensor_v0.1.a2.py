"""
[ tensor_v0.1.a2.py ]
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
            requires_grad=self.requires_grad or other.requires_grad,
            # ↳ `requires_grad` needs to be `True` if either Tensor has `True`
        )


# --- goal: make this work ---

a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=False)
c = a + b
print(c)
print(type(c))

print(f"a.requires_grad: {a.requires_grad}")  # True
print(f"b.requires_grad: {b.requires_grad}")  # False
print(f"c.requires_grad: {c.requires_grad}")  # needs to be True!
