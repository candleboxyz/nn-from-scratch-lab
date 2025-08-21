"""
[ tensor_v0.2.0.py ]

For autograd, remembering the sequence of computations is essential
since it is needed for applying the chain-rule.

This version will add implementation of computational graph, which would
act as a memory of computational order.

To achieve this goal, each `Tensor` instance has to record it's original
form ― the "parent tensor" ― and also what operation was held.
"""

import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _prev=(), _op=""):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad

        self.shape = self.data.shape
        self.grad = None
        self._prev = _prev
        # ↳ this needs to be a tuple, not a set, because the same tensor
        #   could be used multiple times and the gradient accumulation
        #   count has to exactly match the quantity of computations.
        self._op = _op

    def __repr__(self):  # to print out the data
        if self.grad is not None:
            return f"Tensor(data={self.data}, grad={self.grad})"
        return f"Tensor({self.data})"

    def __add__(self, other):
        return Tensor(
            data=self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="+",
        )

    def __mul__(self, other):
        return Tensor(
            data=self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="*",
        )


# --- goal: make this work ---

# computational graph
a = Tensor([2], requires_grad=True)
b = Tensor([3], requires_grad=True)
c = a + b
d = c * Tensor([2])

print(f"c의 값: {c.data}")
print(f"c의 부모들: {c._prev}")  # (Tensor(...), Tensor(...))

print(f"\nc = a + b = {c.data}")
print(f"c의 부모 개수: {len(c._prev)}")  # 2
print(f"c를 만든 연산: {c._op}")  # '+'

print(f"\nd = c * 2 = {d.data}")
print(f"d의 부모 개수: {len(d._prev)}")  # 2
print(f"d를 만든 연산: {d._op}")  # '*'
