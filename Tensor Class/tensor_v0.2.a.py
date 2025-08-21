"""
[ tensor_v0.2.a.py ]
"""

import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _prev=(), _op=""):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad

        self.shape = self.data.shape
        self.grad = None
        self._prev: tuple[Tensor] = _prev

        self._op = _op
        self._backward = lambda: None

    def __repr__(self):  # to print out the data
        if self.grad is not None:
            return f"Tensor(data={self.data}, grad={self.grad})"
        return f"Tensor({self.data})"

    def __add__(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            data=self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="+",
        )

        def _backward():
            # guard clause ('early return' for no upstream gradient)
            if out.grad is None:
                return

            # --- Gradient Accumulation ---
            # local gradient of `__sum__`: d(a+b)/d(a) = 1, d(a+b)/d(b) = 1
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad

            if other.requires_grad:
                other.grad = (
                    other.grad + out.grad if other.grad is not None else out.grad
                )

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            data=self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
            _op="*",
        )

        def _backward() -> None:
            # guard clause (early return for no upstream gradient)
            if out.grad is None:
                return

            # --- Gradient Accumulation ---
            # local gradient of `__mul__`: d(a*b)/d(a) = b, d(a*b)/d(b) = a
            if self.requires_grad:
                # d(a*b)/d(a) = b ← `other.data`
                accumulating_grad = out.grad * other.data
                self.grad = (
                    self.grad + accumulating_grad
                    if self.grad is not None
                    else accumulating_grad
                )

            if other.requires_grad:
                # d(a*b)/d(b) = a ← `self.data`
                accumulating_grad = out.grad * self.data
                other.grad = (
                    other.grad + accumulating_grad
                    if other.grad is not None
                    else accumulating_grad
                )

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Perform Back-propagation"""

        # 1. Topology align (DFS)
        topo = []
        visited = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # 2. Initialize self's gradient to 1 (starting point)  #*★
        self.grad = np.ones_like(self.data)

        # 3. Call each node's `_backward()` in inverse sequence
        for v in reversed(topo):
            v._backward()


# --- goal: make this work ---

# simple test
x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)
z = x * y  # z = 6

z.backward()

print(f"z = {z.data}")
print(f"dz/dx = {x.grad}")  # 3.0 expected
print(f"dz/dy = {y.grad}")  # 2.0 expected

# more complicated computational graph
a = Tensor([2.0], requires_grad=True)
b = a + a  # same tensor as operands
b.backward()
print(f"db/da = {a.grad}")  # 2.0 expected
