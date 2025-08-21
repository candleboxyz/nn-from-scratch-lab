"""
[ tensor_skeleton_en.py ]
"""

# import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        # TODO: save the data
        # TODO: gradient starts with None
        return None

    def __add__(self, other):
        # TODO: element-wise summation of the two tensors
        # TODO: returns new `Tensor`
        return None


# --- goal: make this work ---

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = a + b
print(c)  # if prints, success!
print(type(c))
