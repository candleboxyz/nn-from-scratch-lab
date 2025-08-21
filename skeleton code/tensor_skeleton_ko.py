"""
[ tensor_skeleton_ko.py ]
"""

# import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        # TODO: 데이터 저장
        # TODO: gradient는 None으로 시작
        return None

    def __add__(self, other):
        # TODO: 두 텐서의 데이터의 합(요소별)
        # TODO: 새 Tensor 반환
        return None


# --- 목표: 이게 작동하게 만들기 ---

a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = a + b
print(c)  # 출력되면 성공!
print(type(c))
