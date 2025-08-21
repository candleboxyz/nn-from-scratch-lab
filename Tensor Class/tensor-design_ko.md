
## '텐서'의 구현

Tensor가 무엇을 가져야 하지?
- 데이터
- gradient
- shape


---

### **v0.1**

- **v0.1.0**
    - <u>feat</u>: 텐서의 element-wise 합 연산
- **v0.1.a1**
    - <u>feat</u>: return `Tensor`, not `list`
- **v0.1.a2**
    - <u>fix</u>: `require_grad` 대응 수정  
        : `or` 연산 사용
- **v0.1.b**
    - <u>feat</u>: 텐서의 element-wise 곱 연산
    - `print()` 대응으로 `__str__` 대신 `__repr__` 사용  
    ※ https://stackoverflow.com/a/2626364/31232454  
    ※ https://docs.python.org/3/reference/datamodel.html#object.__repr__

### **v0.2**

- **v0.2.0**
    - <u>feat</u>: computational graph  
    : 각 텐서 인스턴스가 연산 그래프의 노드로 역할
- **v0.2.a**
    - <u>feat</u>: `_backward()` & `backward()`  
    : 연산 그래프를 바탕으로 역전파에 의한 그레이디언트 연산
