import numpy as np
import torch


class A:
    
    a = {
        "a": np.ndarray | torch.Tensor | None,
    }
    
    def __init__(self):
        self.b = self.a
        print(type(self.b["a"]))
        

class B(A):
    pass
    

b = list[B]()
b.append(B())
print(b, type(b))
