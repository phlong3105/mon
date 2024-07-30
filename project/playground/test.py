

class A:
    
    _task = "A"
    
    def __init__(self):
        self.a = 1
    

class B(A):
    
    _task = "B"
    
    def __init__(self):
        super().__init__()
        self.b = 2


b = B()
print(b.a)
print(b._task)
