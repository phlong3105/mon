from torch import FloatTensor
from torch import tensor

from one.core import *

a = [3]
b = Tensor([3, 10, 10])
print(a == list(b.size()))
