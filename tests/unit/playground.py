from torch import FloatTensor
from torch import tensor

from one.core import *

a = [19]
a = Tensor(a if isinstance(a, Sequence) else [a])
print(a)
