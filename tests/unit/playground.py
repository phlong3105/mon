from torch import FloatTensor
from torch import tensor

from one.core import *

a = {
    "a": 1
}

b = a | {"a": 2}
print(b)
