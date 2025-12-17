# Self attention (clip feature) + Cross attention (score + clip feature) + MLP
from .regressor_v1 import Regressor as Regressor_v1
# CBAM (score + clip feature) + MLP
from .regressor_v2 import Regressor as Regressor_v2
# ECA (score + clip feature) + MLP
from .regressor_v3 import Regressor as Regressor_v3
# ECA (prompt feature + image feature) + MLP
from .regressor_v4 import Regressor as Regressor_v4
# Cross attention (prompt feature + image feature) + MLP
from .regressor_v5 import Regressor as Regressor_v5
# CBAM (prompt feature + image feature) + MLP
from .regressor_v6 import Regressor as Regressor_v6
# Gate fusion, SE layer (image feature + extra feature) + MLP
from .regressor_v7 import Regressor as Regressor_v7
# CBAM (image feature + extra feature) + MLP
from .regressor_v8 import Regressor as Regressor_v8
# Cross attention (prompt feature + image feature) + MLP (LBW)
from .regressor_v9 import Regressor as Regressor_v9
# Cross attention (prompt feature + image feature) + Dual MLP
from .regressor_v10 import Regressor as Regressor_v10
__all__ = [
    "Regressor_v1", 
    "Regressor_v2", 
    "Regressor_v3", 
    "Regressor_v4", 
    "Regressor_v5", 
    "Regressor_v6", 
    "Regressor_v7", 
    "Regressor_v8",
    "Regressor_v9",
    "Regressor_v10",
]