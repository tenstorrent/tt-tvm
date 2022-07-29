"""Dropout """
import tvm
from tvm import te
from .. import utils
from .. import tag

# For the purpose of compiling Dropout for Pybuda/Buda, the Topi implementation 
# would be a simple NOP to match the inference behavior. Actual dropout will be
# implemented in Buda

def dropout(x, rate=0.0):
    """Perform softmax activation on the data.

    Parameters
    ----------
    data : tvm.te.Tensor
        can be any dimension

    rate : float
        dropout rate

    Returns
    -------
    output : tvm.te.Tensor
        output shape is the same as input
    """
    return x