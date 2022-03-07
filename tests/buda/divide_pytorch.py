
from pybuda.module import PyBudaModule
import torch
import torch.nn as nn
import numpy as np
import tvm
import tvm.relay as relay

from tvm.contrib.pybuda_compile import compile_tvm_for_buda


def run_test():
    class DoubleLinear(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2):
            out = x1 / x2
            return out


    shape = (64, 128)
    x1 = torch.rand(*shape)
    x2 = torch.rand(*shape)
    torchmod = DoubleLinear()
    traced_model = torch.jit.trace(torchmod, (x1, x2))
    input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
    mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_list)
    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    print(mod.functions)

    func = compile_tvm_for_buda(mod, params)

    res = func(x1, x2).numpy()

    res_pt = torchmod(x1, x2).detach().numpy()

    print(f"Results correct: {np.allclose(res, res_pt, atol=1e-6)}")


if __name__ == "__main__":
    run_test()