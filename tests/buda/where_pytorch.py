
from pybuda.module import PyBudaModule
import torch
import torch.nn as nn
import numpy as np
import tvm
import tvm.relay as relay

from tvm.contrib.pybuda_compile import compile_tvm_for_buda


def run_test():
    class WherePytorch(nn.Module):
        def __init__(self):
            super().__init__()
            self.causal_mask = torch.tril(torch.ones((32,32), dtype=torch.uint8)).view(1, 1, 32, 32)
            self.bias = torch.tensor(-1e4)

        def forward(self, x1):
            out = torch.where(self.causal_mask, x1, self.bias.to(x1.dtype))
            out = nn.Softmax(dim=-1)(out)
            return out


    shape = (1, 1, 32, 32)
    x1 = torch.rand(*shape)
    torchmod = WherePytorch()
    traced_model = torch.jit.trace(torchmod, x1)
    input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
    mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_list)
    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    print(mod.functions)

    func = compile_tvm_for_buda(mod, params)

    res = func(x1)
    res_pt = torchmod(x1)
    
    if not isinstance(res, (list, tuple)):
        res = [res]
        res_pt = [res_pt]

    for rb, rpt in zip(res, res_pt):
        print(f"Results correct: {np.allclose(rb.numpy(), rpt.detach().numpy(), atol=1e-6)}")


if __name__ == "__main__":
    run_test()