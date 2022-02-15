import torch
from torch import nn

import numpy as np
import tvm
import tvm.relay as relay

from ctypes import cast, POINTER
from pybuda._C import cast_graph, dump_graph
import pybuda._C.graph as pygraph




class DoubleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(32, 32, bias=True)
        self.l2 = nn.Linear(32, 32, bias=True)

    def forward(self, x1, x2):
        m1 = self.l1(x1)
        m2 = self.l2(x2)
        return m1 + m2


shape = (32, 32)
x1 = torch.rand(*shape)
x2 = torch.rand(*shape)
torchmod = DoubleLinear()
traced_model = torch.jit.trace(torchmod, (x1, x2))
input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_list)
print(mod)
print(params)
mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))


if not tvm.get_global_func("relay.ext.buda", True):
    print("Buda codegen not available")
    exit(-2)

mod = tvm.relay.op.contrib.buda.partition_for_buda(mod)
# print(mod)

ret = tvm.relay.build_module.build(mod, target="llvm", params=params)
# print(ret)

with tvm.transform.PassContext(opt_level=3):
    func = relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm").evaluate()


@tvm.register_func
def my_py_packed_func(*args):
    t = tuple(args)
    vp = t[-1].value
    graph = cast_graph(vp)

    inputs = [torch.from_numpy(npt.numpy()) for npt in t[:-1]]
    for idx, _ in enumerate(inputs):
        while len(inputs[idx].shape) < 4:
            inputs[idx] = inputs[idx].unsqueeze(0)
    
    inputs = tuple(inputs)
    res = pygraph.eval(graph, inputs)
    return tvm.runtime.ndarray.array(res[0].numpy())

# z = np.zeros(32)
# o = np.ones(32) * 31
# grid = np.linspace(z, o, 32)

res = func(x1, x2).numpy()

res_pt = torchmod(x1, x2).detach().numpy()

print(f"Results correct: {np.allclose(res, res_pt, atol=1e-6)}")


