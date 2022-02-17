import torch
from torch import nn

import numpy as np
import tvm
import tvm.relay as relay

from ctypes import cast, POINTER
from pybuda._C import cast_graph, dump_graph
import pybuda._C.graph as pygraph




class KQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(128, 128, bias=True)
        self.key = nn.Linear(128, 128, bias=True)

        self.num_attention_heads = 4
        self.attention_head_size = 32

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query = self.query(hidden_states)
        query = self.transpose_for_scores(query)
        key = self.key(hidden_states)
        key = self.transpose_for_scores(key)

        key_t = key.transpose(-1, -2)
        scores = torch.matmul(query, key_t)
        
        return scores

shape = (1, 64, 128)
hidden_states = torch.rand(*shape)
torchmod = KQ()
traced_model = torch.jit.trace(torchmod, (hidden_states))
input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_list)
print(mod.functions)

target = "llvm"

with tvm.transform.PassContext(opt_level=4):
    # model_opt, params_opt = tvm.relay.optimize(mod=mod, target=target, params=params)
    model_opt, params_opt = tvm.relay.op.contrib.buda.compile_for_buda(mod, target=target, params=params)

print(model_opt.functions)

if not tvm.get_global_func("relay.ext.buda", True):
    print("Buda codegen not available")
    exit(-2)

mod = tvm.relay.op.contrib.buda.partition_for_buda(model_opt)
print(mod.functions)

ret = tvm.relay.build_module.build(mod, target="llvm", params=params)


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
    import pdb; pdb.set_trace()
    res = pygraph.eval(graph, inputs)
    return tvm.runtime.ndarray.array(res[0].numpy())

# z = np.zeros(32)
# o = np.ones(32) * 31
# grid = np.linspace(z, o, 32)

res = func(hidden_states).numpy()

import pdb; pdb.set_trace()

res_pt = torchmod(hidden_states).detach().numpy()

print(f"Results correct: {np.allclose(res, res_pt, atol=1e-6)}")


