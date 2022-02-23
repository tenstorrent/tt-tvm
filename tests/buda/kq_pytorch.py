import torch
from torch import nn

import numpy as np
import tvm
import tvm.relay as relay

from ctypes import cast, POINTER
import pybuda._C.graph as pygraph

from pybuda_runtime import compile_tvm_for_buda
import math


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
        attention_scores = torch.matmul(query, key_t)
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        return attention_probs


def run_test():


    shape = (1, 64, 128)
    hidden_states = torch.rand(*shape)
    torchmod = KQ()
    traced_model = torch.jit.trace(torchmod, (hidden_states))
    input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
    mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_list)
    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))
    print(mod.functions)

    func = compile_tvm_for_buda(mod, params)

    res = func(hidden_states).numpy()

    res_pt = torchmod(hidden_states).detach().numpy()

    print(f"Results correct: {np.allclose(res, res_pt, atol=1e-6)}")

if __name__ == "__main__":
    run_test()