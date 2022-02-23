import torch
from torch import nn

import numpy as np
import tvm
import tvm.relay as relay

from pybuda_runtime import compile_tvm_for_buda


class KQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(128, 128, bias=True)
        self.key = nn.Linear(128, 128, bias=True)
        self.value = nn.Linear(128, 128, bias=True)

        self.num_attention_heads = 4
        self.attention_head_size = 32
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query = self.query(hidden_states)
        query = self.transpose_for_scores(query)
        key = self.key(hidden_states)
        key = self.transpose_for_scores(key)
        value = self.value(hidden_states)
        value = self.transpose_for_scores(value)

        key_t = key.transpose(-1, -2)
        attention_scores = torch.matmul(query, key_t)
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)

        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_layer_shape)

        return context


def run_test():
    shape = (1, 64, 128)
    hidden_states = torch.rand(*shape)
    torchmod = KQ().eval()
    traced_model = torch.jit.trace(torchmod, (hidden_states))
    input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
    mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_list)
    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))
    
    func = compile_tvm_for_buda(mod, params)
    
    res = func(hidden_states).numpy()

    res_pt = torchmod(hidden_states).detach().numpy()

    print(f"Results correct: {np.allclose(res, res_pt, atol=1e-6)}")

if __name__ == "__main__":
    run_test()