import torch
import tvm
import tvm.relay

import numpy as np

from transformers import GPT2Model, GPT2Config
from pybuda_runtime import compile_tvm_for_buda

def main():
    config = GPT2Config()
    config.activation_function = "gelu"
    config.use_cache = True
    model = GPT2Model(config)
    # model = GPT2Model.from_pretrained('gpt2')

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # hidden_states = model.embeddings(tokens_tensor)
    shape = (1, 32, 768)
    hidden_states = torch.rand(*shape)

    torchmod = model.h[0]

    traced_model = torch.jit.trace(torchmod, hidden_states)
    input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]

    mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, input_list)
    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    func = compile_tvm_for_buda(mod, params)

    res = func(hidden_states)
    if isinstance(res, (list, tuple)):
        res = res[0]
    res = res.numpy()

    res_pt = torchmod(hidden_states)
    if isinstance(res_pt, (list, tuple)):
        res_pt = res_pt[0]

    res_pt = res_pt.detach().numpy()

    print(f"Results correct: {np.allclose(res, res_pt, atol=1e-6)}")

if __name__ == "__main__":
    main()
