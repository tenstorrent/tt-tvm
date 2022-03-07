
from pybuda.module import PyBudaModule
import torch
import torch.nn as nn
import numpy as np
import tvm
import tvm.relay as relay

from tvm.contrib.pybuda_compile import compile_tvm_for_buda, retrieve_vars_passed_to_buda



def run_test():
    class SplitPytorch(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_dim = 128
            self.l = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)

            self.l.weight = torch.nn.Parameter(torch.round(self.l.weight * 100))
            self.l.bias = torch.nn.Parameter(torch.round(self.l.bias * 100))
            # causal_mask = torch.tril(torch.ones(*SoftmaxTest.shape, dtype=torch.uint8)).view(*SoftmaxTest.shape)
            # bias = torch.tensor(-1e4)

        def forward(self, x1):
            out = self.l(x1).reshape((1, -1, 3*self.embedding_dim))
            q, k, v = out.split(self.embedding_dim, dim=2)

            sqw = self.l.weight[:self.embedding_dim, :]
            skw = self.l.weight[self.embedding_dim:2*self.embedding_dim, :]
            svw = self.l.weight[2*self.embedding_dim:, :]

            sqb = self.l.bias[:self.embedding_dim]
            skb = self.l.bias[self.embedding_dim:2*self.embedding_dim]
            svb = self.l.bias[2*self.embedding_dim:]

            nq = torch.matmul(x1, sqw.transpose(-1, -2)) + sqb
            nk = torch.matmul(x1, skw.transpose(-1, -2)) + skb
            nv = torch.matmul(x1, svw.transpose(-1, -2)) + svb

            tot = torch.matmul(x1, self.l.weight.transpose(-1, -2)) + self.l.bias

            passed_to_buda = retrieve_vars_passed_to_buda()
            # print(f"Maximum error is: {(nk-k).max()}")
            # weights = torch.matmul(q, k.transpose(-1, -2))
            # probs =  nn.Softmax(dim=-1)(weights)

            return k


    torchmod = SplitPytorch()
    shape = (64, torchmod.embedding_dim)

    x1 = torch.round(torch.rand(*shape) * 100)
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
        # print(f"{rb.numpy()}")
        # print(f"{rpt.detach().numpy()}")

    res_pt = torchmod(x1)

if __name__ == "__main__":
    run_test()