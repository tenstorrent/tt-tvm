from pybuda.module import PyBudaModule
import torch

import numpy as np
import tvm
import tvm.relay as relay

from tvm.contrib.pybuda_compile import compile_tvm_for_buda


def run_test():
    shape = (256, 32)
    params = {}
    w1_np = np.random.random((shape)).astype("float32") - 0.5
    w2_np = np.random.random((shape)).astype("float32") - 0.5

    w1 = relay.const(tvm.nd.array(w1_np))
    params["weight"] = w1.data
    x1 = relay.var("x1", shape=shape)
    # wt1 = relay.transpose(w1)
    m1 = relay.nn.dense(x1, w1)
    x2 = relay.var("x2", shape=shape)

    w2 = relay.const(tvm.nd.array(w2_np))
    params["weight2"] = w2.data
    # wt2 = relay.transpose(w2)
    m2 = relay.nn.dense(x2, w2)

    y = relay.add(m1, m2)

    func = relay.Function([x1, x2], y)
    mod = tvm.IRModule.from_expr(func)
    print(mod.functions)

    func = compile_tvm_for_buda(mod, params)

    in0 = np.random.random((shape)).astype("float32") - 0.5
    in1 = np.random.random((shape)).astype("float32") - 0.5


    res = func(in0, in1).numpy()

    def np_fun(a, b, c, d):
        mm1 = np.matmul(a, c)
        mm2 = np.matmul(b, d)
        sum = mm1 + mm2
        return sum
    

    sum = np_fun(in0, in1, w1_np.transpose(), w2_np.transpose())

    print(f"Results correct: {np.allclose(res, sum, atol=1e-6)}")


if __name__ == "__main__":
    run_test()
