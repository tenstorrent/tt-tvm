from pybuda.module import PyBudaModule
import torch

import numpy as np
import tvm
import tvm.relay as relay

from ctypes import cast, POINTER
from pybuda._C import cast_graph, dump_graph
import pybuda._C.graph as pygraph
from pybuda import TTDeviceType, TTDevice
import pybuda
# shape = (64, 64)

# torchmod = torch.nn.Linear(64, 64)
# act = torch.rand(*shape)
# scr = torch.jit.trace(torchmod, act)
# mod, params = tvm.relay.frontend.from_pytorch(scr, [('input', (64, 64))])
# print(mod)
# print(params)
# mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))
# import pdb; pdb.set_trace()

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
print(mod)


if not tvm.get_global_func("relay.ext.buda", True):
    print("Buda codegen not available")
    exit(-2)

mod, params = tvm.relay.op.contrib.compile_for_buda(mod, target="llvm", params=params)

mod = tvm.relay.op.contrib.buda.partition_for_buda(mod)
# print(mod)

ret = tvm.relay.build_module.build(mod, target="llvm", params=params)
# print(ret)

with tvm.transform.PassContext(opt_level=5):
    func = relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm").evaluate()

in0 = np.random.random((shape)).astype("float32") - 0.5
in1 = np.random.random((shape)).astype("float32") - 0.5

# import pdb; pdb.set_trace()
# @tvm.register_func
# def initialize_device_packed_func(*agrs):

@tvm.register_func
def my_py_packed_func(*args):

    t = tuple(args)
    vp = t[-1].value
    graph = cast_graph(vp)

    # inputs = [torch.from_numpy(npt.numpy()) for npt in t[:-1]]
    inputs = []
    input_type = []
    node_name = []
    for arg in t[:-1]:
        if isinstance(arg, str):
            _type, _name = arg.split("|||")
            input_type.append(_type)
            node_name.append(_name)
        elif isinstance(arg, tvm.runtime.ndarray.NDArray):
            inputs.append(torch.from_numpy(arg.numpy()) )
        else:
            assert 0, "Unexpected data type"

    for idx, _ in enumerate(inputs):
        while len(inputs[idx].shape) < 4:
            inputs[idx] = inputs[idx].unsqueeze(0)

    # Create parameters
    tt0 = TTDevice("tt0", devtype=TTDeviceType.Model)

    tvm_module = PyBudaModule("tvm")

    for _type, _name, _data in zip(input_type, node_name, inputs):
        if _type == "parameter":
            param = pybuda.Parameter(
                *_data.shape, 
                requires_grad=True,
                name=_name,
                value=_data,
            )
            tvm_module._parameters[_name] = param

    tt0.place_module(tvm_module)
    inputs = tuple(inputs)
    res = pygraph.eval(graph, inputs, tt0)
    return tvm.runtime.ndarray.array(res[0][0].numpy())

z = np.zeros(32)
o = np.ones(32) * 63
grid = np.linspace(z, o, 64)

res = func(in0, in1).numpy()

def np_fun(a, b, c, d):
    mm1 = np.matmul(a, c)
    mm2 = np.matmul(b, d)
    sum = mm1 + mm2
    return sum
    

sum = np_fun(in0, in1, w1_np.transpose(), w2_np.transpose())

print(f"Results correct: {np.allclose(res, sum, atol=1e-6)}")



