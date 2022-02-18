from pybuda.module import PyBudaModule
import torch

import numpy as np
import tvm
import tvm.relay as relay


from tvm.contrib import graph_executor 


from ctypes import cast, POINTER
from pybuda._C import cast_graph, dump_graph
import pybuda._C.graph as pygraph
from pybuda import TTDeviceType, TTDevice
import pybuda



def compile_tvm_for_buda(mod, params):
    target = "llvm"
    mod, params = tvm.relay.op.contrib.compile_for_buda(mod, target=target, params=params)
    mod = tvm.relay.op.contrib.buda.partition_for_buda(mod)

    executor_factory = tvm.relay.build_module.build(mod, target=target, params=params)
    
    # device = tvm.runtime.ndarray.device(str(target), 0)
    # mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    # gmodule = graph_executor.GraphModule(executor_factory["default"](device))

    with tvm.transform.PassContext(opt_level=5):
        func = relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm").evaluate()

    

    return func


@tvm.register_func
def retrieve_pybuda_graph(*args):
    print("In retrieve_pybuda_graph")
    t = tuple(args)
    vp = t[-1].value
    graph = cast_graph(vp)

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
