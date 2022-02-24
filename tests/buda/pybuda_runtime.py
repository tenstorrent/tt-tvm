from pybuda.module import PyBudaModule
import torch

import numpy as np
import tvm
import tvm.relay as relay


from tvm.contrib import graph_executor 


from ctypes import c_void_p
from pybuda._C import cast_c_void_p_to_graph, cast_graph_to_c_void_p, dump_graph
import pybuda._C.graph as pygraph
import pybuda
import pybuda.op2
import pybuda.op2.nn as nn

from pybuda import PyBudaModule, TTDevice, TTDeviceType, Tensor, pybuda_compile


def compile_tvm_for_buda(mod, params):
    target = "llvm"
    mod, params = tvm.relay.op.contrib.compile_for_buda(mod, target=target, params=params)
    mod = tvm.relay.op.contrib.buda.partition_for_buda(mod)

    executor_factory = tvm.relay.build_module.build(mod, target=target, params=params)
    
    with tvm.transform.PassContext(opt_level=5):
        func = relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm").evaluate()

    

    return func


@tvm.register_func
def retrieve_pybuda_graph(*args):
    print("In retrieve_pybuda_graph")
    t = tuple(args)
    vp = t[-1].value
    graph = cast_c_void_p_to_graph(vp)

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



class SubGraph(PyBudaModule):
    def __init__(self, fn_name, attributes):
        super().__init__("subgraph")
        self.fn_name = fn_name
        self.attributes = attributes

    def forward(self, *act):
        if self.fn_name == "nn.softmax":
            return nn.Softmax("softmax", act[0], dim=self.attributes[0])


@tvm.register_func
def expand_compound_ops(*args):
    print("In expand_compound_ops")
    t = tuple(args)
    num_inptus = t[0]
    inputs = t[1 : t[0] * 4 + 1]

    num_attributes = t[t[0] * 4 + 1]
    attributes = t[t[0] * 4 + 2 : -1]

    fn_name = t[-1]
    mod = SubGraph(fn_name=fn_name, attributes=attributes)

    tt0 = TTDevice("tt0", devtype=TTDeviceType.Golden)
    tt0.place_module(mod)

    shapes = [t[i : i + 4] for i in range(0, len(t) - 1, 4)]
    acts = []
    for i in range(num_inptus):
        acts.append(Tensor.create_from_torch(torch.rand(shapes[0])))
    
    acts = tuple(acts)
    graph, _ = tt0.generate_graph(*acts, return_intermediate=False)
    vp = c_void_p(cast_graph_to_c_void_p(graph))

    return vp
