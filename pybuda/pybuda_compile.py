import torch

import numpy as np
import tvm
import tvm.relay as relay


from tvm.contrib import graph_executor 


from ctypes import c_void_p
import copy

passed_to_buda = None
def retrieve_vars_passed_to_buda():
    return passed_to_buda

json_graph = {}
@tvm.register_func
def retrieve_json_graph(*args):
    t = tuple(args)
    global json_graph
    json_graph["func_name"] = t[0]
    json_graph["graph"] = t[1]
    json_graph["param_names"] = t[2]


def compile_pytorch_for_buda(torchmod, *inputs):
    traced_model = torch.jit.trace(torchmod, inputs)
    input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
    mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_list)
    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    _, buda_params = compile_tvm_for_buda(mod, params, return_params=True)
    json_graph["params"] = {name:v.numpy() for (k, v), name in zip(buda_params.items(), json_graph["param_names"])}

    return copy.deepcopy(json_graph)

def compile_tvm_for_buda(mod, params, return_params=False):
    target = "llvm"
    mod, params = tvm.relay.op.contrib.compile_for_buda(mod, target=target, params=params)
    mod, buda_params = tvm.relay.op.contrib.buda.partition_for_buda(mod)

    executor_factory = tvm.relay.build_module.build(mod, target=target, params=params)
    
    with tvm.transform.PassContext(opt_level=5):
        func = relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm").evaluate()

    if return_params:
        return func, buda_params
    else:
        return func

@tvm.register_func
def retrieve_pybuda_graph(*args):
    print("In retrieve_pybuda_graph")
#     t = tuple(args)
#     vp = t[-1].value
#     graph = cast_c_void_p_to_graph(vp)
#     broadcast_shapes(graph)

#     inputs = []
#     input_type = []
#     node_name = []

#     for arg in t[:-1]:
#         if isinstance(arg, str):
#             _type, _name = arg.split("|||")
#             input_type.append(_type)
#             node_name.append(_name)
#         elif isinstance(arg, tvm.runtime.ndarray.NDArray):
#             inputs.append(torch.from_numpy(arg.numpy()) )
#         else:
#             assert 0, "Unexpected data type"

#     for idx, _ in enumerate(inputs):
#         while len(inputs[idx].shape) < 4:
#             inputs[idx] = inputs[idx].unsqueeze(0)

#     # Pad input data to TILE_DIM
#     input_shapes = get_graph_input_shapes(graph)
#     for i, (_data, buda_shape) in enumerate(zip(inputs, input_shapes)):
#         # Last 2 dimensions needs to be padded to TILE_DIM
#         if _data.shape[-2:] != torch.Size(buda_shape[-2:]):
#             numpy_data = _data.numpy()

#             # calculate padding on last 2 dimensions
#             pad = [(0, 0), (0, 0)] + [
#                 (0, a_i - b_i) for a_i, b_i in zip(buda_shape[-2:], numpy_data.shape[-2:])
#             ]
#             inputs[i] = torch.from_numpy(np.pad(numpy_data, pad, mode='constant')) # default pads with zero

#     # Create parameters
#     tt0 = TTDevice("tt0", devtype=TTDeviceType.Model)

#     tvm_module = PyBudaModule("tvm")

#     for _type, _name, _data in zip(input_type, node_name, inputs):
#         if _type == "parameter":
#             param = pybuda.Parameter(
#                 *_data.shape, 
#                 requires_grad=True,
#                 name=_name,
#                 value=_data,
#             )
#             tvm_module._parameters[_name] = param

#     global passed_to_buda
#     passed_to_buda = inputs
#     tt0.place_module(tvm_module)
#     inputs = tuple(inputs)
#     res = pygraph.eval(graph, inputs, tt0, 1, 0, 1, {})
#     passed_to_buda.append(res[0][0])
#     return tvm.runtime.ndarray.array(res[0][0].numpy())



# class SubGraph(PyBudaModule):
#     def __init__(self, fn_name, attributes):
#         super().__init__("subgraph")
#         self.fn_name = fn_name
#         self.attributes = attributes

#     def forward(self, *act):
#         if self.fn_name == "nn.softmax":
#             return nn.Softmax("softmax", act[0], dim=self.attributes[0])

#         if self.fn_name == "layernorm":
#             return nn.Layernorm("layernorm", act[0], act[1], act[2], dim=self.attributes[0], epsilon=self.attributes[1])


@tvm.register_func
def expand_compound_ops(*args):
    print("In expand_compound_ops")
#     t = tuple(args)
#     num_inputs = t[0]
#     inputs = t[1 : t[0] * 4 + 1]

#     num_attributes = t[t[0] * 4 + 1]
#     attributes = t[t[0] * 4 + 2 : -1]
#     fn_name = t[-1]
#     mod = SubGraph(fn_name=fn_name, attributes=attributes)

#     tt0 = TTDevice("tt0", devtype=TTDeviceType.Golden)
#     tt0.place_module(mod)

#     shapes = [inputs[i : i + 4] for i in range(0, len(inputs) - 1, 4)]

#     acts = []
#     for i in range(num_inputs):
#         acts.append(Tensor.create_from_torch(torch.rand(shapes[0])))
    
#     acts = tuple(acts)
#     graph, _ = tt0.generate_graph(*acts, return_intermediate=False)
#     vp = c_void_p(cast_graph_to_c_void_p(graph))

#     return vp