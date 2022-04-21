import torch

import numpy as np
import tvm
import tvm.relay as relay
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
from tvm.contrib import graph_executor
from loguru import logger

from ctypes import c_void_p
import copy
import json

from collections import OrderedDict
import pybuda

from json import JSONEncoder
from os.path import exists as file_exists


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


def load_tvm_graph(inputs, torchmod, compiler_cfg, allow_unsupported=False):
    """
    Loads TVM graph ported to the PyBuda from other frameworks (TensorFlow, Pytorch). Can eather
    run whole compilation process from specific framewrk to the PyBuda graph representation, or
    skip this compilation process and load serilized TVM graph which is already ported to PyBuda
    by initial invocation.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors

    torchmod: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device.

    compiler_cfg: CompilerConfig
        Compiler configurations
        
    Returns
    -------
    Dictionary, OrderedDict, Tuple, Boolean
        TVM ported graph, Weights, Input tensors, Constant evaluation
    """
    if compiler_cfg.tvm_graph_store_path != "" and compiler_cfg.tvm_graph_load_path != "":
        logger.warning(f"TVM serilization logic will be skipped as both store and load paths are provided")

    json_graph = compile_tvm_graph(inputs, torchmod, compiler_cfg, allow_unsupported=allow_unsupported)
    
    pytorch_inputs, weights = format_tvm_graph_weights(inputs, torchmod, compiler_cfg)

    serilize_and_store_tvm_graph(json_graph, compiler_cfg)

    return json_graph, pytorch_inputs, weights


def compile_tvm_graph(inputs, torchmod, compiler_cfg, allow_unsupported):
    """
    Compiles TVM graph ported to the PyBuda from other frameworks (TensorFlow, Pytorch). Can eather
    run whole compilation process or only load serilized TVM graph and thus increase test performance.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors
    
    torchmod: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device

    compiler_cfg: CompilerConfig
        Compiler configurations

    Returns
    -------
    Dictionary
        TVM ported graph
    """
    if compiler_cfg.tvm_graph_load_path != "" and compiler_cfg.tvm_graph_store_path == "" and compiler_cfg.enable_consteval:
        json_graph = load_serilized_tvm_graph(compiler_cfg.tvm_graph_load_path)
        if isinstance(torchmod, pybuda.module.PyTorchModule):
            torchmod.module.eval()
    elif isinstance(torchmod, pybuda.module.PyTorchModule):
        json_graph = compile_pytorch_for_buda(torchmod.module, compiler_cfg.enable_consteval, *inputs, allow_unsupported=allow_unsupported)
    elif isinstance(torchmod, pybuda.module.TFModule):
        # convert pytorch tensors to tf tensors
        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            tf_inputs = tuple(tf.convert_to_tensor(t.detach().numpy()) for t in inputs)
        else:
            tf_inputs = inputs
        json_graph = compile_tf_for_buda(torchmod.module, *tf_inputs, allow_unsupported=allow_unsupported)
    else:
        raise RuntimeError(f"Unsupported module type {type(torchmod)}")

    return json_graph


def compile_pytorch_for_buda(torchmod, consteval_in_pybuda, *inputs, allow_unsupported):
    torchmod.eval()
    framework_outputs = torchmod(*inputs)
    if not isinstance(framework_outputs, (list, tuple)):
        if isinstance(framework_outputs, torch.Tensor):
            framework_outputs = [framework_outputs]
        elif isinstance(framework_outputs, OrderedDict):
            framework_outputs = framework_outputs.to_tuple()
        else:
            assert False, "Don't know what to do with this"

    framework_outputs = [x.detach().numpy() for x in framework_outputs]
    traced_model = torch.jit.trace(torchmod, inputs, strict=False)
    input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
    mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_list)
    if consteval_in_pybuda:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    np_inputs = [x.detach().numpy() for x in inputs]
    _, buda_params = compile_tvm_for_buda(mod, params, np_inputs, framework_outputs, return_params=True, allow_unsupported=allow_unsupported)
    json_graph["params"] = {name:v.numpy() for (k, v), name in zip(buda_params.items(), json_graph["param_names"])}

    return copy.deepcopy(clean_names(json_graph=json_graph, buda_params=buda_params))


def compile_tvm_for_buda(mod, params, inputs, golden_outputs, return_params=False, allow_unsupported=False):
    target = "llvm"
    mod, params = tvm.relay.op.contrib.compile_for_buda(mod, target=target, params=params)

    relay_outputs = relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm", params=params).evaluate()(*inputs)
    if not isinstance(relay_outputs, (list, tuple)):
        relay_outputs = [relay_outputs]
    relay_outputs = [x.numpy() for x in relay_outputs]

    # Verify compile passes (original relay passes + buda passes)
    verify_tvm_compile(golden_outputs, relay_outputs)

    # Reconstruct Ops + export buda graph
    mod = tvm.relay.op.contrib.buda.reconstruct_ops_for_buda(mod)
    mod, buda_params = tvm.relay.op.contrib.buda.partition_for_buda(mod, allow_unsupported=allow_unsupported)

    executor_factory = tvm.relay.build_module.build(mod, target=target, params=params)

    with tvm.transform.PassContext(opt_level=5):
        func = relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm").evaluate()

    if return_params:
        return func, buda_params
    else:
        return func


def verify_tvm_compile(framework_outputs, relay_outputs, rtol=1e-02, atol=1e-04, pcc=None):
    allowed_to_fail = False
    if len(framework_outputs) != len(relay_outputs):
        logger.error(f"Different number of outputs. Framework: {len(framework_outputs)}, TVM: {len(relay_outputs)}")

    for i, (fr_out, tvm_out) in enumerate(zip(framework_outputs, relay_outputs)):

        if pcc is None:
            ok = np.allclose(fr_out, tvm_out, rtol=rtol, atol=atol, equal_nan=True)
        else:
            pcc_value = np.min(np.ma.corrcoef(
                    np.ma.masked_invalid(fr_out.flatten()),
                    np.ma.masked_invalid(tvm_out.flatten())
                    ))
            ok = pcc_value >= pcc

        if not ok:
            logger.error(f"Tensor mismatch on output {i} between framework and TVM.")
            logger.trace(f"Framework: (shape = {fr_out.shape}")
            logger.trace(fr_out)
            logger.trace(f"TVM: (shape = {tvm_out.shape}")
            logger.trace(tvm_out)
            if not allowed_to_fail:
                raise RuntimeError

    logger.info(f"Verified TVM Relay outputs against framework outputs")


def clean_names(json_graph, buda_params):
    if len(json_graph["param_names"]) == 0:
        return json_graph

    clean_names = []

    precursor = "tvmgen_default_buda_main_"
    fn_number = int(json_graph["param_names"][0].replace(precursor, "").split("_")[0])
    precursor = f"tvmgen_default_buda_main_{fn_number}_"
    for idx, name in enumerate(json_graph["param_names"]):
        if precursor in name:
            clean_names.append(str(json_graph["param_names"][idx]).replace(precursor, ""))

    json_graph["params"] = {name:v.numpy() for (k, v), name in zip(buda_params.items(), clean_names)}
    graph = json.loads(json_graph["graph"])

    for node in graph["nodes"]:
        if precursor in node["name"]:
            node["name"] = node["name"].replace(precursor, "")

    json_graph["graph"] = json.dumps(graph)

    return json_graph


def compile_tf_for_buda(tfmod, *inputs, allow_unsupported):
    framework_outputs = tfmod(*inputs)
    if not isinstance(framework_outputs, (list, tuple)):
        framework_outputs = [framework_outputs]

    framework_outputs = [x.numpy() for x in framework_outputs]

    @tf.function
    def trace(*inputs):
        return tfmod(*inputs, training=False)

    full_model = trace.get_concrete_function(*inputs)

    frozen_func = convert_variables_to_constants_v2(full_model)
    graph_def = frozen_func.graph.as_graph_def()
    mod, params = tvm.relay.frontend.from_tensorflow(graph_def)
    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    np_inputs = [x.numpy() for x in inputs if x is not None]
    _, buda_params = compile_tvm_for_buda(mod, params, np_inputs,framework_outputs, return_params=True, allow_unsupported=allow_unsupported)
    json_graph["params"] = {name:v.numpy() for (k, v), name in zip(buda_params.items(), json_graph["param_names"])}

    return copy.deepcopy(clean_names(json_graph=json_graph, buda_params=buda_params))


def load_serilized_tvm_graph(full_file_path):
    """
    Loads serilized TVM graph representation ported to PyBuda in form of python dictionary.

    Parameters
    ----------
    full_file_path: String
        Full source path where serilized TVM graph is stored
        
    Returns
    -------
    Dictionary
        Deserialized TVM graph
    """
    if not file_exists(full_file_path):
        raise RuntimeError(f"Serilized TVM model doesn't exist: {full_file_path}")

    with open(full_file_path, "r") as file:
        serilized_graph_str = json.load(file)

    serilized_dict = {}
    serilized_dict["graph"] = json.dumps(serilized_graph_str["graph"])
    serilized_dict["params"] = serilized_graph_str["params"]

    logger.debug(f"Successfully load serilized TVM graph from {full_file_path} path")

    return serilized_dict


def format_tvm_graph_weights(inputs, torchmod, compiler_cfg):
    """
    Formats model weights based on specific framework.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors

    torchmod: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device.

    compiler_cfg: CompilerConfig
        Compiler configurations
        
    Returns
    -------
    OrderedDict, Boolean, Tuple
        Weights, Constant evaluation, Input tensors
    """
    if isinstance(torchmod, pybuda.module.PyTorchModule):
        weights = torchmod.module.state_dict()
    elif isinstance(torchmod, pybuda.module.TFModule):
        weights = {weight.name: weight.value for weight in torchmod.module.weights}
        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(x.numpy()) for x in inputs if x is not None]  # Maybe we can switch all tensors to numpy?
        compiler_cfg.enable_consteval = False
    else:
        raise RuntimeError(f"Unsupported module type {type(torchmod)}")

    return inputs, weights


def serilize_and_store_tvm_graph(json_graph, compiler_cfg):
    """
    Serilizes TVM graph representation ported to PyBuda in form of JSON and stores it 
    on the desired destination.

    Parameters
    ----------
    json_graph: Dictionary
        Previously compiled TVM graph pored to PyBuda representation

    compiler_cfg: CompilerConfig
        Compiler configurations

    Returns
    -------
    """
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    if compiler_cfg.tvm_graph_store_path == "" or compiler_cfg.tvm_graph_load_path != "" or not compiler_cfg.enable_consteval:
        return

    graph_dict = json.loads(json_graph["graph"])
    params_dict = json_graph["params"]

    serilized_dict = {}
    serilized_dict["graph"] = graph_dict
    serilized_dict["params"] = params_dict
    serilized_str = json.dumps(serilized_dict, cls=NumpyArrayEncoder, indent=2)
    
    with open(compiler_cfg.tvm_graph_store_path, 'w') as file:
        file.write(serilized_str)

    logger.debug(f"Successfully serilized TVM graph from {compiler_cfg.tvm_graph_store_path} path")


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
