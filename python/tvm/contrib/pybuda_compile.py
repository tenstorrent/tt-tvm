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


json_graph = {"function_names": [], "graph" : "", "param_names": {}}
@tvm.register_func
def retrieve_json_graph(*args):
    t = tuple(args)
    global json_graph
    function_name = t[0]
    if function_name in json_graph["function_names"]:
        return
    json_graph["function_names"].append(function_name)

    if len(json_graph["graph"]) > 0:
        graph = json.loads(t[1])
        existing_graph = json.loads(json_graph["graph"])
        
        num_new_nodes = len(graph["nodes"])
        for node in existing_graph["nodes"]:
            if "num_inputs" not in node["attrs"]:
                continue

            node["inputs"] = [[input_nid[0] + num_new_nodes, 0, 0] for input_nid in node["inputs"]]
            
        existing_graph["heads"] = [[head[0] + num_new_nodes, 0, 0] for head in existing_graph["heads"]]
        existing_graph["arg_nodes"] = [arg_node + num_new_nodes for arg_node in existing_graph["arg_nodes"]]
        num_node_rows = len(graph["node_row_ptr"])
        existing_graph["node_row_ptr"] = [node_row_ptr + num_node_rows for node_row_ptr in existing_graph["node_row_ptr"]]

        for key in graph.keys():
            graph[key] =  graph[key] + existing_graph[key]

        json_graph["graph"] = json.dumps(graph)
    else:
        json_graph["graph"] = t[1]
    
    json_graph["param_names"][function_name] = t[2]


def load_tvm_graph(inputs, module, compiler_cfg, graph_name, allow_unsupported=False, output_names=None):
    """
    Loads TVM graph ported to the PyBuda from other frameworks (TensorFlow, Pytorch). Can eather
    run whole compilation process from specific framework to the PyBuda graph representation, or
    skip this compilation process and load serialize TVM graph which is already ported to PyBuda
    by initial invocation.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors

    module: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device.

    compiler_cfg: CompilerConfig
        Compiler configurations
        
    Returns
    -------
    Dictionary, OrderedDict, Tuple, Boolean
        TVM ported graph, Weights, Input tensors, Constant evaluation
    """
    if compiler_cfg.tvm_graph_store_path != "" and compiler_cfg.tvm_graph_load_path != "":
        logger.warning(f"TVM serialization logic will be skipped as both store and load paths are provided")

    json_graph = compile_tvm_graph(inputs, module, compiler_cfg, graph_name=graph_name, allow_unsupported=allow_unsupported, output_names=output_names)
    
    pytorch_inputs, weights = format_tvm_graph_weights(inputs, module, compiler_cfg)

    serialize_and_store_tvm_graph(json_graph, compiler_cfg)

    return json_graph, pytorch_inputs, weights


def compile_tvm_graph(inputs, module, compiler_cfg, graph_name, allow_unsupported, output_names=None):
    """
    Compiles TVM graph ported to the PyBuda from other frameworks (TensorFlow, Pytorch). Can eather
    run whole compilation process or only load serilized TVM graph and thus increase test performance.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors
    
    module: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device

    compiler_cfg: CompilerConfig
        Compiler configurations

    Returns
    -------
    Dictionary
        TVM ported graph
    """
    global json_graph
    json_graph = {"function_names": [], "graph" : "", "param_names": {}}
    if compiler_cfg.tvm_graph_load_path != "" and compiler_cfg.tvm_graph_store_path == "" and compiler_cfg.enable_consteval:
        json_graph = load_serialized_tvm_graph(compiler_cfg.tvm_graph_load_path)
        if isinstance(module, torch.nn.Module):
            module.eval()
    elif isinstance(module, torch.nn.Module):
        inputs = (act.float() if torch.is_floating_point(act) else act for act in inputs)
        json_graph = compile_pytorch_for_buda(module, *inputs, graph_name=graph_name, allow_unsupported=allow_unsupported, compiler_cfg=compiler_cfg)
    elif isinstance(module, tf.keras.Model):
        # convert pytorch tensors to tf tensors
        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            tf_inputs = tuple(None if t is None else tf.convert_to_tensor(t.float().detach().numpy() if t.dtype.is_floating_point else t.detach().numpy()) for t in inputs)
        else:
            tf_inputs = inputs

        json_graph = compile_tf_for_buda(module, *tf_inputs, graph_name=graph_name, allow_unsupported=allow_unsupported, compiler_cfg=compiler_cfg)
    elif isinstance(module, tf.compat.v1.GraphDef):
        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            tf_inputs = tuple(None if t is None else tf.convert_to_tensor(t.detach().numpy()) for t in inputs)
        else:
            tf_inputs = inputs
        json_graph = compile_tf_graphdef_for_buda(module, *tf_inputs, graph_name=graph_name, allow_unsupported=allow_unsupported, compiler_cfg=compiler_cfg, output_names=output_names)
    else:
        raise RuntimeError(f"Unsupported module type {type(module)}")

    return json_graph

def save_nid_to_input_idx(traced_model_inputs):
    existing_graph = json.loads(json_graph["graph"])

    nid_to_input_idx = {}

    # reorder arg nodes to be inputs first, then parameters
    for i, arg_idx in enumerate(existing_graph["arg_nodes"]):
        name = existing_graph["nodes"][arg_idx]["name"]
        if name not in traced_model_inputs:
            continue

        nid_to_input_idx[arg_idx] = traced_model_inputs.index(name)

    json_graph["nid_to_input_idx"] = nid_to_input_idx


def compile_pytorch_for_buda(torchmod, *inputs, graph_name, allow_unsupported, compiler_cfg):
    torchmod.eval()
    framework_outputs = torchmod(*inputs)
    if not isinstance(framework_outputs, (list, tuple)):
        if isinstance(framework_outputs, torch.Tensor):
            framework_outputs = [framework_outputs]
        elif isinstance(framework_outputs, OrderedDict):
            framework_outputs = tuple(framework_outputs.values())
        else:
            assert False, "Don't know what to do with this"
    elif any([isinstance(x, (tuple, list)) for x in framework_outputs]):
        output_list = []
        for sublist in framework_outputs:
            if isinstance(sublist, (list, tuple)):
                output_list.extend(sublist)
            else:
                output_list.append(sublist)
        framework_outputs = output_list

    framework_outputs = [x.detach().numpy() for x in framework_outputs]

    convert_dtype = False
    for key, value in torchmod.state_dict().items():
        if value.dtype not in (torch.float32, torch.float64):
            convert_dtype = True

    torchmod = copy.deepcopy(torchmod) if convert_dtype else torchmod
    traced_model = torch.jit.trace(torchmod, inputs, strict=False)
    traced_model = traced_model.float() if convert_dtype else traced_model
    input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
    mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_list)
    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        if len(compiler_cfg.tvm_constnat_prop_mask):
            propped_params = {k : v for k, v, in params.items() if any([mask in k for mask in compiler_cfg.tvm_constnat_prop_mask])}
        else:
            propped_params = params
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    np_inputs = {i.debugName().split('.')[0]:x.detach().numpy() for i, x in  zip(list(traced_model.graph.inputs())[1:], inputs)}
    _, buda_params = compile_tvm_for_buda(mod, params, np_inputs, framework_outputs, graph_name=graph_name, return_params=True, allow_unsupported=allow_unsupported)

    json_graph["params"] = {}
    for function_name in buda_params.keys():
        json_graph["params"].update({name:v.numpy() for (k, v), name in zip(buda_params[function_name].items(), json_graph["param_names"][function_name])})

    traced_model_inputs = [i.debugName().split('.')[0] for i in  list(traced_model.graph.inputs())[1:]]
    save_nid_to_input_idx(traced_model_inputs) # Input order might not be preserved by TVM

    return copy.deepcopy(clean_names(json_graph=json_graph, buda_params=buda_params))


def get_relay_output(mod, params, inputs, target):
    # Build and Run Relay modules with inputs as (key : tensor) pair
    # Then, inputs dont need to be in the same order as 'mod' defines.
    ret_type = mod["main"].checked_type.ret_type
    lib = relay.build(mod, target=target, params=params)
    m = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    m.run(**inputs)

    def _unflatten(flat_iter, cur_type):
        import tvm.relay.ty as _ty
        if isinstance(cur_type, _ty.TensorType):
            return next(flat_iter)
        if isinstance(cur_type, _ty.TupleType):
            fields = []
            for field_type in cur_type.fields:
                field = _unflatten(flat_iter, field_type)
                fields.append(field)
            return fields
        raise ValueError("Return type", ret_type, "contains unsupported type", cur_type)

    flattened = []
    from .. import nd as _nd
    for i in range(m.get_num_outputs()):
        flattened.append(m.get_output(i).copyto(_nd.cpu(0)))
    relay_outputs = _unflatten(iter(flattened), ret_type)

    if not isinstance(relay_outputs, (list, tuple)):
        relay_outputs = [relay_outputs]
    relay_outputs = [x.numpy() for x in relay_outputs]

    return relay_outputs


def compile_tvm_for_buda(mod, params, inputs, golden_outputs, graph_name, return_params=False, allow_unsupported=False):
    target = "llvm"
    mod, params = tvm.relay.op.contrib.compile_for_buda(mod, target=target, params=params, graph_name=graph_name)

    relay_outputs = get_relay_output(mod, params, inputs, target)

    # Verify compile passes (original relay passes + buda passes)
    verify_tvm_compile(golden_outputs, relay_outputs)

    # Reconstruct Ops + export buda graph
    mod = tvm.relay.op.contrib.buda.reconstruct_ops_for_buda(mod)
    mod, buda_params = tvm.relay.op.contrib.buda.partition_for_buda(mod, allow_unsupported=allow_unsupported, graph_name=graph_name)

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


def clean_names(json_graph, buda_params, param_name_lookup=None):
    clean_names = []

    precursor = "tvmgen_default_buda_main_"
    if len(json_graph["params"]) > 0:
        precursor = f"tvmgen_default_buda_main_"

        old_params = json_graph["params"]
        json_graph["params"] = {}
        for k, v in old_params.items():
            key = k.replace(precursor, "")[2:] + k.replace(precursor, "")[0]
            json_graph["params"][key] = v

    graph = json.loads(json_graph["graph"])

    for node in graph["nodes"]:
        if precursor in node["name"]:
            node["name"] = node["name"].replace(precursor, "")[2:] + node["name"].replace(precursor, "")[0]
        elif param_name_lookup is not None and node["name"] in param_name_lookup:
            node["name"] = param_name_lookup[node["name"]]

    json_graph["graph"] = json.dumps(graph)

    return json_graph


def compile_tf_for_buda(tfmod, *inputs, graph_name, compiler_cfg, allow_unsupported):
    framework_outputs = tfmod(*inputs)
    if not isinstance(framework_outputs, (list, tuple)):
        framework_outputs = [framework_outputs]

    supported_outputs = (tf.Tensor, torch.Tensor)
    framework_outputs = [x.numpy() for x in framework_outputs if isinstance(x, supported_outputs)]

    @tf.function
    def trace(*inputs):
        return tfmod(*inputs, training=False)

    full_model = trace.get_concrete_function(*inputs)

    frozen_func = convert_variables_to_constants_v2(full_model)
    graph_def = frozen_func.graph.as_graph_def()
    outputs = [output.name for output in frozen_func.outputs]
    mod, params = tvm.relay.frontend.from_tensorflow(graph_def, outputs=outputs)

    # TODO: Destupidify this! (Maybe we can sort by a substring of the weight names to make this more efficient)
    found_weights = []
    param_name_lookup = {}
    non_weight_params = {} # Some parameters (like causal mask) are not weights

    for (bad_name, value) in params.items():
        weight_found = False
        for tf_weight in tfmod.weights:
            if np.array_equal(tf_weight.value().numpy(), value.numpy()) and tf_weight.name not in found_weights:
                param_name_lookup[bad_name] = tf_weight.name.replace(":", ".")
                weight_found = True
                found_weights.append(tf_weight.name)
                break
        if not weight_found:
            param_name_lookup[bad_name] = bad_name.replace(":", ".")
            non_weight_params[bad_name] = value
    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], non_weight_params))
    else:
        if len(compiler_cfg.tvm_constnat_prop_mask):
            propped_params = {k : v for k, v, in params.items() if any([mask in param_name_lookup[k] for mask in compiler_cfg.tvm_constnat_prop_mask])}
            propped_params.update(non_weight_params)
        else:
            propped_params = params
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    np_inputs = {i.name.split(':')[0] : None if x is None else x.numpy() for i, x in  zip(frozen_func.inputs, inputs)}
    _, buda_params = compile_tvm_for_buda(mod, params, np_inputs, framework_outputs, graph_name=graph_name, return_params=True, allow_unsupported=allow_unsupported)
    
    json_graph["params"] = {}
    for function_name in buda_params.keys():
        json_graph["params"].update({name:v.numpy() for (k, v), name in zip(buda_params[function_name].items(), json_graph["param_names"][function_name])})
    
    return copy.deepcopy(clean_names(json_graph=json_graph, buda_params=buda_params, param_name_lookup=param_name_lookup))

# TODO (arui) : Verify graphdef output vs. TVM output
def compile_tf_graphdef_for_buda(graph_def, *inputs, graph_name, compiler_cfg, allow_unsupported, output_names=None):
    if output_names == None:
        output_list_ = []
    else:
        output_list_ = output_names
    # framework_outputs = tfmod(*inputs)
    # if not isinstance(framework_outputs, (list, tuple)):
    #     framework_outputs = [framework_outputs]

    # supported_outputs = (tf.Tensor, torch.Tensor)
    # framework_outputs = [x.numpy() for x in framework_outputs if isinstance(x, supported_outputs)]

    # @tf.function
    # def trace(*inputs):
    #     return tfmod(*inputs, training=False)

    # full_model = trace.get_concrete_function(*inputs)

    # frozen_func = convert_variables_to_constants_v2(full_model)
    # graph_def = frozen_func.graph.as_graph_def()

    mod, params = tvm.relay.frontend.from_tensorflow(graph_def, outputs=output_list_)

    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        if len(compiler_cfg.tvm_constnat_prop_mask):
            propped_params = {k : v for k, v, in params.items() if any([mask in k for mask in compiler_cfg.tvm_constnat_prop_mask])}
        else:
            propped_params = params
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    target = "llvm"
    mod, params = tvm.relay.op.contrib.compile_for_buda(mod, target=target, params=params, graph_name=graph_name)

    # Reconstruct Ops + export buda graph
    mod = tvm.relay.op.contrib.buda.reconstruct_ops_for_buda(mod)
    mod, buda_params = tvm.relay.op.contrib.buda.partition_for_buda(mod, allow_unsupported=allow_unsupported, graph_name=graph_name)

    executor_factory = tvm.relay.build_module.build(mod, target=target, params=params)

    with tvm.transform.PassContext(opt_level=5):
        func = relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm").evaluate()


    json_graph["params"] = {}
    for function_name in buda_params.keys():
        json_graph["params"].update({name:v.numpy() for (k, v), name in zip(buda_params[function_name].items(), json_graph["param_names"][function_name])})

    return copy.deepcopy(clean_names(json_graph=json_graph, buda_params=buda_params))


def load_serialized_tvm_graph(full_file_path):
    """
    Loads serialized TVM graph representation ported to PyBuda in form of python dictionary.

    Parameters
    ----------
    full_file_path: String
        Full source path where serialized TVM graph is stored
        
    Returns
    -------
    Dictionary
        Deserialized TVM graph
    """
    if not file_exists(full_file_path):
        raise RuntimeError(f"Serialized TVM model doesn't exist: {full_file_path}")

    with open(full_file_path, "r") as file:
        serialized_graph_str = json.load(file)

    serialized_dict = {}
    serialized_dict["graph"] = json.dumps(serialized_graph_str["graph"])
    serialized_dict["params"] = serialized_graph_str["params"]
    if "nid_to_input_idx" in serialized_graph_str.keys():
        serialized_dict["nid_to_input_idx"] = serialized_graph_str["nid_to_input_idx"]
    logger.debug(f"Successfully load serialized TVM graph from {full_file_path} path")

    return serialized_dict


def format_tvm_graph_weights(inputs, module, compiler_cfg):
    """
    Formats model weights based on specific framework.

    Parameters
    ----------
    inputs: Tuple[Tensor, ...]
        Input tensors

    module: Module(PyTorchModule or TFModule)
        Module that contains workload which can be assigned to a single device.

    compiler_cfg: CompilerConfig
        Compiler configurations
        
    Returns
    -------
    OrderedDict, Boolean, Tuple
        Weights, Constant evaluation, Input tensors
    """
    if isinstance(module, torch.nn.Module):
        if compiler_cfg.enable_training:
            for param in module.parameters():
                param.requires_grad = True

        torch_weights = module.state_dict()
        named_params = dict(module.named_parameters())
        weights = {key: (value, named_params[key].requires_grad if key in named_params else False) for key, value in torch_weights.items()}
    elif isinstance(module, tf.keras.Model):
        weights = {weight.name.replace(":", "."): (torch.Tensor((tf.cast(weight.value(), tf.float32) if weight.value().dtype.is_floating else weight.value()).numpy()), True) for weight in module.weights}
        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(x.numpy()) for x in inputs if x is not None]  # Maybe we can switch all tensors to numpy?
    elif isinstance(module, tf.compat.v1.GraphDef):
        weights = {}
    else:
        raise RuntimeError(f"Unsupported module type {type(module)}")

    return inputs, weights


def serialize_and_store_tvm_graph(json_graph, compiler_cfg):
    """
    Serializes TVM graph representation ported to PyBuda in form of JSON and stores it 
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
    if "nid_to_input_idx" in json_graph.keys():
        serilized_dict["nid_to_input_idx"] = json_graph["nid_to_input_idx"]
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
