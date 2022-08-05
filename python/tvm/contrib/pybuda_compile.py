from pybuda.tensor import to_tf_tensors
from pybuda.tvm_utils import flatten_inputs
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
import onnxruntime as ort
import onnx
import onnx.numpy_helper
import mxnet as mx

#TODO: Extend to allow multiple cpu/tt graphs
dev_json_graph = {"function_names": [], "graph" : "", "param_names": {}, "device" : "tt"}
cpu_json_graph = {"function_names": [], "graph" : "", "param_names": {}, "device" : "cpu"}

def retrieve_graph(json_graph, t):
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

@tvm.register_func
def retrieve_pybuda_json_graph(*args):
    t = tuple(args)
    global dev_json_graph
    retrieve_graph(dev_json_graph, t)

@tvm.register_func
def retrieve_pybuda_cpudevice_json_graph(*args):
    t = tuple(args)
    global cpu_json_graph
    retrieve_graph(cpu_json_graph, t)


def load_tvm_graph(inputs, module, compiler_cfg, graph_name, output_names=None, path=None):
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

    output_names: List[str]
        Model output names to extract from Tensorflow -> TVM conversion. If None, the last node will be treated
        as model output node.

    path: str
        Path to onnx file on disk. This is used to verify TVM results vs. framework results.

    Returns
    -------
    Dictionary, OrderedDict, Tuple, Boolean
        TVM ported graph, Weights, Input tensors, Constant evaluation
    """
    if compiler_cfg.tvm_graph_store_path != "" and compiler_cfg.tvm_graph_load_path != "":
        logger.warning(f"TVM serialization logic will be skipped as both store and load paths are provided")

    json_graphs, flattened_inputs = compile_tvm_graph(inputs, module, compiler_cfg, graph_name=graph_name, output_names=output_names, path=path)
    
    flattened_pytorch_inputs, weights = format_tvm_graph_weights(flattened_inputs, module, compiler_cfg)

    serialize_and_store_tvm_graph(dev_json_graph, compiler_cfg)

    return json_graphs, flattened_pytorch_inputs, weights


def compile_tvm_graph(inputs, module, compiler_cfg, graph_name, output_names=None, path=None):
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

    output_names: List[str]
        Model output names to extract from Tensorflow -> TVM conversion. If None, the last node will be treated
        as model output node.

    path: str
        Path to onnx file on disk. This is used to verify TVM results vs. framework results.

    Returns
    -------
    Dictionary
        TVM ported graph
    """
    # clear out graphs before starting in case python session is ongoing
    global dev_json_graph
    global cpu_json_graph
    dev_json_graph = {"function_names": [], "graph" : "", "param_names": {}, "device" : "tt"}
    cpu_json_graph = {"function_names": [], "graph" : "", "param_names": {}, "device" : "cpu"}

    if compiler_cfg.tvm_graph_load_path != "" and compiler_cfg.tvm_graph_store_path == "" and compiler_cfg.enable_consteval:
        json_graphs = load_serialized_tvm_graph(compiler_cfg.tvm_graph_load_path)
    elif isinstance(module, torch.nn.Module):
        json_graphs, inputs = compile_pytorch_for_buda(module, *inputs, graph_name=graph_name, compiler_cfg=compiler_cfg)
    elif isinstance(module, tf.keras.Model):
        # convert pytorch tensors to tf tensors
        tf_inputs = to_tf_tensors(inputs, force_float32=True)
        json_graphs, inputs = compile_tf_for_buda(module, *tf_inputs, graph_name=graph_name, compiler_cfg=compiler_cfg)
    elif isinstance(module, tf.compat.v1.GraphDef):
        if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            tf_inputs = tuple(None if t is None else tf.convert_to_tensor(t.detach().numpy()) for t in inputs)
        else:
            tf_inputs = inputs
        json_graphs = compile_tf_graphdef_for_buda(module, *tf_inputs, graph_name=graph_name, compiler_cfg=compiler_cfg, output_names=output_names)
    elif isinstance(module, onnx.onnx_ml_pb2.ModelProto):
        assert all([isinstance(x, torch.Tensor) for x in inputs])
        onnx_inputs = [x.detach().numpy() for x in inputs]
        json_graphs = compile_onnx_for_buda(module, path, *onnx_inputs, graph_name=graph_name, compiler_cfg=compiler_cfg)
    elif isinstance(module, mx.gluon.HybridBlock):
        assert all([isinstance(x, torch.Tensor) for x in inputs])
        mxnet_inputs = [mx.nd.array(x.detach().numpy()) for x in inputs]
        json_graphs = compile_mxnet_for_buda(module, *mxnet_inputs, graph_name=graph_name, compiler_cfg=compiler_cfg)
    else:
        raise RuntimeError(f"Unsupported module type {type(module)}")

    return json_graphs, inputs

def save_nid_to_input_idx(traced_model_inputs, json_graph):
    existing_graph = json.loads(json_graph["graph"])

    nid_to_input_idx = {}

    # reorder arg nodes to be inputs first, then parameters
    for i, arg_idx in enumerate(existing_graph["arg_nodes"]):
        name = existing_graph["nodes"][arg_idx]["name"]
        if name not in traced_model_inputs:
            continue

        nid_to_input_idx[arg_idx] = traced_model_inputs.index(name)

    json_graph["nid_to_input_idx"] = nid_to_input_idx


def compile_pytorch_for_buda(torchmod, *inputs, graph_name, compiler_cfg):
    training_mode = torchmod.training
    if training_mode:
        torchmod.eval()

    framework_outputs = []
    if compiler_cfg is not None and compiler_cfg.varify_tvm_compile:
        if training_mode:
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

        framework_outputs = (act.float() if torch.is_floating_point(act) else act for act in framework_outputs)
        framework_outputs = [x.detach().numpy() for x in framework_outputs]

    traced_model = torch.jit.trace(torchmod, inputs, strict=False)
    # ensures unique names for every input
    input_structure = []
    input_names = [i.debugName().split('.')[0] for i in list(traced_model.graph.inputs())[1:]]#list(torchmod.forward.__code__.co_varnames[1:][:len(inputs)])
    input_count = 0

    if isinstance(inputs, (list, tuple)):
        
        for i in range(len(inputs)):
            input = inputs[i]
            if isinstance(input, (list, tuple)):
                structure = (input_names[i], tuple([tuple(t.shape) for t in input]))
            elif isinstance(input, dict):
                structure = (input_names[i], {k: v.shape for k, v in input.items()})
            else:
                structure = (input_names[i], tuple(input.shape))
            input_structure.append(tuple(structure))
    else:
        input_structure = OrderedDict()
        for k, v in inputs.items():
            input_structure[k] = v.shape


    mod, params = tvm.relay.frontend.from_pytorch(traced_model, input_structure)
    flattened_inputs, flattened_input_names, flattened_name_map = flatten_inputs(inputs, input_names)
    flattened_inputs = [inp.float() if torch.is_floating_point(inp) else inp for inp in flattened_inputs]
    mod = tvm.relay.op.contrib.flatten_inputs(mod, flattened_inputs, flattened_name_map)
    
    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        if len(compiler_cfg.tvm_constnat_prop_mask):
            propped_params = {k : v for k, v, in params.items() if any([mask in k for mask in compiler_cfg.tvm_constnat_prop_mask])}
        else:
            propped_params = params
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], propped_params))

    flattened_inputs_as_float = (act.float() if torch.is_floating_point(act) else act for act in flattened_inputs)
    np_inputs = {name:inp.detach().numpy() for name, inp in zip(flattened_input_names, flattened_inputs_as_float)}

    partitioned_mod, buda_params = compile_tvm_for_buda(mod, params, np_inputs, framework_outputs, graph_name=graph_name, return_params=True, compiler_cfg=compiler_cfg)

    if training_mode:
        torchmod.train()

    dev_json_graph["params"] = {}
    cpu_json_graph["params"] = {}
    for function_name in buda_params.keys():
        if function_name in dev_json_graph["param_names"]:
            dev_json_graph["params"].update({name:v.numpy() for (k, v), name in zip(buda_params[function_name].items(), dev_json_graph["param_names"][function_name])})
        if function_name in cpu_json_graph["param_names"]:
            cpu_json_graph["params"].update({name:v.numpy() for (k, v), name in zip(buda_params[function_name].items(), cpu_json_graph["param_names"][function_name])})


    json_graphs = []

    # TODO: Extract pipeline info from partitioned_mod
    if cpu_json_graph["graph"] != "":
        save_nid_to_input_idx(flattened_input_names, cpu_json_graph) # Input order might not be preserved by TVM
        cpu_json_graph["num_pybuda_inputs"] = len(flattened_inputs)
        json_graphs.append(copy.deepcopy(clean_names(json_graph=cpu_json_graph, buda_params=buda_params)))
    else:
        save_nid_to_input_idx(flattened_input_names, dev_json_graph) # Input order might not be preserved by TVM
        dev_json_graph["num_pybuda_inputs"] = len(flattened_inputs)

    json_graphs.append(copy.deepcopy(clean_names(json_graph=dev_json_graph, buda_params=buda_params)))

    return json_graphs, flattened_inputs


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
    relay_outputs = [x.numpy() for x in flattened]

    return relay_outputs


def compile_tvm_for_buda(mod, params, inputs, golden_outputs, graph_name, return_params=False, compiler_cfg=None):
    target = "llvm"
    mod, params = tvm.relay.op.contrib.compile_for_buda(mod, target=target, params=params, graph_name=graph_name)

    if compiler_cfg is not None and compiler_cfg.varify_tvm_compile:
        relay_outputs = get_relay_output(mod, params, inputs, target)

        # Verify compile passes (original relay passes + buda passes)
        verify_tvm_compile(golden_outputs, relay_outputs)

    # Reconstruct Ops + export buda graph
    mod = tvm.relay.op.contrib.buda.reconstruct_ops_for_buda(mod)
    mod, buda_params = tvm.relay.op.contrib.buda.partition_for_buda(mod, graph_name=graph_name, compiler_cfg=compiler_cfg)
    tvm.relay.build_module.build(mod, target=target, params=params)

    if return_params:
        return mod, buda_params
    else:
        return mod


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
    precursor = "tvmgen_default_pybuda_main_" if json_graph["device"] != "cpu" else "tvmgen_default_pybuda_cpudevice_main_"
    if len(json_graph["params"]) > 0:

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

def compile_onnx_for_buda(onnx_mod, path, *inputs, graph_name, compiler_cfg):
    input_names = []
    for inp in onnx_mod.graph.input:
        input_names.append(inp.name)
    
    assert len(input_names) == len(inputs), "Number of input names must match number of inputs"
    output_names = []
    for out in onnx_mod.graph.output:
        output_names.append(out.name)

    input_dict = {}
    input_shape_dict = {}
    for name, tensor in zip(input_names, inputs):
        input_dict[name] = tensor
        input_shape_dict[name] = tensor.shape

    framework_outputs = []
    if compiler_cfg is not None and compiler_cfg.varify_tvm_compile:
        assert path != None, "Onnx compile needs path to onnx file on disk."
        ort_sess = ort.InferenceSession(path)
        framework_outputs = ort_sess.run(output_names, input_dict)

    mod, params = relay.frontend.from_onnx(onnx_mod, input_shape_dict)

    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    _, buda_params = compile_tvm_for_buda(mod, params, input_dict, framework_outputs, graph_name=graph_name, return_params=True, compiler_cfg=compiler_cfg)

    dev_json_graph["params"] = {}
    for function_name in buda_params.keys():
        dev_json_graph["params"].update({name:v.numpy() for (k, v), name in zip(buda_params[function_name].items(), dev_json_graph["param_names"][function_name])})

    json_graphs = []
    json_graphs.append(copy.deepcopy(clean_names(json_graph=dev_json_graph, buda_params=buda_params)))
    return json_graphs

def compile_tf_for_buda(tfmod, *inputs, graph_name, compiler_cfg):
    framework_outputs = []
    if compiler_cfg is not None and compiler_cfg.varify_tvm_compile:
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

    # The tensorflow trace automatically flattens inputs
    flattened_input_names = [tensor.name.split(':')[0] for tensor in frozen_func.inputs]

    mod, params = tvm.relay.frontend.from_tensorflow(graph_def, layout="NCHW", outputs=outputs)
    mod = tvm.transform.Sequential([tvm.relay.transform.Inline()])(mod)
    flattened_inputs, _, _ = flatten_inputs(inputs)


    # TODO: Destupidify this! (Maybe we can sort by a substring of the weight names to make this more efficient)
    found_weights = []
    param_name_lookup = {}
    non_weight_params = {} # Some parameters (like causal mask) are not weights

    for (bad_name, value) in params.items():
        weight_found = False
        for tf_weight in tfmod.weights:
            if np.array_equal(tf_weight.value().numpy(), value.numpy()) and tf_weight.name not in found_weights:
                param_name_lookup[bad_name] = tf_weight.name
                weight_found = True
                found_weights.append(tf_weight.name)
                break
        if not weight_found:
            param_name_lookup[bad_name] = bad_name
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

    np_inputs = {i : None if x is None else x.numpy() for i, x in  zip(flattened_input_names, flattened_inputs)}
    _, buda_params = compile_tvm_for_buda(mod, params, np_inputs, framework_outputs, graph_name=graph_name, return_params=True, compiler_cfg=compiler_cfg)
    
    dev_json_graph["params"] = {}
    for function_name in buda_params.keys():
        dev_json_graph["params"].update({name:v.numpy() for (k, v), name in zip(buda_params[function_name].items(), dev_json_graph["param_names"][function_name])})

    traced_model_inputs = [i.name.split(':')[0] for i in frozen_func.inputs]
    save_nid_to_input_idx(traced_model_inputs, json_graph = dev_json_graph)
    json_graphs = []
    json_graphs.append(copy.deepcopy(clean_names(json_graph=dev_json_graph, buda_params=buda_params, param_name_lookup=param_name_lookup)))
    return json_graphs, flattened_inputs

# TODO (arui) : Verify graphdef output vs. TVM output
def compile_tf_graphdef_for_buda(graph_def, *inputs, graph_name, compiler_cfg, output_names=None):
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

    assert compiler_cfg.enable_tvm_constant_prop == True, "Pybuda Compile only support tf graphdef model with TVM parameter binding."
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
    mod, buda_params = tvm.relay.op.contrib.buda.partition_for_buda(mod, graph_name=graph_name, compiler_cfg=compiler_cfg)

    executor_factory = tvm.relay.build_module.build(mod, target=target, params=params)

    with tvm.transform.PassContext(opt_level=5):
        func = relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm").evaluate()


    dev_json_graph["params"] = {}
    for function_name in buda_params.keys():
        dev_json_graph["params"].update({name:v.numpy() for (k, v), name in zip(buda_params[function_name].items(), dev_json_graph["param_names"][function_name])})

    json_graphs = []
    json_graphs.append(copy.deepcopy(clean_names(json_graph=dev_json_graph, buda_params=buda_params)))
    return json_graphs


def compile_mxnet_for_buda(module, *inputs, graph_name, compiler_cfg):
    framework_outputs = []
    if compiler_cfg is not None and compiler_cfg.varify_tvm_compile:
        framework_outputs = module(*inputs)
        if not isinstance(framework_outputs, (list, tuple)):
            framework_outputs = [framework_outputs]

        framework_outputs = [x.asnumpy() for x in framework_outputs if isinstance(x, mx.ndarray.ndarray.NDArray)]

    input_dict = {
        "input_" + str(i) : inp.shape
        for i, inp in enumerate(inputs)
    }
    input_name_to_tensor = {name : tensor.asnumpy() for name, tensor in zip(input_dict.keys(), inputs)}
    mod, params = relay.frontend.from_mxnet(module, shape=input_dict)

    if not compiler_cfg.enable_tvm_constant_prop:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], {}))
    else:
        mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    _, buda_params = compile_tvm_for_buda(mod, params, input_name_to_tensor, framework_outputs, graph_name=graph_name, return_params=True, compiler_cfg=compiler_cfg)

    dev_json_graph["params"] = {}
    for function_name in buda_params.keys():
        dev_json_graph["params"].update({name:v.numpy() for (k, v), name in zip(buda_params[function_name].items(), dev_json_graph["param_names"][function_name])})

    json_graph = []
    json_graph.append(copy.deepcopy(clean_names(json_graph=dev_json_graph, buda_params=buda_params)))
    return json_graph


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
        serialized_dict["nid_to_input_idx"] = {int(k) : v for k, v in serialized_graph_str["nid_to_input_idx"].items()}
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
        named_buffers = dict(module.named_buffers())
        torch_weights.update(named_buffers)
        named_params = dict(module.named_parameters())
        weights = {key: (value, named_params[key].requires_grad if key in named_params else False) for key, value in torch_weights.items()}
    elif isinstance(module, tf.keras.Model):
        weights = {weight.name: (torch.Tensor((tf.cast(weight.value(), tf.float32) if weight.value().dtype.is_floating else weight.value()).numpy()), True) for weight in module.weights}
        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(x.numpy()) for x in inputs if x is not None]  # Maybe we can switch all tensors to numpy?
    elif isinstance(module, tf.compat.v1.GraphDef):
        weights = {}
    elif isinstance(module, onnx.onnx_ml_pb2.ModelProto):
        numpy_weights = [onnx.numpy_helper.to_array(weight) for weight in module.graph.initializer]
        names = [weight.name for weight in module.graph.initializer]
        weights = {
            name : (torch.Tensor(weight), issubclass(weight.dtype.type, np.floating))
            for name, weight in zip(names, numpy_weights)
        }

        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(onnx.numpy_helper.to_array(x)) for x in inputs if x is not None]

    elif isinstance(module, mx.gluon.HybridBlock):
        weights = {
            name : (torch.Tensor(mx_param.data().asnumpy()), issubclass(mx_param.data().asnumpy().dtype.type, np.floating))
            for name, mx_param in module.collect_params().items()
        }
        if not (len(inputs) > 0 and isinstance(inputs[0], torch.Tensor)):
            inputs = [torch.tensor(x.asnumpy()) for x in inputs if x is not None]
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

