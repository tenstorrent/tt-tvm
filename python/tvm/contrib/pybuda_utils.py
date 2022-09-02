from collections import OrderedDict, MutableMapping

import torch
import numpy as np
import tensorflow as tf
from transformers.utils.generic import ModelOutput as HFModelOutput

import tvm
from pybuda.config import CompilerConfig
from pybuda.tvm_utils import flatten_inputs


def extract_framework_model_outputs(framework: str, model, inputs, compiler_cfg: CompilerConfig):
    framework_outputs = []

    if compiler_cfg is None or not compiler_cfg.varify_tvm_compile:
        return framework_outputs

    if framework == "pytorch":
        assert model.training == False

        framework_outputs = model(*inputs)
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

        framework_outputs = (
            act.float() if torch.is_floating_point(act) else act for act in framework_outputs
        )
        framework_outputs = [x.detach().numpy() for x in framework_outputs]

    elif framework == "tensorflow":
        framework_outputs = model(*inputs)
        # TODO (aknezevic); ref sha: 1fe78625c809e6ca887a8da5fdde44836830f990
        # Figure out how to sort dictionary outputs:
        #
        # if isinstance(framework_outputs, dict):
        #     framework_outputs = list(framework_outputs.values())
        if not isinstance(framework_outputs, (list, tuple)):
            framework_outputs = [framework_outputs]

        supported_outputs = (tf.Tensor, torch.Tensor)
        framework_outputs = [
            x.numpy() for x in framework_outputs if isinstance(x, supported_outputs)
        ]

    elif framework == "jax":
        framework_outputs = model(*inputs)
        if isinstance(framework_outputs, HFModelOutput):
            framework_outputs = list(framework_outputs.values())

        if not isinstance(framework_outputs, (list, tuple)):
            framework_outputs = [framework_outputs]

        supported_outputs = (tf.Tensor, torch.Tensor)
        framework_outputs = [
            x.numpy() for x in framework_outputs if isinstance(x, supported_outputs)
        ]
    else:
        raise RuntimeError("Unsupported framework type: {}".format(framework))

    return framework_outputs


def extract_flatten_inputs(framework: str, model, inputs):
    if framework == "pytorch":
        input_structure = []
        input_names = [i.debugName().split(".")[0] for i in list(model.graph.inputs())[1:]]

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

        flattened_inputs, flattened_input_names, flattened_name_map = flatten_inputs(
            inputs, input_names
        )
        flattened_inputs = [
            inp.float() if torch.is_floating_point(inp) else inp for inp in flattened_inputs
        ]

    elif framework == "tensorflow":
        # The tensorflow trace automatically flattens inputs
        flattened_inputs, _, _ = flatten_inputs(inputs)
        flattened_input_names = [tensor.name.split(":")[0] for tensor in model.inputs]
        flattened_name_map = None
        input_structure = None

    elif framework == "jax":
        # The tensorflow trace automatically flattens inputs
        flattened_inputs, _, _ = flatten_inputs(inputs)
        flattened_input_names = [tensor.name.split(":")[0] for tensor in model.inputs]
        flattened_name_map = None
        input_structure = None

    else:
        raise RuntimeError("Unsupported framework type: {}".format(framework))

    return flattened_inputs, flattened_input_names, flattened_name_map, input_structure


def construct_tvm_ir(framework: str, model, tvm_mod, params, compiler_cfg: CompilerConfig):
    if framework == "pytorch":
        param_name_lookup = {}

        if not compiler_cfg.enable_tvm_constant_prop:
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], {})
            )
        else:
            if len(compiler_cfg.tvm_constnat_prop_mask):
                propped_params = {
                    k: (v, True)
                    for k, v, in params.items()
                    if any([mask in k for mask in compiler_cfg.tvm_constnat_prop_mask])
                }
            else:
                propped_params = {k: (v, True) for k, v, in params.items()}
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], propped_params)
            )

    elif framework == "tensorflow":
        # TODO: Destupidify this! (Maybe we can sort by a substring of the weight names to make this more efficient)
        found_weights = []
        param_name_lookup = {}
        non_weight_params = {}  # Some parameters (like causal mask) are not weights

        for (bad_name, value) in params.items():
            weight_found = False
            for tf_weight in model.weights:
                if (
                    np.array_equal(tf_weight.value().numpy(), value.numpy())
                    and tf_weight.name not in found_weights
                ):
                    param_name_lookup[bad_name] = tf_weight.name
                    weight_found = True
                    found_weights.append(tf_weight.name)
                    break
            if not weight_found:
                param_name_lookup[bad_name] = bad_name
                non_weight_params[bad_name] = (value, False)

        if not compiler_cfg.enable_tvm_constant_prop:
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], non_weight_params)
            )
        else:
            if len(compiler_cfg.tvm_constnat_prop_mask):
                propped_params = {
                    k: (v, True)
                    for k, v, in params.items()
                    if any(
                        [
                            mask in param_name_lookup[k]
                            for mask in compiler_cfg.tvm_constnat_prop_mask
                        ]
                    )
                }
                propped_params.update(non_weight_params)
            else:
                propped_params = {k: (v, True) for k, v in params.items()}
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], propped_params)
            )

    elif framework == "jax":

        def flatten(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, MutableMapping):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # TODO: Destupidify this! (Maybe we can sort by a substring of the weight names to make this more efficient)
        found_weights = []
        param_name_lookup = {}
        non_weight_params = {}  # Some parameters (like causal mask) are not weights

        try:
            model_params = model.params
        except AttributeError as ex:
            model_params = model.variables['params']._dict

        model_params = flatten(model_params)
        for (bad_name, value) in params.items():
            weight_found = False
            for name, jax_value in model_params.items():
                if name not in found_weights and np.array_equal(jax_value.to_py(), value.numpy()):
                    param_name_lookup[bad_name] = name
                    weight_found = True
                    found_weights.append(name)
                    break
            if not weight_found:
                param_name_lookup[bad_name] = bad_name
                non_weight_params[bad_name] = value

        if not compiler_cfg.enable_tvm_constant_prop:
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], non_weight_params)
            )
        else:
            if len(compiler_cfg.tvm_constnat_prop_mask):
                propped_params = {
                    k: v
                    for k, v, in params.items()
                    if any(
                        [
                            mask in param_name_lookup[k]
                            for mask in compiler_cfg.tvm_constnat_prop_mask
                        ]
                    )
                }
                propped_params.update(non_weight_params)
            else:
                propped_params = params
            tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], propped_params)
            )
    else:
        raise RuntimeError("Unsupported framework type: {}".format(framework))

    return tvm_mod, param_name_lookup
