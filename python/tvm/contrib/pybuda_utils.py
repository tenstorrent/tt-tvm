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

    else:
        raise RuntimeError("Unsupported framework type: {}".format(framework))

    return flattened_inputs, flattened_input_names, flattened_name_map, input_structure
