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

