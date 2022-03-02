from pybuda.module import PyBudaModule
import torch
import torch.nn as nn
import numpy as np
import tvm
import tvm.relay as relay

from pybuda_runtime import compile_tvm_for_buda
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def run_test():


    x = tf.constant([5.0, 4.0, 1.0, 3.0], dtype=tf.float32)
    x_tf = tf.constant([5.0, 4.0, 1.0, 3.0], dtype=tf.float32)
    @tf.function
    def test(inputs):
        return tf.pow(inputs, -0.5)

    full_model = test.get_concrete_function(x)
    # Get frozen graph def
    frozen_func = convert_variables_to_constants_v2(full_model)
    graph_def = frozen_func.graph.as_graph_def()

    mod, params = tvm.relay.frontend.from_tensorflow(graph_def)
    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    func = compile_tvm_for_buda(mod, params)

    res = func(x)

    if isinstance(res, (list, tuple)):
        res = res[0]
    res = res.numpy()

    res_tf = tf.pow(x_tf, -0.5)
    if isinstance(res_tf, (list, tuple)):
        res_tf = res_tf[0]

    res_tf = res_tf.numpy()

    print(f"Results correct: {np.allclose(res, res_tf, atol=1e-6)}")


if __name__ == "__main__":
    run_test()