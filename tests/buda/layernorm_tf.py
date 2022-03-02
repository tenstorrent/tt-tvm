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


    data = tf.convert_to_tensor(np.random.rand(1, 32, 128) , dtype=tf.float32)
    data_2 = tf.identity(data)
    # if epsilon > 1.001e-5, TF will fuse into batchnorm
    layer = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)

    @tf.function
    def test(data):
        return layer(inputs=data, training=False)

    full_model = test.get_concrete_function(
        tf.TensorSpec(data.shape, data.dtype)
    )

    # Get frozen graph def
    frozen_func = convert_variables_to_constants_v2(full_model)
    graph_def = frozen_func.graph.as_graph_def()

    mod, params = relay.frontend.from_tensorflow(graph_def)

    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    func = compile_tvm_for_buda(mod, params)

    res = func(data)

    if isinstance(res, (list, tuple)):
        res = res[0]
    res = res.numpy()

    res_tf = layer(data_2)
    if isinstance(res_tf, (list, tuple)):
        res_tf = res_tf[0]

    res_tf = res_tf.numpy()

    print(f"Results correct: {np.allclose(res, res_tf, atol=1e-6)}")


if __name__ == "__main__":
    run_test()