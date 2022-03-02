
import tvm
import tvm.relay

import numpy as np
import tensorflow as tf
from transformers.models.gpt2.modeling_tf_gpt2 import TFGPT2Model, TFBlock
from transformers import GPT2Config
from pybuda_runtime import compile_tvm_for_buda
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
import tvm.relay.testing.tf as tf_testing
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf


def main():
    config = GPT2Config()
    config.activation_function = "gelu"
    config.use_cache = True

    tf_layer = TFBlock(config.n_ctx, config)

    hidden_states = tf.convert_to_tensor(np.random.rand(1, 32, 768).astype(np.float32))

    trace_inputs = {
        "x" : hidden_states,
        "layer_past" : None, 
        "attention_mask" : None,
        "head_mask" : None,
        "use_cache" : False,
        "output_attentions" : None,
    }

    @tf.function
    def test(**inputs):
        return tf_layer(**inputs)

    # Trace
    full_model = test.get_concrete_function(**trace_inputs)

    # Get frozen graph def
    frozen_func = convert_variables_to_constants_v2(full_model)
    graph_def = frozen_func.graph.as_graph_def()

    mod, params = tvm.relay.frontend.from_tensorflow(graph_def)

    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    func = compile_tvm_for_buda(mod, params)
    
    res = func(hidden_states)

    if isinstance(res, (list, tuple)):
        res = res[0]
    res = res.numpy()

    res_pt = tf_layer(hidden_states)
    if isinstance(res_pt, (list, tuple)):
        res_pt = res_pt[0]

    res_pt = res_pt.numpy()

    print(f"Results correct: {np.allclose(res, res_pt, atol=1e-6)}")


if __name__ == "__main__":
    main()
