
import tvm
import tvm.relay

import numpy as np
import tensorflow as tf
from transformers.models.bert.modeling_tf_bert import TFBertModel, TFBertLayer
from transformers import BertTokenizer, BertConfig
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

    # Bert tiny config
    model_config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 128,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522
    }
    config = BertConfig(**model_config)
    tf_layer = TFBertLayer(config)
    # model = TFBertModel.from_pretrained("https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/config.json")

    # tf_layer = model.bert.encoder.layer[0]
    hidden_states = tf.convert_to_tensor(np.random.rand(1, 64, 128).astype(np.float32))

    trace_inputs = {
        "hidden_states" : hidden_states,
        "attention_mask" : None, 
        "head_mask" : None,
        "output_attentions" : False,
        "encoder_hidden_states" : None,
        "encoder_attention_mask" : None,
        "past_key_value": None,
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
