import torch
import tvm
import tvm.relay

import numpy as np

from transformers import BertModel, BertTokenizer
from python.contrib.pybuda_compile import compile_tvm_for_buda

def main():
    enc = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenizing input text
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)

    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    dummy_input = [tokens_tensor, segments_tensors]

    # model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
    model = BertModel.from_pretrained("prajjwal1/bert-tiny", torchscript=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # hidden_states = model.embeddings(tokens_tensor)
    shape = (1, 64, 128)
    hidden_states = torch.rand(*shape)

    torchmod = model.encoder.layer[0].attention

    traced_model = torch.jit.trace(torchmod, hidden_states)
    input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]

    mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, input_list)
    mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

    func = compile_tvm_for_buda(mod, params)

    res = func(hidden_states)
    if isinstance(res, (list, tuple)):
        res = res[0]
    res = res.numpy()

    res_pt = torchmod(hidden_states)
    if isinstance(res_pt, (list, tuple)):
        res_pt = res_pt[0]

    res_pt = res_pt.detach().numpy()

    print(f"Results correct: {np.allclose(res, res_pt, atol=1e-6)}")

if __name__ == "__main__":
    main()
