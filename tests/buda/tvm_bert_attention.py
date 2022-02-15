import torch
import tvm
import tvm.relay

# from visualize_tvm import visualize

from transformers import BertModel, BertTokenizer

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

    hidden_states = model.embeddings(tokens_tensor)
    bert_layer = model.encoder.layer[0]

    traced_model = torch.jit.trace(bert_layer.attention, hidden_states)
    input_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]

    tvm_model, tvm_params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, input_list, default_dtype="float32")
    target = "llvm"
    tvm.relay.backend.te_compiler.get().clear()
    # with tvm.transform.PassContext(opt_level=4):
    tvm_model_opt, tvm_params_opt = tvm.relay.op.contrib.compile_for_buda(tvm_model, target=target, params=tvm_params)


    # # Need graphviz to visualize
    # viz = visualize(tvm_model["main"])
    # viz.save()

    # tvm.relay.backend.te_compiler.get().clear()
    # with tvm.transform.PassContext(opt_level=4):
    #     tvm_model_opt, tvm_params_opt = tvm.relay.optimize(mod=tvm_model, target=target, params=tvm_params)
        # graph, lib, params = tvm.relay.build(tvm_model,
        #                              target=target,
        #                              params=tvm_params)

    # viz_opt = visualize(tvm_model_opt["main"])
    # viz_opt.filename='Digraph_opt.gv'
    # viz_opt.save()
    # dev = tvm.cpu()
    # module = tvm.contrib.graph_executor.create(graph, lib, dev)

    # hs_a = tvm.nd.array(hidden_states.numpy(), dev)
    # module.set_input("hidden_states", hs_a)
    # module.set_input(**params)

    # module.run()

    # output = module.get_output(0)

if __name__ == "__main__":
    main()
