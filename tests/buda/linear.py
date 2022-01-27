import numpy as np
import tvm
import tvm.relay as relay

import torch


shape = (64, 64)

torchmod = torch.nn.Linear(64, 64)
act = torch.rand(*shape)
scr = torch.jit.trace(torchmod, act)
mod, params = tvm.relay.frontend.from_pytorch(scr, [('input', (64, 64))])
#print(mod)
#print(params)
#mod = tvm.IRModule.from_expr(tvm.relay.build_module.bind_params_by_name(mod["main"], params))

shape = (64, 64)
w1 = relay.var("weight", shape=shape)
x1 = relay.var("x1", shape=shape)
wt1 = relay.transpose(w1)
m1 = relay.nn.dense(wt1, x1)
x2 = relay.var("x2", shape=shape)

w2 = relay.var("weight2", shape=shape)
m2 = relay.nn.dense(w2, x2)

y = relay.add(m1, m2)

func = relay.Function([x1, x2, w1, w2], y)
mod = tvm.IRModule.from_expr(func)
print(mod)


from tvm.relay.dataflow_pattern import *
class DenseWeightTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.weight = wildcard()
        act = wildcard()
        #self.opt_t = weight.optional(lambda x: is_op('transpose')(x))
        #self.pattern = is_op('nn.dense')(act, self.opt_t)
        self.pattern = is_op('nn.dense')(act, self.weight)
        self.transpose_pattern = is_op('transpose')(wildcard())

    def callback(self, pre, post, node_map):
        print("match:")
        print("PRE", pre)
        print("POST", post)
        print("NODE_MAP", node_map)

        #t = node_map[self.opt_t][0]
        #print("Transpose: ", t)
        print(type(pre))
        print(type(post))
        if self.transpose_pattern.match(pre.args[0]):
            print("has transpose")
            return post

        # Doesn't have transpose, add it
        print("doesn't have transpose")
        weight = pre.args[0]
        act = pre.args[1]
        wt1 = relay.transpose(weight)
        wt2 = relay.transpose(wt1)
        return relay.nn.dense(wt2, act)

#from tvm.relay.dataflow_pattern import rewrite
#out = rewrite(DenseWeightTranspose(), mod["main"])
#print(out)

#mod = relay.transform.FoldConstant()(mod)
#mod = relay.transform.FuseOps(fuse_opt_level=3)(mod)
#tvm.transform.PrintIR()(mod)
#mod = relay.transform.EliminateCommonSubexpr()(mod)
#print(mod)


if not tvm.get_global_func("relay.ext.buda", True):
    print("Buda codegen not available")
    exit(-2)

#mod = create_relay_module_from_model() # Output: Figure 1
mod = tvm.relay.op.contrib.buda.partition_for_buda(mod)
#from tvm.relay import transform
#mod = transform.AnnotateTarget("buda")(mod)
#mod = transform.MergeCompilerRegions()(mod)
#mod = transform.PartitionGraph()(mod)
print(mod)


ret = tvm.relay.build_module.build(mod, target="llvm")
print(ret)
exit(-1)

with tvm.transform.PassContext(opt_level=3):
    func = relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm").evaluate()

print(func)

#res = tvm.build(mod, target="buda")
#print(res)

