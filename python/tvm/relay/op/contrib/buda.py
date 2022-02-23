import logging

import tvm
from tvm.ir.transform import PassContext
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name, BuildModule,build_target_by_device_type_map
from tvm.ir import IRModule
from tvm.relay import function as _function
from tvm.target.compilation_config import make_compilation_config
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

from tvm.relay.dataflow_pattern import *

logger = logging.getLogger("Buda")

def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.buda")
    def _func_wrapper(expr):
        return supported
    return _func_wrapper

_register_external_op_helper("transpose")
_register_external_op_helper("add")
_register_external_op_helper("subtract")
_register_external_op_helper("multiply")
_register_external_op_helper("reshape")
_register_external_op_helper("nn.batch_matmul")
_register_external_op_helper("nn.softmax")

def dense_to_matmul():
    data = wildcard()
    weight = wildcard()
    weight_t = is_op('transpose')(weight)
    return is_op('nn.dense')(data, weight_t)

def is_reshape_transpose_hslice(call):
    t_axes = call.attrs.axes
    hslice_t_axes = (0, 2, 1, 3)
    
    if not (len(t_axes) == 4 and all([hslice_t_axes[i] == t_axes[i] for i in range(4)])):
        return False

    r_input_shape = call.args[0].type_args[0].shape
    r_newshape = call.args[0].attrs.newshape

    if not (len(r_newshape) == 3 or (len(r_newshape) == 4 and r_newshape[0] == 1)) and (r_input_shape[-2] == r_newshape[-3]):
        return False

    return True

def reshape_transpose_to_hslice():
    act = wildcard()
    act_r = is_op('reshape')(act)
    return is_op('transpose')(act_r)

@register_pattern_table("buda")
def pattern_table():
    matmul = ("buda.matmul", dense_to_matmul())
    hslice = ("buda.hslice", reshape_transpose_to_hslice(), is_reshape_transpose_hslice)
    buda_patterns = [hslice, matmul]
    return buda_patterns


class DenseWeightTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.weight = wildcard()
        act = wildcard()
        self.pattern = is_op('nn.dense')(act, self.weight)
        self.transpose_pattern = is_op('transpose')(wildcard())

    def callback(self, pre, post, node_map):
        # If there's already a transpose, we don't need another one to 
        # fuse into buda.matmul
        if self.transpose_pattern.match(pre.args[1]):
            print("has transpose")
            return post

        # Doesn't have transpose, add two, one to fuse, the other to undo
        print("doesn't have transpose")
        act = pre.args[0]
        weight = pre.args[1]
        wt1 = tvm.relay.transpose(weight)
        wt2 = tvm.relay.transpose(wt1)
        return tvm.relay.nn.dense(act, wt2)


class FoldReshapes(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()
        self.pattern = is_op('reshape')(self.input_tensor)
        self.visited_ops = []

    def callback(self, pre, post, node_map):
        input_shape = node_map[self.input_tensor][0].checked_type.shape
        output_shape = node_map[self.pattern][0].checked_type.shape
        
        assert len(input_shape) <= 4

        superfluous_reshape = False
        if len(input_shape) > len(output_shape):
            extra_dims = len(input_shape) - len(output_shape)
            if all([extra_dim == 1 for extra_dim in input_shape[:extra_dims]]):
                superfluous_reshape = True
        elif len(output_shape) > len(input_shape):
            extra_dims = len(output_shape) - len(input_shape)
            if all([extra_dim == 1 for extra_dim in output_shape[:extra_dims]]):
                superfluous_reshape = True
            
        if superfluous_reshape:
            a = pre.args[0]
            return tvm.relay.reshape(a, newshape=pre.attrs.newshape, not_squeeze_unsqueeze=False)

        return post


class InvertDivide(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.in_a = wildcard()
        self.in_b = is_constant()

        self.pattern = is_op('divide')(self.in_a, self.in_b)

    def callback(self, pre, post, node_map):
        one = tvm.relay.const(1.0)
        multiplicand = tvm.relay.divide(one, pre.args[1])
        return tvm.relay.multiply(pre.args[0], multiplicand)

class ExplicateTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()

        self.pattern = is_op('nn.batch_matmul')(wildcard(), wildcard())

    def callback(self, pre, post, node_map):
        transpose_a = pre.attrs.transpose_a
        transpose_b = pre.attrs.transpose_b

        if not (transpose_a or transpose_b):
            return post

        a = pre.args[0]
        ndim = len(pre.args[0].checked_type.shape)
        axes = list(range(ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        if transpose_a:
            a = tvm.relay.transpose(a, axes=axes)
        b = pre.args[1]
        if transpose_b:
            b = tvm.relay.transpose(b, axes=axes)
            
        return tvm.relay.nn.batch_matmul(a, b, transpose_a=False, transpose_b=False)

def partition_for_buda(mod):
    print_all = False
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.transform.Sequential([transform.CanonicalizeOps()])(mod)
        if print_all:
            print("After CanonicalizeOps")
            print(mod.functions)
        mod["main"] = rewrite(DenseWeightTranspose(), mod["main"])
        if print_all:
            print("After DenseWeightTranspose")
            print(mod.functions)
        mod["main"] = rewrite(InvertDivide(), mod["main"])
        if print_all:
            print("After InvertDivide")
            print(mod.functions)
        mod = tvm.transform.Sequential([transform.InferType()])(mod)
        if print_all:
            print("After InferType")
            print(mod.functions)
        mod["main"] = rewrite(ExplicateTranspose(), mod["main"])
        if print_all:
            print("After ExplicateTranspose")
            print(mod.functions)
        mod = tvm.transform.Sequential([transform.InferType()])(mod)
        if print_all:
            print("After InferType")
            print(mod.functions)
        mod = tvm.transform.Sequential([transform.MergeComposite(pattern_table())])(mod)
        if print_all:
            print("After MergeComposite")
            print(mod.functions)
        mod = tvm.transform.Sequential([transform.FoldConstant()])(mod)
        if print_all:
            print("After FoldConstant")
            print(mod.functions)
        mod = tvm.transform.Sequential([transform.AnnotateTarget("buda")])(mod)
        if print_all:
            print("After AnnotateTarget")
            print(mod.functions)
        mod = tvm.transform.Sequential([transform.MergeCompilerRegions()])(mod)
        if print_all:
            print("After MergeCompilerRegions")
            print(mod.functions)
        mod = tvm.transform.Sequential([transform.PartitionGraph()])(mod)
        if print_all:
            print("After PartitionGraph")
            print(mod.functions)
    return mod


def compile_for_buda(relay_module, target='llvm', params=None):

    if not isinstance(relay_module, (IRModule, _function.Function)):
        raise ValueError("Type of input parameter mod must be tvm.IRModule")

    if isinstance(relay_module, _function.Function):
        if params:
            relay_module = bind_params_by_name(relay_module, params)
        relay_module = IRModule.from_expr(relay_module)
        logger.warning(
            "Please use input parameter mod (tvm.IRModule) "
            "instead of deprecated parameter func (tvm.relay.function.Function)"
        )

    target = build_target_by_device_type_map(target)

    if isinstance(tvm.autotvm.DispatchContext.current, tvm.autotvm.FallbackContext):
        tophub_context = tvm.autotvm.tophub.context(list(target.values()))
    else:
        tophub_context = tvm.autotvm.utils.EmptyContext()


    with tophub_context, tvm.transform.PassContext(opt_level=5):
        bld_mod = BuildModule()
        if params:
            bld_mod._set_params(params)
        context = PassContext().current()
        compiler_config = make_compilation_config(context,target)

        passes = tvm.transform.Sequential(
            [
                transform.InferType(),
                transform.RemoveUnusedFunctions(),
                transform.ToBasicBlockNormalForm(),
                transform.Legalize(),
                transform.SimplifyInference(),
                transform.DynamicToStatic(),
                transform.EliminateCommonSubexpr(),
                transform.SimplifyExpr(),
                transform.CombineParallelConv2D(3),
                transform.CombineParallelDense(3),
                transform.CombineParallelBatchMatmul(3),
                transform.FoldConstant(),
                transform.FoldScaleAxis(),
                transform.CanonicalizeCast(),
                transform.CanonicalizeOps(),
                transform.InferType(),
                transform.FoldConstant(),
                transform.InferType(),
                transform.Inline(),
                transform.InferType(),
                transform.DecomposeVariance(),
                transform.FoldConstant(),
                transform.InferType(),
            ]
        )

        compiled_relay_module = passes(relay_module)

    return compiled_relay_module, params