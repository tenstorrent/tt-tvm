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

from tvm.relay.testing import run_infer_type

import math
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
_register_external_op_helper("mean")
_register_external_op_helper("nn.batch_matmul")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("sqrt")
_register_external_op_helper("reciprocal")
_register_external_op_helper("gelu")
_register_external_op_helper("nn.layer_norm")


def nn_layernorm_to_buda_layernorm():
    act = wildcard()
    return is_op("nn.layer_norm")

def dense_to_matmul():
    data = wildcard()
    weight = wildcard()
    weight_t = is_op('transpose')(weight)
    return is_op('nn.dense')(data, weight_t)

def is_unsqueeze(call):
    input_shape = call.args[0].checked_type.shape
    output_shape = call.checked_type.shape

    joint_size = min(len(input_shape), len(output_shape))
    
    superfluous_reshape = all([input_shape[i] == output_shape[i] for i in range(-1, -1*joint_size - 1, -1)])

    if superfluous_reshape and len(input_shape) < len(output_shape):
        return True
        
    return False

def is_squeeze(call):
    input_shape = call.args[0].checked_type.shape
    output_shape = call.checked_type.shape

    joint_size = min(len(input_shape), len(output_shape))
    
    superfluous_reshape = all([input_shape[i] == output_shape[i] for i in range(-1, -1*joint_size - 1, -1)])

    if superfluous_reshape and len(input_shape) > len(output_shape):
        return True
        
    return False

def is_superfluous_reshape(call):
    input_shape = call.args[0].checked_type.shape
    output_shape = call.checked_type.shape

    joint_size = min(len(input_shape), len(output_shape))

    superfluous_reshape = all([input_shape[i] == output_shape[i] for i in range(-1, -1*joint_size - 1, -1)])

    if len(input_shape) > len(output_shape):
        extra_dims = len(input_shape) - len(output_shape)
        if all([extra_dim == 1 for extra_dim in input_shape[:extra_dims]]):
            superfluous_reshape = superfluous_reshape and True
    elif len(output_shape) > len(input_shape):
        extra_dims = len(output_shape) - len(input_shape)
        if all([extra_dim == 1 for extra_dim in output_shape[:extra_dims]]):
            superfluous_reshape = superfluous_reshape and True
    
    return superfluous_reshape
            
def is_reshape_hslice(call):
    r_input_shape = call.args[0].type_args[0].shape
    r_newshape = call.args[0].checked_type.shape

    if (not (len(r_newshape) == 3 or (len(r_newshape) == 4 and r_newshape[0].value == 1)) 
    or not (r_input_shape[-2].value == r_newshape[-3].value) 
    or is_superfluous_reshape(call)):
            return False

    return True

def is_transpose_hslice(call):
    t_axes = call.attrs.axes
    hslice_t_axes = (0, 2, 1, 3)
    
    if not (len(t_axes) == 4 and all([hslice_t_axes[i] == t_axes[i] for i in range(4)])):
        return False

    return True

def is_reshape_transpose_hslice(call):
    return is_reshape_hslice(call) and is_transpose_hslice(call)

def is_transpose_reshape_hstack(call):
    t_axes = call.args[0].attrs.axes
    hstack_t_axes = (0, 2, 1, 3)
    
    if not (len(t_axes) == 4 and all([hstack_t_axes[i] == t_axes[i] for i in range(4)])):
        return False

    r_newshape = call.checked_type.shape
    r_input_shape = call.type_args[0].shape
    
    if (not len(r_newshape) == 2
    or not all([dim == 1 for dim in r_newshape[:-2]])
    or not (r_input_shape[-3].value == r_newshape[-2].value)
    or is_superfluous_reshape(call)):
            return False

    return True

def reshape_transpose_to_hslice():
    act = wildcard()
    act_r = is_op('reshape')(act)
    return is_op('transpose')(act_r)
    
def transpose_reshape_to_hstack():
    act = wildcard()
    act_t = is_op("transpose")(act)
    return is_op("reshape")(act_t)

@register_pattern_table("buda")
def pattern_table():
    matmul = ("buda.matmul", dense_to_matmul())
    hslice = ("buda.hslice", reshape_transpose_to_hslice(), is_reshape_transpose_hslice)
    hstack = ("buda.hstack", transpose_reshape_to_hstack(), is_transpose_reshape_hstack)
    layernorm = ("buda.layernorm", nn_layernorm_to_buda_layernorm())
    buda_patterns = [hstack, hslice, matmul]
    return buda_patterns


class ReconstructGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        act = wildcard()
        times_root_two = is_op("multiply")(act, is_constant())
        erf = is_op("erf")(times_root_two)
        times_half = is_op("multiply")(erf, is_constant())
        add = is_op("add")(is_constant(), times_half)
        gelu = is_op("multiply")(act, add)

        self.pattern = gelu

    def callback(self, pre, post, node_map):
        half_added = math.isclose(post.args[1].args[0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)
        half_multiplied = math.isclose(post.args[1].args[1].args[1].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)
        root_two_multiplied = math.isclose(post.args[1].args[1].args[0].args[0].args[1].data.numpy(), 0.70710677, rel_tol=1e-6, abs_tol=1e-6)
        
        if not (half_added and half_multiplied and root_two_multiplied):
            return post

        return tvm.relay.gelu(post.args[0])


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
        if self.transpose_pattern.match(post.args[1]):
            print("has transpose")
            return post

        # Doesn't have transpose, add two, one to fuse, the other to undo
        print("doesn't have transpose")
        act = post.args[0]
        weight = post.args[1]
        wt1 = tvm.relay.transpose(weight)
        wt2 = tvm.relay.transpose(wt1)
        return tvm.relay.nn.dense(act, wt2)


class LiftLinearSplit(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        act = wildcard()
        self.dense = is_op("nn.dense")(act, is_constant())
        self.add = is_op("add")(self.dense, is_constant())
        self.reshape = is_op("reshape")(self.add)
        self.pattern = is_op('split')(self.reshape)

    def callback(self, pre, post, node_map):
        weight = node_map[self.dense][0].args[1]
        bias = node_map[self.add][0].args[1]

        indices_or_sections = post.attrs.indices_or_sections
        axis = post.attrs.axis
        input_shape = node_map[self.dense][0].checked_type.shape
        output_shape = node_map[self.reshape][0].checked_type.shape

        newshape = list(output_shape)
        newshape[axis] = -1

        if (is_unsqueeze(node_map[self.reshape][0])):
            # Weight should be transposed in nn.dense, so if splitting
            # along the final output axis, split along the first weight
            if axis == len(output_shape) - 1:
                assert output_shape[axis] == weight.data.shape[0]
                axis = 0

        split_weights = tvm.relay.split(weight, indices_or_sections=indices_or_sections, axis=axis)
        split_biases = tvm.relay.split(bias, indices_or_sections=indices_or_sections, axis=0)

        outputs = []
        act = node_map[self.dense][0].args[0]
        for split_weight, split_bias in zip(split_weights, split_biases):
            dense_out = tvm.relay.nn.dense(act, split_weight)
            add_out = tvm.relay.add(dense_out, split_bias)
            reshape_out = tvm.relay.reshape(add_out, newshape=newshape)
            outputs.append(reshape_out)

        return tvm.relay.expr.Tuple(outputs)

        

class ExplicateHSliceTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        act = wildcard()
        act_r = is_op('reshape')(act)
        self.pattern = is_op('transpose')(act_r)

    def callback(self, pre, post, node_map):
        if is_reshape_transpose_hslice(post) or not is_reshape_hslice(post):
            return post

        t_axes = post.attrs.axes
        hslicet_t_taxes = [0, 2, 3, 1]
        
        if not (len(t_axes) == 4 and all([hslicet_t_taxes[i] == t_axes[i] for i in range(4)])):
            return post

        act = post.args[0].args[0]
        r = tvm.relay.reshape(act, newshape=post.args[0].attrs.newshape)
        rt = tvm.relay.transpose(r, axes=[0, 2, 1, 3])
        rtt = tvm.relay.transpose(rt, axes=[0, 1, 3, 2])

        return rtt


class EstimateWhere(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)

        self.pattern = is_op('where')(wildcard(), wildcard(), wildcard())
        
    def callback(self, pre, post, node_map):
        # by assuming the masked value is >> activation, this allows
        # so simulate causal masking with eltwise ops, i.e. simply add
        # the masked value
        causal_mask = post.args[0]
        zero = tvm.relay.const(0.0, dtype=post.checked_type.dtype)
        one = tvm.relay.const(1.0, dtype=post.checked_type.dtype)
        inverse_causal_mask = tvm.relay.where(causal_mask, zero, one)
        
        value = post.args[2]
        mask = tvm.relay.multiply(inverse_causal_mask, value)

        act = post.args[1]
        return tvm.relay.add(act, mask)

class InvertDivide(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.in_a = wildcard()
        self.in_b = wildcard()

        self.pattern = is_op('divide')(self.in_a, self.in_b)

    def callback(self, pre, post, node_map):
        rep = tvm.relay.reciprocal(post.args[1])
        return tvm.relay.multiply(post.args[0], rep)


class ExplicateTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()

        self.pattern = is_op('nn.batch_matmul')(wildcard(), wildcard())

    def callback(self, pre, post, node_map):
        transpose_a = post.attrs.transpose_a
        transpose_b = post.attrs.transpose_b

        if not (transpose_a or transpose_b):
            return post

        a = post.args[0]
        ndim = len(post.args[0].checked_type.shape)
        axes = list(range(ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        if transpose_a:
            a = tvm.relay.transpose(a, axes=axes)
        b = post.args[1]
        if transpose_b:
            b = tvm.relay.transpose(b, axes=axes)
            
        return tvm.relay.nn.batch_matmul(a, b, transpose_a=False, transpose_b=False)

def partition_for_buda(mod):
    print_all = False
    with tvm.transform.PassContext(opt_level=3):
        if print_all:
            print("At Entry")
            print(mod.functions)
        mod = tvm.transform.Sequential([transform.CanonicalizeOps()])(mod)
        if print_all:
            print("After CanonicalizeOps")
            print(mod.functions)
        mod["main"] = rewrite(LiftLinearSplit(), mod["main"])
        if print_all:
            print("After LiftLinearSplit")
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
        mod["main"] = rewrite(ExplicateHSliceTranspose(), mod["main"])
        if print_all:
            print("After ExplicateHSliceTranspose")
            print(mod.functions)
        mod["main"] = rewrite(ReconstructGelu(), mod["main"])
        if print_all:
            print("After ReconstructGelu")
            print(mod.functions)
        mod["main"] = rewrite(EstimateWhere(), mod["main"])
        if print_all:
            print("After EstimateWhere")
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

    print_all = False
    with tophub_context, tvm.transform.PassContext(opt_level=5):
        bld_mod = BuildModule()
        if params:
            bld_mod._set_params(params)
        context = PassContext().current()
        compiler_config = make_compilation_config(context,target)

        if print_all:
            print("Before Compiling")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
        if print_all:
            print("After InferType")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.RemoveUnusedFunctions()])(relay_module)
        if print_all:
            print("After RemoveUnusedFunctions")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.ToBasicBlockNormalForm()])(relay_module)
        if print_all:
            print("After ToBasicBlockNormalForm")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.Legalize()])(relay_module)
        if print_all:
            print("After Legalize")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.SimplifyInference()])(relay_module)
        if print_all:
            print("After SimplifyInference")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.DynamicToStatic()])(relay_module)
        if print_all:
            print("After DynamicToStatic")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.EliminateCommonSubexpr()])(relay_module)
        if print_all:
            print("After EliminateCommonSubexpr")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.SimplifyExpr()])(relay_module)
        if print_all:
            print("After SimplifyExpr")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.CombineParallelConv2D(3)])(relay_module)
        if print_all:
            print("After CombineParallelConv2D")
            print(relay_module.functions)

        # relay_module = tvm.transform.Sequential([transform.CombineParallelDense(3)])(relay_module)
        # if print_all:
        #     print("After CombineParallelDense")
        #     print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.CombineParallelBatchMatmul(3)])(relay_module)
        if print_all:
            print("After CombineParallelBatchMatmul")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.FoldConstant()])(relay_module)
        if print_all:
            print("After FoldConstant")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.FoldScaleAxis()])(relay_module)
        if print_all:
            print("After FoldScaleAxis")
            print(relay_module.functions)

        # relay_module = tvm.transform.Sequential([transform.CanonicalizeCast()])(relay_module)
        if print_all:
            print("After CanonicalizeCast")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.CanonicalizeOps()])(relay_module)
        if print_all:
            print("After CanonicalizeOps")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
        if print_all:
            print("After InferType")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.FoldConstant()])(relay_module)
        if print_all:
            print("After FoldConstant")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
        if print_all:
            print("After InferType")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.Inline()])(relay_module)
        if print_all:
            print("After Inline")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
        if print_all:
            print("After InferType")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.DecomposeVariance()])(relay_module)
        if print_all:
            print("After DecomposeVariance")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.FoldConstant()])(relay_module)
        if print_all:
            print("After FoldConstant")
            print(relay_module.functions)

        relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
        if print_all:
            print("After InferType")
            print(relay_module.functions)

        compiled_relay_module = relay_module

    return compiled_relay_module, params