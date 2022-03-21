import logging

import tvm
from tvm import relay
from tvm._ffi.base import TVMError
from tvm.ir.transform import PassContext
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name, BuildModule,build_target_by_device_type_map
from tvm.ir import IRModule
from tvm.relay import function as _function
from tvm.target.compilation_config import make_compilation_config
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

from tvm.relay.testing import run_infer_type
import numpy as np
import math
import numpy as np
from tvm.relay.dataflow_pattern import *

logger = logging.getLogger("Buda")

def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.buda")
    def _func_wrapper(expr):
        return supported
    return _func_wrapper

_register_external_op_helper("add")
_register_external_op_helper("gelu")
_register_external_op_helper("layernorm")
_register_external_op_helper("log")
_register_external_op_helper("mean")
_register_external_op_helper("multiply")
_register_external_op_helper("nn.batch_matmul")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("reciprocal")
_register_external_op_helper("reshape")
_register_external_op_helper("sqrt")
_register_external_op_helper("subtract")
_register_external_op_helper("transpose")

def nn_layernorm_to_buda_layernorm():
    act = wildcard()
    return is_op("layernorm")

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
    
    if (not (len(r_newshape) == 2 or (len(r_newshape) == 3) and r_newshape[0].value == 1)
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

# def is_stack_reshape_squeeze_to_hstack(call):
#     dim = len(call.checked_type.shape)
#     squeeze_axis = call.attrs.axis
#     assert len(squeeze_axis) == 1, "TODO"
#     squeeze_axis = squeeze_axis[0].value

#     if squeeze_axis < 0:
#         squeeze_axis = squeeze_axis + dim
#     stack_axis = call.args[0].args[0].attrs.axis.value
#     if stack_axis < 0:
#         stack_axis = stack_axis + dim

#     return squeeze_axis == dim and stack_axis == (dim - 1)

# def stack_reshape_squeeze_to_hstack():
#     act = is_tuple(None)
#     stack = is_op("stack")(act)
#     rshp = is_op("reshape")(stack)
#     return is_op("squeeze")(rshp)


@register_pattern_table("buda")
def pattern_table():
    matmul = ("buda.matmul", dense_to_matmul())
    hslice = ("buda.hslice", reshape_transpose_to_hslice(), is_reshape_transpose_hslice)
    hstack = ("buda.hstack", transpose_reshape_to_hstack(), is_transpose_reshape_hstack)
    layernorm = ("buda.layernorm", nn_layernorm_to_buda_layernorm())
    buda_patterns = [hstack, hslice, matmul]
    return buda_patterns


class DecomposeMultiAxisTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.transpose = is_op("transpose")(self.act)
        self.pattern = self.transpose

    def callback(self, pre, post, node_map):

        transpose_axes = post.attrs.axes
        acts = node_map[self.act][0]
        trans = node_map[self.transpose][0]

        if transpose_axes == None:
            return post

        changed_axis = []

        for i, axis in enumerate(transpose_axes):
            if i != axis:
                changed_axis.append([i, axis])


        assert len(changed_axis) >= 2, "Transpose op has at least 2 axes changed"
        if len(changed_axis) == 2:
            return post

        ndim = len(trans.type_args[0].shape)
        total_permute = list(transpose_axes)
        total_permute = [int(x) for x in total_permute]
        no_permute = list(range(ndim))
        output = acts

        for i in range(ndim):
            if total_permute[i] != i:
                new_order = total_permute.copy()
                new_order[i + 1 :] = no_permute[i + 1 :]
                new_order[new_order[i]] = i

                total_permute = [new_order[i] for i in total_permute]
                output = tvm.relay.transpose(output, axes=new_order)

        return output


class DecomposePower(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.exponent = is_constant()
        power = is_op("power")(self.act, self.exponent)

        self.pattern = power

    def callback(self, pre, post, node_map):

        act = node_map[self.act][0]
        exponent = node_map[self.exponent][0].data.numpy().item()

        op = act
        if exponent < 0:
            exponent *= -1
            ADD_DIVIDE = True
        else:
            ADD_DIVIDE = False

        if not exponent.is_integer():
            dec = exponent - int(exponent)
            assert dec == 0.5 , "Only support a single sqrt for now"
            op = tvm.relay.sqrt(op)
            exponent -= dec

        if exponent.is_integer():
            original_op = op
            while exponent > 1:
                op = tvm.relay.multiply(op, original_op)
                exponent -= 1

        if ADD_DIVIDE:
            # add an divide
            op = tvm.relay.reciprocal(op)

        assert exponent == 0 or exponent == 1, "Exponent has not been decomposed"

        return op


class ReconstructTFGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.sqrt_half = wildcard()
        self.half = is_constant()
        self.one = is_constant()
        times_root_two = is_op("multiply")(self.act, self.sqrt_half)
        erf = is_op("erf")(times_root_two)
        times_half = is_op("multiply")(self.act, self.half)
        add = is_op("add")(self.one, erf)
        gelu = is_op("multiply")(times_half, add)

        self.pattern = gelu

    def callback(self, pre, post, node_map):

        one_added = math.isclose(node_map[self.one][0].data.numpy(), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        half_multiplied = math.isclose(node_map[self.half][0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)
        root_two_multiplied = math.isclose(node_map[self.sqrt_half][0].args[0].data.numpy(), 1.4142135, rel_tol=1e-6, abs_tol=1e-6)
        
        if not (one_added and half_multiplied and root_two_multiplied):
            return post

        return tvm.relay.gelu(node_map[self.act][0])

class ReconstructPyTorchGelu(DFPatternCallback):
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

class ReconstructPyTorchLayerNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.gamma = is_constant()
        self.beta = is_constant()
        self.eps = is_constant()

        mean_act = is_op("mean")(self.act)
        sub_0 = is_op("subtract")(self.act, mean_act)
        mul_0 = is_op("multiply")(sub_0, sub_0)
        var = is_op("mean")(mul_0)

        sum_denom = var.optional(lambda x: is_op("add")(x, self.eps))
        sub = is_op("subtract")(self.act, mean_act)
        denom = is_op("sqrt")(sum_denom)
        recp = is_op("reciprocal")(denom)
        coef = is_op("multiply")(sub, recp)
        mul = is_op("multiply")(coef, self.gamma)
        layernorm = is_op("add")(mul, self.beta)

        self.pattern = layernorm


    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        gamma = node_map[self.gamma][0]
        beta = node_map[self.beta][0]

        try:
            eps = node_map[self.eps][0].data.asnumpy().item()
        except TVMError: # Does not have epsilon addition
            eps = 0

        act_shape = list(act.checked_type.shape)

        assert len(gamma.data.shape) == 1, "TVM Layernorm only supports single dim layernorm"
        axis = None
        # Find the last dimension of the specific size
        for i, dim in enumerate(reversed(act_shape)):
            if dim == gamma.data.shape[0]:
                axis = (i * -1) - 1 # i == 0 means axis = -1
                break

        assert axis is not None, "Cannot find an axis in input activation that matches weight shape"

        return tvm.relay.layernorm(act, gamma, beta, eps, axis)

class ReconstructTFLayerNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.gamma = is_constant()
        self.beta = is_constant()
        self.eps = is_constant()

        mean_act = is_op("mean")(self.act)
        sub_0 = is_op("subtract")(self.act, mean_act)
        mul_0 = is_op("multiply")(sub_0, sub_0)
        var = is_op("mean")(mul_0)

        sum_denom = var.optional(lambda x: is_op("add")(x, self.eps))
        denom = is_op("sqrt")(sum_denom)
        recp = is_op("reciprocal")(denom)

        weight = is_op("multiply")(self.gamma, recp)
        mean_part = is_op("multiply")(mean_act, weight)
        act_part = is_op("multiply")(weight, self.act)
        sub_1 = is_op("subtract")(self.beta, mean_part)
        layernorm = is_op("add")(sub_1, act_part)
        # import pdb; pdb.set_trace()
        self.pattern = layernorm

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        gamma = node_map[self.gamma][0]
        beta = node_map[self.beta][0]

        try:
            eps = node_map[self.eps][0].data.numpy().item()
        except TVMError: # Does not have epsilon addition
            eps = 0

        act_shape = list(act.checked_type.shape)

        assert len(gamma.data.shape) == 1, "TVM Layernorm only supports single dim layernorm"
        axis = None
        # Find the last dimension of the specific size
        for i, dim in enumerate(reversed(act_shape)):
            if dim == gamma.data.shape[0]:
                axis = (i * -1) - 1 # i == 0 means axis = -1
                break

        assert axis is not None, "Cannot find an axis in input activation that matches weight shape"

        return tvm.relay.layernorm(act, gamma, beta, eps, axis)

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
            return post

        # Doesn't have transpose, add two, one to fuse, the other to undo
        act = post.args[0]
        weight = post.args[1]
        wt1 = tvm.relay.transpose(weight)
        wt2 = tvm.relay.transpose(wt1)
        return tvm.relay.nn.dense(act, wt2)


class LiftLinearSplit(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        act = wildcard()
        self.dense_weight = is_constant()
        self.add_bias = is_constant()
        self.dense = is_op("nn.dense")(act, self.dense_weight)
        self.add = is_op("add")(self.dense, self.add_bias)
        self.reshape = is_op("reshape")(self.add)
        self.pattern = is_op('split')(self.reshape)

    def callback(self, pre, post, node_map):
        weight = node_map[self.dense_weight][0].data.numpy()
        bias = node_map[self.add_bias][0].data.numpy()

        indices_or_sections = post.attrs.indices_or_sections
        if isinstance(indices_or_sections, tvm.tir.expr.IntImm):
            # Convert to list of split positions
            num_parts = int(indices_or_sections)
            split_idx = np.linspace(0, weight.shape[axis], num_parts + 1).astype(np.int)[1:-1]
            indices_or_sections = tuple(split_idx)
    
        ios = np.array(indices_or_sections).astype(int)
        axis = post.attrs.axis

        input_shape = node_map[self.dense][0].checked_type.shape
        output_shape = node_map[self.reshape][0].checked_type.shape

        newshape = list(output_shape)
        newshape[axis] = -1

        if (is_unsqueeze(node_map[self.reshape][0])):
            # Weight should be transposed in nn.dense, so if splitting
            # along the final output axis, split along the first weight
            if axis == len(output_shape) - 1:
                assert output_shape[axis] == weight.shape[0]
                axis = 0

        act = node_map[self.dense][0].args[0]

        split_weights = np.split(weight, ios, axis)
        split_biases = np.split(bias, ios, len(bias.shape) - 1)

        outputs = []
        for split_weight, split_bias in zip(split_weights, split_biases):
            dense_out = tvm.relay.nn.dense(act, tvm.relay.Constant(tvm.nd.array(split_weight)))
            add_out = tvm.relay.add(dense_out, tvm.relay.Constant(tvm.nd.array(split_bias)))
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

class DecomposeNegative(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.in_a = wildcard()

        self.pattern = is_op('negative')(self.in_a)

    def callback(self, pre, post, node_map):
        negative_one = tvm.relay.const(-1.0, dtype=post.checked_type.dtype)
        mul = tvm.relay.multiply(post.args[0], negative_one)
        return mul

class DecomposeRsqrt(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.in_a = wildcard()

        self.pattern = is_op('rsqrt')(self.in_a)

    def callback(self, pre, post, node_map):
        sqrt = tvm.relay.sqrt(post.args[0])
        rsqrt = tvm.relay.reciprocal(sqrt)
        return rsqrt

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


class UpdateConstants(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.params = {}
        self.const_idx = 0
        self.pattern = is_constant()

    def callback(self, pre, post, node_map):
        self.params[self.const_idx] = post.data
        self.const_idx += 1
        return post

def run_relay_compile_passes(relay_module, print_all=False):

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

    relay_module = tvm.transform.Sequential([transform.CanonicalizeCast()])(relay_module)
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

    # relay_module = tvm.transform.Sequential([transform.transform.FuseOps()])(relay_module)
    # if print_all:
    #     print("After FuseOps")
    #     print(relay_module.functions)

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


    relay_module = tvm.transform.Sequential([transform.FoldConstant()])(relay_module)
    if print_all:
        print("After FoldConstant")
        print(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    if print_all:
        print("After InferType")
        print(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CanonicalizeOps()])(relay_module)
    if print_all:
        print("After CanonicalizeOps")
        print(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    if print_all:
        print("After InferType")
        print(relay_module.functions)

    return relay_module


def run_buda_compile_passes(relay_module, print_all=False):
    relay_module = tvm.transform.Sequential([transform.DecomposeVariance()])(relay_module)
    if print_all:
        print("After DecomposeVariance")
        print(relay_module.functions)
    if print_all:
        print("After CanonicalizeOps")
        print(relay_module.functions)
    relay_module["main"] = rewrite(LiftLinearSplit(), relay_module["main"])
    if print_all:
        print("After LiftLinearSplit")
        print(relay_module.functions)
    relay_module["main"] = rewrite(DenseWeightTranspose(), relay_module["main"])
    if print_all:
        print("After DenseWeightTranspose")
        print(relay_module.functions)
    relay_module["main"] = rewrite(DecomposePower(), relay_module["main"])
    if print_all:
        print("After DecomposePower")
        print(relay_module.functions)
    relay_module["main"] = rewrite(DecomposeNegative(), relay_module["main"])
    if print_all:
        print("After DecomposeNegative")
        print(relay_module.functions)
    relay_module["main"] = rewrite(DecomposeRsqrt(), relay_module["main"])
    if print_all:
        print("After DecomposeRsqrt")
        print(relay_module.functions)
    relay_module["main"] = rewrite(InvertDivide(), relay_module["main"])
    if print_all:
        print("After InvertDivide")
        print(relay_module.functions)
    relay_module["main"] = rewrite(ExplicateTranspose(), relay_module["main"])
    if print_all:
        print("After ExplicateTranspose")
        print(relay_module.functions)
    relay_module["main"] = rewrite(ExplicateHSliceTranspose(), relay_module["main"])
    if print_all:
        print("After ExplicateHSliceTranspose")
        print(relay_module.functions)
    relay_module["main"] = rewrite(DecomposeMultiAxisTranspose(), relay_module["main"])
    if print_all:
        print("After DecomposeMultiAxisTranspose")
        print(relay_module.functions)
    relay_module["main"] = rewrite(EstimateWhere(), relay_module["main"])
    if print_all:
        print("After EstimateWhere")
        print(relay_module.functions)

    return relay_module


def reconstruct_ops_for_buda(mod):
    print_all = False
    if print_all:
        print("reconstruct_ops_for_buda:: At Entry")
        print(mod.functions)
    mod["main"] = rewrite(ReconstructPyTorchGelu(), mod["main"])
    if print_all:
        print("After ReconstructPyTorchGelu")
        print(mod.functions)
    mod["main"] = rewrite(ReconstructTFGelu(), mod["main"])
    if print_all:
        print("After ReconstructTFGelu")
        print(mod.functions)
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    if print_all:
        print("After InferType")
        print(mod.functions)
    mod["main"] = rewrite(ReconstructTFLayerNorm(), mod["main"])
    if print_all:
        print("After ReconstructTFLayerNorm")
        print(mod.functions)
    mod["main"] = rewrite(ReconstructPyTorchLayerNorm(), mod["main"])
    if print_all:
        print("After ReconstructPyTorchLayerNorm")
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

        relay_module = run_relay_compile_passes(relay_module, print_all=print_all)
        compiled_relay_module = run_buda_compile_passes(relay_module, print_all=print_all)

    return compiled_relay_module, params


def partition_for_buda(mod):
    print_all = False
    with tvm.transform.PassContext(opt_level=5):
        if print_all:
            print("partition_for_buda:: At Entry")
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
        mod = tvm.transform.Sequential([transform.PartitionGraph(bind_constants=True)])(mod)
        if print_all:
            print("After PartitionGraph")
            print(mod.functions)
        assert len(mod.get_global_vars()) == 2

        constant_updator = UpdateConstants()
        rewrite(constant_updator, mod[mod.get_global_vars()[1]])
        params = constant_updator.params
        
    return mod, params