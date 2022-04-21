import logging

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
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

from loguru import logger

def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.buda")
    def _func_wrapper(expr):
        return supported
    return _func_wrapper

_register_external_op_helper("add")
_register_external_op_helper("exp")
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
_register_external_op_helper("sigmoid")
_register_external_op_helper("sqrt")
_register_external_op_helper("strided_slice")
_register_external_op_helper("subtract")
_register_external_op_helper("transpose")
_register_external_op_helper("where")
_register_external_op_helper("nn.conv2d_transpose")
_register_external_op_helper("image.resize2d")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("clip")

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

def is_stack_reshape_reshape_to_binary_stack(call):
    dim = len(call.checked_type.shape)
    stack_axis = call.args[0].args[0].attrs.axis.value
    if stack_axis < 0:
        stack_axis = stack_axis + dim

    input_shape = [int(dim) for dim in call.args[0].args[0].args[0][0].checked_type.shape]
    output_shape = [int(dim) for dim in call.checked_type.shape]

    works = all([i == o or (dim == stack_axis and o == 2 * i) for dim, (i, o) in enumerate(zip(input_shape, output_shape))])
    return works

def stack_reshape_reshape_to_binary_stack():
    act = is_tuple(None)
    stack = is_op("stack")(act)
    rshp = is_op("reshape")(stack)
    return is_op("reshape")(rshp)


def decompose_concat_input_tuple():
    act = is_tuple(None)
    return is_op("concatenate")(act)


@register_pattern_table("buda")
def pattern_table():
    matmul = ("buda.matmul", dense_to_matmul())
    hslice = ("buda.hslice", reshape_transpose_to_hslice(), is_reshape_transpose_hslice)
    hstack = ("buda.hstack", transpose_reshape_to_hstack(), is_transpose_reshape_hstack)
    layernorm = ("buda.layernorm", nn_layernorm_to_buda_layernorm())
    binary_stack = ("buda.binary_stack", stack_reshape_reshape_to_binary_stack(), is_stack_reshape_reshape_to_binary_stack)
    concatenate = ("buda.concatenate", decompose_concat_input_tuple())
    buda_patterns = [hstack, hslice, matmul, binary_stack, concatenate]
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

        ndim = len(trans.args[0].checked_type.shape)
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

class DecomposeMultiAxisMean(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.mean = is_op("mean")(self.act)
        self.pattern = self.mean

    def callback(self, pre, post, node_map):
        reduce_axes = list(post.attrs.axis)
        if len(reduce_axes) == 1:
            return post

        acts = node_map[self.act][0]
        out = node_map[self.mean][0]
        keepdims = bool(post.attrs.keepdims)
        output_shape = list(out.checked_type.shape)

        for axis in reduce_axes:
            acts = tvm.relay.mean(acts, axis=int(axis), keepdims=True)
        
        if keepdims == False:
            acts = tvm.relay.reshape(acts, newshape=output_shape)
        return acts    

class DecomposeConv1DToConv2D(DFPatternCallback):
    def __init__(self): 
        super().__init__(rewrite_once=True)
        
        self.act = wildcard()
        self.weight = wildcard()
        self.conv_pattern = is_op("nn.conv1d")(self.act, self.weight)
        self.pattern = self.conv_pattern

    def callback(self, pre, post, node_map):
        
        if post.attrs.data_layout == 'NWC' and post.attrs.kernel_layout == 'WIO':
            assert False, "Conv1d from TF is not supported yet"
            # TODO: converting TF conv1d - channel-last to channel-first Conv2d
        else:
            assert post.attrs.kernel_size[0] == 1, "Conv2d only support square kernels, since Conv1d is decomposed to Conv2d, it can only have kernel size of 1"
            assert post.attrs.padding[0] == 0, "Paddings are not support for conv1d"
            # reshape activation and reshape weight 
            expected_output_shape = node_map[self.pattern][0].checked_type.shape
            acts = node_map[self.act][0]
            acts_shape = acts.checked_type.shape
            weights = node_map[self.weight][0]
            weights_shape = weights.checked_type.shape
            
            acts_shape = [acts_shape[0], acts_shape[1], acts_shape[2], 1]
            weights_shape = [weights_shape[0], weights_shape[1], weights_shape[2], 1]
            reshaped_acts = tvm.relay.reshape(acts, newshape=acts_shape)
            reshaped_weights = tvm.relay.reshape(weights, newshape=weights_shape)
            
            new_conv2d = tvm.relay.op.nn.conv2d(
                reshaped_acts, 
                reshaped_weights,
                strides=[post.attrs.strides[0], 1],
                padding=[post.attrs.padding[0], 0, post.attrs.padding[0], 0],
                groups=post.attrs.groups,
                channels=post.attrs.channels,
                kernel_size=[post.attrs.kernel_size[0], 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
            )

            # reshape back result 
            reshaped_back_result = tvm.relay.reshape(new_conv2d, newshape=expected_output_shape)
            return reshaped_back_result




class ReformatTFConv2d(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.weight = is_constant()
        conv = is_op("nn.conv2d")(self.act, self.weight)
        self.pattern = conv

    def callback(self, pre, post, node_map):
        if post.attrs.data_layout == 'NHWC' and post.attrs.kernel_layout == 'HWIO':
            # convert TF channel-last to channel-first
            conv2d_shape = node_map[self.pattern][0].checked_type.shape
            act = node_map[self.act][0]
            weight = node_map[self.weight][0]
            act_shape = act.checked_type.shape
            weight_shape = weight.checked_type.shape

            channel_first_act = tvm.relay.transpose(act, axes=[0, 3, 1, 2])
            channel_first_weight = tvm.relay.transpose(weight, axes=[3, 2, 0, 1])

            new_conv2d = tvm.relay.op.nn.conv2d(
                channel_first_act,
                channel_first_weight,
                strides=post.attrs.strides,
                padding=post.attrs.padding,
                groups=post.attrs.groups,
                channels=post.attrs.channels,
                kernel_size=post.attrs.kernel_size,
                data_layout="NCHW",
                kernel_layout="OIHW",
            )
            out_reshape = tvm.relay.transpose(new_conv2d, axes=[0,2,3,1])
            return out_reshape
        else:
            return post


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
        self.gamma = wildcard()
        self.beta = wildcard()
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
        assert len(gamma.checked_type.shape) == 1, "TVM Layernorm only supports single dim layernorm"
        axis = None
        # Find the last dimension of the specific size
        for i, dim in enumerate(reversed(act_shape)):
            if dim == gamma.checked_type.shape[0]:
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


class LowerSplitToStridedSlice(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        act = wildcard()
        self.split = is_op("split")(act)

        self.pattern = is_tuple_get_item(wildcard())

    def callback(self, pre, post, node_map):
        split = post.tuple_value().op

        if not self.split.match(split):
            return post
        
        act = split.args[0]
        axis = split.attrs.axis
        if axis < 0:
            axis += len(act.checked_type.shape)
        ios = [int(dim) for dim in split.attrs.indices_or_sections]
        ios.append(act.checked_type.shape[axis])

        begin = 0 if post.index == 0 else ios[post.index - 1]
        end = ios[post.index]

        sliced_act = tvm.relay.strided_slice(act, (begin,), (end,), axes=(axis,))
        return sliced_act

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

class RemoveRedundantTake(DFPatternCallback):
    def __init__(self, rewrite_once=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.indices = is_constant()
        self.pattern = is_op("take")(self.input_tensor, self.indices)

    def callback(self, pre, post, node_map):
        act = node_map[self.input_tensor][0]
        act_shape = list(act.checked_type.shape)
        indices = node_map[self.indices][0].data.numpy().item()
        axis = post.attrs.axis

        if act_shape[int(axis)] == 1 and indices == 0:
            newshape = act_shape
            del newshape[int(axis)]

            out = tvm.relay.reshape(act, newshape=newshape)
            return out

        return post


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

class DecomposeEinsum(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = is_tuple(None)

        self.pattern = is_op('einsum')(self.act)

    def callback(self, pre, post, node_map):
        equation = str(post.attrs.equation)
        if equation == "bct,bcs->bts":
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            result = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=True, transpose_b=False)
            return result
        elif equation == "bts,bcs->bct":
            assert len(node_map[self.act][0]) == 2
            srcB = node_map[self.act][0][0]
            srcA = node_map[self.act][0][1]

            result = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=True)
            return result
        else:
            assert False, f"TVM einsum decomposition does not support {equation} yet."

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

class LowerAdaptiveAvgPool(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()
        self.pattern = is_op('nn.adaptive_avg_pool2d')(wildcard())

    def callback(self, pre, post, node_map):
        input_shape = [int(dim) for dim in post.args[0].checked_type.shape]
        output_shape = [int(dim) for dim in post.checked_type.shape]

        assert post.attrs.layout == "NCHW"
        assert input_shape[-1] == input_shape[-2], "Only support same factor of the input for H and W"
        assert output_shape[-1] == output_shape[-2], "Only support same factor of the output for H and W"

        input_size = input_shape[-1]
        output_size = output_shape[-1]

        stride = input_size // output_size
        kernel = input_size - (output_size - 1) * stride
        padding = 0

        return tvm.relay.nn.avg_pool2d(
            post.args[0],
            pool_size=kernel,
            strides=stride,
            padding=padding,
        )


class LowerAdaptiveMaxPool(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()
        self.pattern = is_op('nn.adaptive_max_pool2d')(wildcard())

    def callback(self, pre, post, node_map):
        input_shape = [int(dim) for dim in post.args[0].checked_type.shape]
        output_shape = [int(dim) for dim in post.checked_type.shape]

        assert post.attrs.layout == "NCHW"
        assert input_shape[-1] == input_shape[-2], "Only support same factor of the input for H and W"
        assert output_shape[-1] == output_shape[-2], "Only support same factor of the output for H and W"

        input_size = input_shape[-1]
        output_size = output_shape[-1]

        stride = input_size // output_size
        kernel = input_size - (output_size - 1) * stride
        padding = 0

        return tvm.relay.nn.max_pool2d(
            post.args[0],
            pool_size=kernel,
            strides=stride,
            padding=padding,
        )


class LowerSqueezeToReshape(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()

        self.pattern = is_op('squeeze')(wildcard())

    def callback(self, pre, post, node_map):
        return tvm.relay.reshape(post.args[0], newshape=post.checked_type.shape)


class CStridedSliceToRStridedSlice(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()
        self.pattern = is_op('strided_slice')(wildcard())

    def callback(self, pre, post, node_map):
        strides = [int(stride) for stride in post.attrs.strides]
        begin = [int(begin) for begin in post.attrs.begin]
        end = [int(end) for end in post.attrs.end]

        is_cslice = all([stride == 1 for stride in strides[:-1]]) and strides[-1] != 1
        if not is_cslice:
            return post

        strides[-1], strides[-2] = strides[-2], strides[-1]
        begin[-1], begin[-2] = begin[-2], begin[-1]
        end[-1], end[-2] = end[-2], end[-1]

        taxes = list(range(len(strides)))
        taxes[-1], taxes[-2] = taxes[-2], taxes[-1]
        t = tvm.relay.transpose(post.args[0], axes=taxes)
        rslice = tvm.relay.strided_slice(t, begin=begin, end=end, strides=strides)
        trslice = tvm.relay.transpose(rslice, axes=taxes)
        return trslice
        
class PopulateStridedSliceAxes(DFPatternCallback):
    def __init__(self, rewrite_once=True):
        super().__init__()
        self.input_tensor = wildcard()

        self.pattern = is_op('strided_slice')(wildcard())

    def callback(self, pre, post, node_map):
        if post.attrs.axes is not None:
            return post

        post = run_infer_type(post)
        input_shape = [int(dim) for dim in post.args[0].checked_type.shape]
        output_shape = [int(dim) for dim in post.checked_type.shape]

        begin = [int(dim) for dim in post.attrs.begin]
        end = [int(dim) for dim in post.attrs.end]
        stride = [int(dim) for dim in post.attrs.strides]

        act = post.args[0]
        for dim, (in_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
            if in_dim != out_dim:
                if post.attrs.slice_mode == "size":
                    final_stride = None
                    final_end = (begin[dim] + end[dim], )
                else:
                    final_stride = (stride[dim],)
                    final_end = (end[dim], )
                act = tvm.relay.strided_slice(act, begin=(begin[dim],), end=final_end, strides=final_stride,axes=(dim,), slice_mode="end")

        return act

        
class PopulateTransposeAxes(DFPatternCallback):
    def __init__(self, rewrite_once=True):
        super().__init__()
        self.input_tensor = wildcard()

        self.pattern = is_op('transpose')(wildcard())

    def callback(self, pre, post, node_map):
        if post.attrs.axes is not None:
            return post

        post = run_infer_type(post)
        transpose_dims = len(post.checked_type.shape)
        last_dim = -transpose_dims - 1
        taxes = list(range(-1, last_dim, -1))

        return tvm.relay.transpose(post.args[0], axes=taxes)


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


class AllowUnsupportedOps(ExprMutator):
    def __init__(self):
        super().__init__()

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op):
            if call.op.get_attr("target.buda") is None:
                def _func_wrapper(expr):
                    return True
                tvm.ir.register_op_attr(call.op.name, "target.buda", _func_wrapper)

        return super().visit_call(call)

class ConvertExpandDimsToReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.pattern = is_op('expand_dims')(self.act)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        axis = int(post.attrs.axis)
        num_new_axes = int(post.attrs.num_newaxis)

        if not isinstance(act, tvm.relay.expr.Var) and act.op.name == "reshape":
            target_shape = list(act.attrs.newshape)
        else:
            target_shape = list(act.checked_type.shape)

        for i in range(num_new_axes):
            target_shape.insert(axis, 1)

        return tvm.relay.reshape(act, newshape=target_shape)

def run_relay_compile_passes(relay_module, print_all=False):

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.RemoveUnusedFunctions()])(relay_module)
    logger.trace("After RemoveUnusedFunctions")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.ToBasicBlockNormalForm()])(relay_module)
    logger.trace("After ToBasicBlockNormalForm")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.Legalize()])(relay_module)
    logger.trace("After Legalize")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.SimplifyInference()])(relay_module)
    logger.trace("After SimplifyInference")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.DynamicToStatic()])(relay_module)
    logger.trace("After DynamicToStatic")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.EliminateCommonSubexpr()])(relay_module)
    logger.trace("After EliminateCommonSubexpr")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.SimplifyExpr()])(relay_module)
    logger.trace("After SimplifyExpr")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CombineParallelConv2D(3)])(relay_module)
    logger.trace("After CombineParallelConv2D")
    logger.trace(relay_module.functions)

    # relay_module = tvm.transform.Sequential([transform.CombineParallelDense(3)])(relay_module)
    # if print_all:
    # logger.trace("After CombineParallelDense")
    # logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CombineParallelBatchMatmul(3)])(relay_module)
    logger.trace("After CombineParallelBatchMatmul")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.FoldConstant()])(relay_module)
    logger.trace("After FoldConstant")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.FoldScaleAxis()])(relay_module)
    logger.trace("After FoldScaleAxis")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CanonicalizeCast()])(relay_module)
    logger.trace("After CanonicalizeCast")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CanonicalizeOps()])(relay_module)
    logger.trace("After CanonicalizeOps")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.FoldConstant()])(relay_module)
    logger.trace("After FoldConstant")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.Inline()])(relay_module)
    logger.trace("After Inline")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.FoldConstant()])(relay_module)
    logger.trace("After FoldConstant")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.CanonicalizeOps()])(relay_module)
    logger.trace("After CanonicalizeOps")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    return relay_module


def run_buda_compile_passes(relay_module, print_all=False):
    relay_module = tvm.transform.Sequential([transform.DecomposeVariance()])(relay_module)
    logger.trace("After DecomposeVariance")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeEinsum(), relay_module["main"])
    logger.trace("After DecomposeEinsum")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(LowerSplitToStridedSlice(), relay_module["main"])
    logger.trace("After LowerSplitToStridedSlice")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DenseWeightTranspose(), relay_module["main"])
    logger.trace("After DenseWeightTranspose")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposePower(), relay_module["main"])
    logger.trace("After DecomposePower")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeNegative(), relay_module["main"])
    logger.trace("After DecomposeNegative")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeRsqrt(), relay_module["main"])
    logger.trace("After DecomposeRsqrt")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(InvertDivide(), relay_module["main"])
    logger.trace("After InvertDivide")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(ExplicateTranspose(), relay_module["main"])
    logger.trace("After ExplicateTranspose")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(ExplicateHSliceTranspose(), relay_module["main"])
    logger.trace("After ExplicateHSliceTranspose")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeConv1DToConv2D(), relay_module["main"])
    logger.trace("After DecomposeConv1DtoConv2D")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(ReformatTFConv2d(), relay_module["main"])
    logger.trace("After ReformatTFConv2d")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.InferType()])(relay_module)
    logger.trace("After InferType")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeMultiAxisTranspose(), relay_module["main"])
    logger.trace("After DecomposeMultiAxisTranspose")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(EstimateWhere(), relay_module["main"])
    logger.trace("After EstimateWhere")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(LowerAdaptiveAvgPool(), relay_module["main"])
    logger.trace("After LowerAdaptiveAvgPool")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(LowerAdaptiveMaxPool(), relay_module["main"])
    logger.trace("After LowerAdaptiveMaxPool")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(LowerSqueezeToReshape(), relay_module["main"])
    logger.trace("After LowerSqueezeToReshape")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(PopulateTransposeAxes(), relay_module["main"])
    logger.trace("After PopulateTransposeAxes")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(PopulateStridedSliceAxes(), relay_module["main"])
    logger.trace("After PopulateStridedSliceAxes")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(ConvertExpandDimsToReshape(), relay_module["main"])
    logger.trace("After ConvertExpandDimsToReshape")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeMultiAxisMean(), relay_module["main"])
    logger.trace("After DecomposeMultiAxisMean")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(RemoveRedundantTake(), relay_module["main"])
    logger.trace("After RemoveRedundantTake")
    logger.trace(relay_module.functions)

    

    return relay_module


def reconstruct_ops_for_buda(mod):
    print_all = False

    logger.trace("reconstruct_ops_for_buda:: At Entry")
    logger.trace(mod.functions)
    mod["main"] = rewrite(ReconstructPyTorchGelu(), mod["main"])

    logger.trace("After ReconstructPyTorchGelu")
    logger.trace(mod.functions)
    mod["main"] = rewrite(ReconstructTFGelu(), mod["main"])

    logger.trace("After ReconstructTFGelu")
    logger.trace(mod.functions)
    mod = tvm.transform.Sequential([transform.InferType()])(mod)

    logger.trace("After InferType")
    logger.trace(mod.functions)
    mod["main"] = rewrite(ReconstructTFLayerNorm(), mod["main"])

    logger.trace("After ReconstructTFLayerNorm")
    logger.trace(mod.functions)
    mod["main"] = rewrite(ReconstructPyTorchLayerNorm(), mod["main"])

    logger.trace("After ReconstructPyTorchLayerNorm")
    logger.trace(mod.functions)

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

        logger.trace("Before Compiling")
        logger.trace(relay_module.functions)

        relay_module = run_relay_compile_passes(relay_module)
        compiled_relay_module = run_buda_compile_passes(relay_module)

    return compiled_relay_module, params


def partition_for_buda(mod, allow_unsupported=False):
    with tvm.transform.PassContext(opt_level=5):
        logger.trace("partition_for_buda:: At Entry")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.InferType()])(mod)
        logger.trace("After InferType")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.MergeComposite(pattern_table())])(mod)
        logger.trace("After MergeComposite")
        logger.trace(mod.functions)

        if allow_unsupported:
            mod["main"] = AllowUnsupportedOps().visit(mod["main"])
            logger.trace("After AllowUnsupportedOps")
            logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.FoldConstant()])(mod)
        logger.trace("After FoldConstant")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.AnnotateTarget("buda")])(mod)
        logger.trace("After AnnotateTarget")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.MergeCompilerRegions()])(mod)
        logger.trace("After MergeCompilerRegions")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.PartitionGraph(bind_constants=True)])(mod)
        logger.trace("After PartitionGraph")
        logger.trace(mod.functions)

        assert len(mod.global_var_map_) == 2, mod["main"]
        if not isinstance(mod["main"].body, tvm.relay.expr.Tuple):
            main_body_call_node = [mod["main"].body]
        else:
            main_body_call_node = mod["main"].body

        for item in main_body_call_node:
            if isinstance(item, tvm.relay.expr.Call):
                assert isinstance(item.op, tvm.ir.expr.GlobalVar), mod["main"]
                assert item.op in mod.global_var_map_.values(), mod["main"]

        constant_updator = UpdateConstants()
        rewrite(constant_updator, mod[mod.get_global_vars()[1]])
        params = constant_updator.params
        
    return mod, params