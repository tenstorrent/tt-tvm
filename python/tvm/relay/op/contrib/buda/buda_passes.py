import logging

from pkg_resources import require

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
from tvm._ffi.base import TVMError
from tvm.ir.transform import PassContext
from tvm.relay import transform
from ....dataflow_pattern import wildcard, is_op

import numpy as np
import math
import numpy as np
from tvm.relay.dataflow_pattern import *

from loguru import logger
from .utils import *

# NOTE: TVM crashes when groups != 1 or groups != input_channels
class ConvertLayout(DFPatternCallback):
    def __init__(self, ):
        super().__init__()

        self.input = wildcard()
        self.conv2d = is_op('nn.conv2d')(self.input, wildcard())
        self.max_pool2d = is_op('nn.max_pool2d')(self.input)
        self.avg_pool2d = is_op('nn.avg_pool2d')(self.input)
        self.conv2d_tran = is_op('nn.conv2d_transpose')(self.input, wildcard())
        self.globalmax_pool2d = is_op('nn.global_max_pool2d')(self.input)
        self.globalavg_pool2d = is_op('nn.global_avg_pool2d')(self.input)
        self.adaptivemax_pool2d = is_op('nn.adaptive_max_pool2d')(self.input)
        self.adaptiveavg_pool2d = is_op('nn.adaptive_avg_pool2d')(self.input)
        self.imageresize2d = is_op('image.resize2d')(self.input)

        self.pattern = (
            self.conv2d
            | self.max_pool2d
            | self.avg_pool2d
            | self.conv2d_tran
            | self.globalmax_pool2d
            | self.globalavg_pool2d
            | self.adaptivemax_pool2d
            | self.adaptiveavg_pool2d
            | self.imageresize2d
        )

    def callback(self, pre, post, node_map):
        act = node_map[self.input][0]

        if node_map[self.pattern][0].op.name == "nn.conv2d" and node_map[self.conv2d][0].attrs.data_layout == "NHWC":
            channel_first_act = tvm.relay.transpose(act, axes=[0, 3, 1, 2])
            weight = node_map[self.conv2d][0].args[1]

            assert post.attrs.channels == post.attrs.groups or post.attrs.groups == 1, \
                 "TVM only supports nn.conv2d with groups = 1 or groups = input channels"

            if post.attrs.channels == post.attrs.groups:
                channel_first_weight = tvm.relay.transpose(weight, axes=[2, 3, 0, 1])
            elif post.attrs.groups == 1:
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

        elif node_map[self.pattern][0].op.name == "nn.conv2d_transpose" and node_map[self.conv2d_tran][0].attrs.data_layout == "NHWC":
            raise NotImplementedError
            # channel_first_act = tvm.relay.transpose(act, axes=[0, 3, 1, 2])
            # weight = node_map[self.conv2d][0].args[1]
            # channel_first_weight = tvm.relay.transpose(weight, axes=[3, 2, 0, 1])
            # new_conv2d = tvm.relay.op.nn.conv2d(
            #     channel_first_act,
            #     channel_first_weight,
            #     strides=post.attrs.strides,
            #     padding=post.attrs.padding,
            #     groups=post.attrs.groups,
            #     channels=post.attrs.channels,
            #     kernel_size=post.attrs.kernel_size,
            #     data_layout="NCHW",
            #     kernel_layout="OIHW",
            # )
            # out_reshape = tvm.relay.transpose(new_conv2d, axes=[0,2,3,1])
            # return out_reshape
        elif node_map[self.pattern][0].op.name == "nn.max_pool2d" and node_map[self.max_pool2d][0].attrs.layout == "NHWC":

            channel_first_act = tvm.relay.transpose(act, axes=[0, 3, 1, 2])

            new_pool = tvm.relay.op.nn.max_pool2d(
                channel_first_act,
                pool_size=post.attrs.pool_size,
                strides=post.attrs.strides,
                padding=post.attrs.padding,
                layout="NCHW",
                ceil_mode=post.attrs.ceil_mode,
            )
            out_reshape = tvm.relay.transpose(new_pool, axes=[0,2,3,1])
            return out_reshape
        elif node_map[self.pattern][0].op.name == "nn.avg_pool2d" and node_map[self.avg_pool2d][0].attrs.layout == "NHWC":

            channel_first_act = tvm.relay.transpose(act, axes=[0, 3, 1, 2])

            new_pool = tvm.relay.op.nn.avg_pool2d(
                channel_first_act,
                pool_size=post.attrs.pool_size,
                strides=post.attrs.strides,
                padding=post.attrs.padding,
                layout="NCHW",
                ceil_mode=post.attrs.ceil_mode,
            )
            out_reshape = tvm.relay.transpose(new_pool, axes=[0,2,3,1])
            return out_reshape
        elif node_map[self.pattern][0].op.name == "nn.global_max_pool2d" and node_map[self.globalmax_pool2d][0].attrs.layout == "NHWC":
            raise NotImplementedError
        elif node_map[self.pattern][0].op.name == "nn.global_avg_pool2d" and node_map[self.globalavg_pool2d][0].attrs.layout == "NHWC":
            raise NotImplementedError
        elif node_map[self.pattern][0].op.name == "nn.adaptive_max_pool2d" and node_map[self.adaptivemax_pool2d][0].attrs.layout == "NHWC":
            raise NotImplementedError
        elif node_map[self.pattern][0].op.name == "nn.adaptive_avg_pool2d" and node_map[self.adaptiveavg_pool2d][0].attrs.layout == "NHWC":
            raise NotImplementedError
        elif node_map[self.pattern][0].op.name == "image.resize2d" and node_map[self.imageresize2d][0].attrs.layout == "NHWC":
            channel_first_act = tvm.relay.transpose(act, axes=[0, 3, 1, 2])
            new_resize2d = tvm.relay.image.resize2d(
                channel_first_act,
                size=post.attrs.size,
                roi=post.attrs.roi,
                layout="NCHW",
                method=post.attrs.method,
                coordinate_transformation_mode=post.attrs.coordinate_transformation_mode,
                rounding_method=post.attrs.rounding_method,
                cubic_alpha=post.attrs.cubic_alpha,
                cubic_exclude=post.attrs.cubic_exclude,
                extrapolation_value=post.attrs.extrapolation_value,
                out_dtype=post.attrs.out_dtype,
            )
            out_reshape = tvm.relay.transpose(new_resize2d, axes=[0,2,3,1])
            return out_reshape
        else:
            return post



class RemoveCast(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.pattern = is_op("cast")(self.act)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        return act


class DecomposeStack(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)

        # This occurs when compiling tf.stack
        self.act1 = wildcard()
        self.act2 = wildcard()
        self.exp_dims1 = is_op("expand_dims")(self.act1)
        self.exp_dims2 = is_op("expand_dims")(self.act2)
        
        self.tup = is_tuple([self.exp_dims1, self.exp_dims2])
        self.concatenate = is_op("concatenate")(self.tup)
        self.r4 = is_op("reshape")(self.concatenate)
        self.pattern = self.r4
        
    
    def callback(self, pre: Expr, post: Expr, node_map: tvm.ir.container.Map) -> Expr:
        # replacing concat op with stack, as is done in the pytorch decomposition
        act1 = node_map[self.act1][0]
        act2 = node_map[self.act2][0]
        tup = tvm.relay.Tuple([act1, act2])
        stacked = tvm.relay.stack(tup, axis=-1)
        r = tvm.relay.reshape(stacked, newshape=[0, 0, 0, -1, 1])
        output = tvm.relay.squeeze(r, axis=[4])

        return output

class DecomposeMultiAxisMax(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.max = is_op("max")(self.act)

        self.pattern = self.max

    def callback(self, pre, post, node_map):

        reduce_axes = list(post.attrs.axis)
        if len(reduce_axes) == 1:
            return post

        acts = node_map[self.act][0]
        out = node_map[self.max][0]
        keepdims = bool(post.attrs.keepdims)
        output_shape = list(out.checked_type.shape)

        for axis in reduce_axes:
            acts = tvm.relay.max(acts, axis=int(axis), keepdims=True)
        
        if keepdims == False:
            acts = tvm.relay.reshape(acts, newshape=output_shape)
        return acts   

class DecomposeMultiAxisTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.transpose = is_op("transpose")(self.act)
        self.pattern = self.transpose

    def callback(self, pre, post, node_map):

        transpose_axes = pre.attrs.axes
        act_shape = pre.args[0].checked_type.shape
        acts = node_map[self.act][0]

        if transpose_axes == None:
            return post

        changed_axis = []

        for i, axis in enumerate(transpose_axes):
            if i != axis:
                changed_axis.append([i, axis])


        assert len(changed_axis) >= 2, "Transpose op has at least 2 axes changed"
        if len(changed_axis) == 2:
            return post

        ndim = len(act_shape)
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
        super().__init__(rewrite_once=True, require_type=True)
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

class DecomposeMultiAxisBroadcast(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.broadcast_to = is_op("broadcast_to")(self.act)
        self.pattern = self.broadcast_to

    def callback(self, pre, post, node_map):
        acts = node_map[self.act][0]
        inp_shape = list(acts.checked_type.shape)
        target_shape = list(pre.attrs.shape)

        for i, (inp_dim, target_dim) in enumerate(zip(inp_shape, target_shape)):
            if inp_dim == target_dim:
                continue

            one_axis_target = target_shape[:i + 1] + inp_shape[i + 1:]
            acts = tvm.relay.broadcast_to(acts, one_axis_target)

        return acts

class DecomposeConv1DToConv2D(DFPatternCallback):
    def __init__(self): 
        super().__init__(rewrite_once=True, require_type=True)
        
        self.act = wildcard()
        self.weight = wildcard()
        self.conv_pattern = is_op("nn.conv1d")(self.act, self.weight)
        self.pattern = self.conv_pattern

    def callback(self, pre, post, node_map):
        if post.attrs.data_layout == 'NWC' and post.attrs.kernel_layout == 'WIO':
            assert False, "Conv1d from TF is not supported yet"
            # TODO: converting TF conv1d - channel-last to channel-first Conv2d
        else:
            # reshape activation and reshape weight 
            expected_output_shape = pre.checked_type.shape
            
            acts = node_map[self.act][0]
            acts_shape = list(pre.args[0].checked_type.shape)
            weights = node_map[self.weight][0]
            weights_shape = list(pre.args[1].checked_type.shape)
            
            new_acts_shape = [acts_shape[0], acts_shape[1], acts_shape[2], 1]
            new_weights_shape = [weights_shape[0], weights_shape[1], weights_shape[2], 1]
            reshaped_acts = tvm.relay.reshape(acts, newshape=new_acts_shape)
            reshaped_weights = tvm.relay.reshape(weights, newshape=new_weights_shape)

            new_conv2d = tvm.relay.op.nn.conv2d(
                reshaped_acts, 
                reshaped_weights,
                strides=[post.attrs.strides[0], 1],
                padding=[post.attrs.padding[0], 0, post.attrs.padding[1], 0],
                # (TODO arui) Since weight kernel is 1 on unsqueezed dim, dilation shouldnt matter. This is needed because we dont support different
                # dilation for each dim in pybuda conv2d.
                dilation=[post.attrs.dilation[0], post.attrs.dilation[0]],
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
        self.weight = wildcard()
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

class ReformatTFMaxpool(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        maxpool = is_op("nn.max_pool2d")(self.act,)
        self.pattern = maxpool

    def callback(self, pre, post, node_map):
        if post.attrs.layout == 'NHWC':
            # convert TF channel-last to channel-first
            act = node_map[self.act][0]

            channel_first_act = tvm.relay.transpose(act, axes=[0, 3, 1, 2])

            new_pool = tvm.relay.op.nn.max_pool2d(
                channel_first_act,
                pool_size=post.attrs.pool_size,
                strides=post.attrs.strides,
                padding=post.attrs.padding,
                layout="NCHW",
                ceil_mode=post.attrs.ceil_mode,
            )
            out_reshape = tvm.relay.transpose(new_pool, axes=[0,2,3,1])
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
        super().__init__(rewrite_once=True, require_type=True)
        act = wildcard()
        self.dense_weight = wildcard()
        self.add_bias = wildcard()
        self.dense = is_op("nn.dense")(act, self.dense_weight)
        self.add = is_op("add")(self.dense, self.add_bias)
        self.reshape = is_op("reshape")(self.add)
        self.split = is_op('split')(self.reshape)
        self.pattern = self.split

    def callback(self, pre, post, node_map):
        weight = node_map[self.dense_weight][0]
        bias = node_map[self.add_bias][0]

        if isinstance(post.attrs.indices_or_sections, tvm.tir.expr.IntImm):
            total_len = int(pre.args[0].checked_type.shape[post.attrs.axis])
            section_len = total_len // int(post.attrs.indices_or_sections)
            indices_or_sections = list(range(section_len, total_len, section_len))
        else:
            indices_or_sections = [int(ios) for ios in post.attrs.indices_or_sections]
        axis = post.attrs.axis

        output_shape = node_map[self.reshape][0].checked_type.shape
        newshape = list(output_shape)
        newshape[axis] = -1

        if (is_unsqueeze(node_map[self.reshape][0])):
            # Weight should be transposed in nn.dense, so if splitting
            # along the final output axis, split along the first weight
            if axis == len(output_shape) - 1:
                assert output_shape[axis] == weight.checked_type.shape[0]
                axis = 0

        act = node_map[self.dense][0].args[0]

        split_weights = tvm.relay.split(weight, indices_or_sections, axis)
        split_biases = tvm.relay.split(bias, indices_or_sections, -1)

        outputs = []
        for i in range(split_weights.size):
            dense_out = tvm.relay.nn.dense(act, tvm.relay.TupleGetItem(split_weights.tuple_value, i))
            add_out = tvm.relay.add(dense_out, tvm.relay.TupleGetItem(split_biases.tuple_value, i))
            reshape_out = tvm.relay.reshape(add_out, newshape=newshape)
            outputs.append(reshape_out)

        return tvm.relay.expr.Tuple(outputs)

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

        if isinstance(split.attrs.indices_or_sections, tvm.tir.expr.IntImm):
            sections = int(split.attrs.indices_or_sections)
            total_length = int(act.checked_type.shape[axis])
            ios = list(range(total_length//sections, total_length, total_length//sections))
        else:
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
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.indices = is_constant()
        self.pattern = is_op("take")(self.input_tensor, self.indices)

    def callback(self, pre, post, node_map):
        if node_map[self.indices][0].data.numpy().size == 1:
            indices = node_map[self.indices][0].data.numpy().item()
        else:
            return post

        act = node_map[self.input_tensor][0]
        act_shape = list(act.checked_type.shape)
        axis = pre.attrs.axis

        if act_shape[int(axis)] == 1 and indices == 0:
            newshape = act_shape
            del newshape[int(axis)]

            out = tvm.relay.reshape(act, newshape=newshape)
            return out

        return post

class PopulateTransposeAxes(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__()
        self.input_tensor = wildcard()

        self.pattern = is_op('transpose')(wildcard())

    def callback(self, pre, post, node_map):
        if pre.attrs.axes is not None:
            return post

        transpose_dims = len(pre.checked_type.shape)
        last_dim = -transpose_dims - 1
        taxes = list(range(-1, last_dim, -1))

        return tvm.relay.transpose(post.args[0], axes=taxes)


class RemoveRedundantReshape(DFPatternCallback):
    def __init__(self, rewrite_once=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.reshape = is_op("reshape")(self.input_tensor)
        self.pattern = self.reshape

    def callback(self, pre, post, node_map):
        act = node_map[self.input_tensor][0]
        reshape_op = node_map[self.reshape][0]
        new_shape = list(reshape_op.attrs.newshape)
        if len(new_shape) == 0:
            return act
        else:
            return post

class LowerTakeToStridedSlice(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.indices = is_constant()
        self.pattern = is_op("take")(self.input_tensor, self.indices)

    def callback(self, pre, post, node_map):
        act = node_map[self.input_tensor][0]
        
        act_shape = list(pre.args[0].checked_type.shape)

        try:
            indices = node_map[self.indices][0].data.numpy().item()
        except ValueError as v:
            assert False, "we only support take with a single index currently"
        
        axis = pre.attrs.axis
        strided_slice = tvm.relay.strided_slice(act, begin=(indices, ), end=(indices + 1,), strides=(1, ), axes=(axis, ))
        newshape = act_shape
        del newshape[int(axis)]
        reshape = tvm.relay.reshape(strided_slice, newshape=newshape)
        return reshape

class AddSqueezeForArgmax(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once, require_type=require_type)
        self.input_tensor = wildcard()
        self.pattern = is_op("argmax")(self.input_tensor)

    def callback(self, pre, post, node_map):
        if post.attrs.keepdims:
            return post
        
        inp = node_map[self.input_tensor][0]
        axis = post.attrs.axis
        new_argmax = tvm.relay.argmax(inp, axis=axis, keepdims=True,)
        result = tvm.relay.squeeze(new_argmax, axis=axis)
        return result

class ConvertArgmaxTakeToReduceMax(DFPatternCallback):
    def __init__(self, rewrite_once=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.argmax = wildcard()
        self.reshape = is_op("reshape")(self.input_tensor)
        self.const = is_constant()
        self.add = is_op("add")(self.const, self.argmax)
        self.take = is_op("take")(self.reshape, self.add)
        self.pattern = self.take

    def callback(self, pre, post, node_map):

        act = node_map[self.input_tensor][0]
        input_shape = list(act.checked_type.shape)

        argmax_op = node_map[self.argmax][0]
        if argmax_op.op.name != "argmax":
            return post

        argmax_axis = int(argmax_op.attrs.axis[0])
        if argmax_axis != -1 and int(input_shape[0]) == 1:
            return post

        reshape_op = node_map[self.reshape][0]
        new_shape = list(reshape_op.attrs.newshape)
        if not (len(new_shape) == 1 and int(new_shape[0]) == -1):
            return post

        const_data = node_map[self.const][0].data.numpy()
        arange_max = int(input_shape[-2]) * int(input_shape[-1])
        test_np = np.expand_dims(np.arange(start=0, stop=arange_max, step=int(input_shape[-1]), dtype=const_data.dtype), axis=0)
        match = np.array_equal(const_data, test_np, equal_nan=False)
        if not match:
            return post

        result = tvm.relay.max(act, axis=-1, keepdims=True)
        result = tvm.relay.squeeze(result, axis=[-1])
        return result

class DecomposeMultiRangeTake(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.indices = is_constant()
        self.pattern = is_op("take")(self.input_tensor, self.indices)

    def callback(self, pre, post, node_map):
        """ 
            The goal here is to decompose an original take op to multiple take ops where 
            the new ops, each represent a single continouos range. 
            tvm.take can have a list of arbitrary indices. 
        """
        act = node_map[self.input_tensor][0]
        try:
            indices = node_map[self.indices][0].data.numpy().item()
        except ValueError as v:
            assert False, "we only support take with a single index currently"
        
        return post

class EstimateWhereInCausalMask(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act1 = wildcard()
        self.act2 = wildcard()
        self.matmul = is_op("nn.batch_matmul")(self.act1, self.act2)
        self.reshape = is_op("reshape")(self.matmul)
        self.reciprocal = is_op("reciprocal")(is_constant())
        self.strided_slice = is_op("strided_slice")(wildcard())
        self.multiply = is_op("multiply")(self.reshape, self.reciprocal)

        self.pattern = is_op("where")(self.strided_slice | is_constant(), self.reshape | self.multiply | wildcard(), wildcard()) 

        
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
        super().__init__(rewrite_once=True, require_type=True)
        self.act = is_tuple(None)

        self.pattern = is_op('einsum')(self.act)

    def callback(self, pre, post, node_map):
        equation = str(post.attrs.equation)
        if match_einsum_pattern("bct,bcs->bts", equation):
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            result = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=True, transpose_b=False)
            return result
        elif match_einsum_pattern("bts,bcs->bct", equation):
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            result = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=True)
            return tvm.relay.transpose(result, axes=[0, 2, 1])

        elif match_einsum_pattern("bnqd,bnkd->bnqk", equation):
            assert len(node_map[self.act][0]) == 2
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcB = tvm.relay.transpose(srcB, axes=[0, 1, 3, 2,])
            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[1]), int(srcA_shape[2]), int(srcA_shape[3])]
            new_shape_srcB = [int(srcB_shape[0]) * int(srcB_shape[1]), int(srcB_shape[3]), int(srcB_shape[2])]

            reshape_srcA = tvm.relay.reshape(srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)
            return tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[1], srcA_shape[2], srcB_shape[2]])

        elif match_einsum_pattern("abc,cde->abde", equation):
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]
            srcA_shape = srcA.checked_type.shape
            srcB_shape = srcB.checked_type.shape

            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[1]), int(srcA_shape[2])] # a*b c 
            new_shape_srcB = [int(srcB_shape[0]), int(srcB_shape[1]) * int(srcB_shape[2])] # c d*e

            reshape_srcA = tvm.relay.reshape(srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(srcB, newshape=new_shape_srcB)

            # MatMul
            result = tvm.relay.nn.matmul(reshape_srcA, reshape_srcB) # a*b d*e

            # Reshape 
            reshape_result = tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[1], srcB_shape[1], srcB_shape[2]])
            return reshape_result

        elif match_einsum_pattern("ibnd,jbnd->bnij", equation):
            srcA_shape = list(pre.args[0][0].checked_type.shape)
            srcB_shape = list(pre.args[0][1].checked_type.shape)
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            if srcA_shape[1] != srcB_shape[1] and srcA_shape[1] == 1:
                # broadcast
                srcA_shape[1] = srcA_shape[1] * srcB_shape[1]
                srcA = tvm.relay.broadcast_to(srcA, srcA_shape)
            elif srcA_shape[1] != srcB_shape[1]:
                # Dont know how to handle
                assert False, f"TVM einsum decomposition does not support {equation} with srcA shape {srcA_shape}."

            transpose_srcA = tvm.relay.transpose(srcA, axes=[1, 2, 0, 3]) # bnid
            transpose_srcB = tvm.relay.transpose(srcB, axes=[1, 2, 3, 0]) # bndj
            new_shape_srcA = [int(srcA_shape[1]) * int(srcA_shape[2]), int(srcA_shape[0]), int(srcA_shape[3])]
            new_shape_srcB = [int(srcB_shape[1]) * int(srcB_shape[2]), int(srcB_shape[3]), int(srcB_shape[0])]

            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            # Batch MatMul
            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)

            # Reshape 
            reshape_result = tvm.relay.reshape(result, newshape=[srcA_shape[1], srcA_shape[2], srcA_shape[0], srcB_shape[0]])
            return reshape_result

        elif match_einsum_pattern("ibnd,snd->ibns", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(srcA, axes=[2, 0, 1, 3]) # nibd
            transpose_srcB = tvm.relay.transpose(srcB, axes=[1, 2, 0]) # nds
            new_shape_srcA = [int(srcA_shape[2]), int(srcA_shape[0]) * int(srcA_shape[1]), int(srcA_shape[3])] # n i*b d
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            # Batch MatMul
            result = tvm.relay.nn.batch_matmul(reshape_srcA, transpose_srcB, transpose_a=False, transpose_b=False) # n i*b s

            # Reshape 
            reshape_result = tvm.relay.reshape(result, newshape=[srcA_shape[2], srcA_shape[0], srcA_shape[1], srcB_shape[0]])
            reshape_result = tvm.relay.transpose(reshape_result, axes=[1, 2, 0, 3])
            return reshape_result

        elif match_einsum_pattern("ijbs,ibns->bnij", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]
            
            transpose_srcA = tvm.relay.transpose(srcA, axes=[0, 2, 1, 3,]) # ibjs
            transpose_srcB = tvm.relay.transpose(srcB, axes=[0, 1, 3, 2]) # ibsn
            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[2]), int(srcA_shape[1]), int(srcA_shape[3])] # i*b j s
            new_shape_srcB = [int(srcB_shape[0]) * int(srcB_shape[1]), int(srcB_shape[3]), int(srcB_shape[2])] # i*b s n
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)
            # MatMul
            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False) # ibjn

            # Reshape 
            reshape_result = tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[2], srcA_shape[1], srcB_shape[2]])
            reshape_result = tvm.relay.transpose(reshape_result, axes=[1, 3, 0, 2])
            return reshape_result

        elif match_einsum_pattern("ijbn->bnij", equation):
            srcA = node_map[self.act][0][0]
            return tvm.relay.transpose(srcA, axes=[2, 3, 0, 1])
        elif match_einsum_pattern("bnij,jbnd->ibnd", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcB = tvm.relay.transpose(srcB, axes=[1, 2, 0, 3]) # bnjd
            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[1]),int(srcA_shape[2]), int(srcA_shape[3])] # b*n i j
            new_shape_srcB = [int(srcB_shape[1]) * int(srcB_shape[2]), int(srcB_shape[0]), int(srcB_shape[3])]  # b*n j d
            reshape_srcA = tvm.relay.reshape(srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            if srcA_shape[-1] != srcB_shape[0] and srcB_shape[0] == 1:
                # broadcast
                new_shape_srcB[-2] = new_shape_srcB[-2] * srcA_shape[-1]
                reshape_srcB = tvm.relay.broadcast_to(reshape_srcB, new_shape_srcB)
            elif srcA_shape[-1] != srcB_shape[0]:
                # Dont know how to handle
                assert False, f"TVM einsum decomposition does not support {equation} with srcB shape {srcB_shape}."

            # MatMul
            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)

            # Reshape 
            reshape_result = tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[1], srcA_shape[2], srcB_shape[3]])
            reshape_result = tvm.relay.transpose(reshape_result, axes=[2, 0, 1, 3])
            return reshape_result

        elif match_einsum_pattern("ibnd,hnd->ibh", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcB = tvm.relay.transpose(srcB, axes=[1, 2, 0]) # ibsn
            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[1]), int(srcA_shape[2]) * int(srcA_shape[3])]
            new_shape_srcB = [int(srcB_shape[1]) * int(srcB_shape[2]), int(srcB_shape[0])]
            reshape_srcA = tvm.relay.reshape(srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            # MatMul
            result = tvm.relay.nn.matmul(reshape_srcA, reshape_srcB) # ibjn

            # Reshape 
            reshape_result = tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[1], srcB_shape[0]])
            return reshape_result

        elif match_einsum_pattern("bhd,bmd->bhmd", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert srcA_shape[0] == srcB_shape[0], f"Only support {equation} with srcA/srcB shape being the same on dim b and d"
            assert srcA_shape[-1] == srcB_shape[-1], f"Only support {equation} with srcA/srcB shape being the same on dim b and d"

            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(srcA, axes=[0, 2, 1,]) # dbh
            transpose_srcB = tvm.relay.transpose(srcB, axes=[0, 2, 1,]) # bdm
            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[2]), int(srcA_shape[1]), 1,]
            new_shape_srcB = [int(srcB_shape[0]) * int(srcB_shape[2]), 1, int(srcB_shape[1]),]
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA) # bxd h 1
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB) # bxd 1 m

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)

            # Reshape 
            reshape_result = tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[2], srcA_shape[1], srcB_shape[1]])
            reshape_result = tvm.relay.transpose(reshape_result, axes=[0, 2, 3, 1])
            return reshape_result

        elif match_einsum_pattern("ijk, qr -> ijr", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            assert len(srcA_shape) == 3 and len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            A = tvm.relay.sum(A, axis=[2], keepdims=True) # i j 1
            B = tvm.relay.sum(B, axis=[0], keepdims=True) # 1 r
            
            B = tvm.relay.reshape(B, newshape=[1, srcB_shape[1], 1]) # 1 r 1
            
            result = tvm.relay.nn.batch_matmul(A, B) # 
            return tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[1], srcB_shape[1]])

        elif match_einsum_pattern("ijk, kr -> ijr", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]
            
            assert len(srcA_shape) == 3 and len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            A = tvm.relay.reshape(A, newshape=[srcA_shape[0] * srcA_shape[1], srcA_shape[2]]) # i*j k
            result = tvm.relay.nn.matmul(A, B) # i*j r
            return tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[1], srcB_shape[1]]) # i j r

        elif match_einsum_pattern("ij, qr -> i", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            # have to sum over each axis one by one for pybuda
            B_sum = tvm.relay.sum(srcB, axis=[0])
            B_sum = tvm.relay.sum(B_sum, axis=[0])

            A_transpose = tvm.relay.transpose(srcA, axes=[1, 0])
            A_sum = tvm.relay.sum(A_transpose, axis=[0])
            out = tvm.relay.multiply(B_sum, A_sum)
            out = tvm.relay.reshape(out, newshape=post.checked_type.shape)

            return out

        elif match_einsum_pattern("ij, jk -> ik", equation) or match_einsum_pattern("ij, kr -> ir", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            A_sum = tvm.relay.sum(srcA, axis=[1], keepdims=True)
            B_sum = tvm.relay.sum(srcB, axis=[0], keepdims=True)
            return tvm.relay.nn.matmul(A_sum, B_sum)

        elif match_einsum_pattern("abc, bde -> acde", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 3, "input tensors have incorrect number of dimensions"

            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]
            
            A = tvm.relay.transpose(A, axes=[0, 2, 1])

            new_shape_A = [int(srcA_shape[0]) * int(srcA_shape[2]), int(srcA_shape[1])] # a*c b 
            new_shape_B = [int(srcB_shape[0]), int(srcB_shape[1]) * int(srcB_shape[2])] # b d*e

            reshape_A = tvm.relay.reshape(A, newshape=new_shape_A)
            reshape_B = tvm.relay.reshape(B, newshape=new_shape_B)

            # MatMul
            result = tvm.relay.nn.matmul(reshape_A, reshape_B) # a*c d*e

            # Reshape 
            reshape_result = tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[2], srcB_shape[1], srcB_shape[2]])
            return reshape_result

        elif match_einsum_pattern("abcd, dcbe -> ae", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 4, "input tensors have incorrect number of dimensions"

            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            B = tvm.relay.transpose(B, axes=[2, 1, 0, 3])

            new_shape_A = [int(srcA_shape[0]), int(srcA_shape[1]) * int(srcA_shape[2]) * int(srcA_shape[3])] # a b*c*d 
            new_shape_B = [int(srcB_shape[0]) * int(srcB_shape[1]) * int(srcB_shape[2]), int(srcB_shape[3])] # b*c*d e

            A = tvm.relay.reshape(A, newshape=new_shape_A)
            B = tvm.relay.reshape(B, newshape=new_shape_B)

            # MatMul
            result = tvm.relay.nn.matmul(A, B) # a e
            return result

        elif match_einsum_pattern("abc, bcd -> ad", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 3, "input tensors have incorrect number of dimensions"

            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            new_shape_A = [int(srcA_shape[0]), int(srcA_shape[1]) * int(srcA_shape[2])] # a b*c 
            new_shape_B = [int(srcB_shape[0]) * int(srcB_shape[1]), int(srcB_shape[2])] # b*c d

            reshape_A = tvm.relay.reshape(A, newshape=new_shape_A)
            reshape_B = tvm.relay.reshape(B, newshape=new_shape_B)

            # MatMul
            result = tvm.relay.nn.matmul(reshape_A, reshape_B) # a d
            return result

        elif match_einsum_pattern("abc, bd -> acd", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == 3 and len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            A = tvm.relay.transpose(A, axes=[0, 2, 1]) # a c b

            new_shape_A = [int(srcA_shape[0]) * int(srcA_shape[2]), int(srcA_shape[1])] # a*c b 

            A = tvm.relay.reshape(A, newshape=new_shape_A) # a*c b 

            # MatMul
            result = tvm.relay.nn.matmul(A, B) # a*c b X b d = a*c d
            return tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[2], srcB_shape[1]])

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

class DecomposeLayoutTransform(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()

        self.pattern = is_op('layout_transform')(self.input_tensor)

    def callback(self, pre, post, node_map):
        if post.attrs.src_layout == "NHWC" and post.attrs.dst_layout == "NCHW":
            act = node_map[self.input_tensor][0]

            axes = [0, 3, 1, 2]
            return tvm.relay.transpose(act, axes=axes)
        elif post.attrs.src_layout == "NCHW" and post.attrs.dst_layout == "NHWC":
            act = node_map[self.input_tensor][0]

            axes = [0, 2, 3, 1]
            return tvm.relay.transpose(act, axes=axes)
        else:
            return post


class ExplicateTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True)
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
        super().__init__(require_type=True)
        self.input_tensor = wildcard()

        self.pattern = is_op('squeeze')(wildcard())

    def callback(self, pre, post, node_map):
        return tvm.relay.reshape(post.args[0], newshape=pre.checked_type.shape)

class TransposePad(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.pattern = is_op('nn.pad')(wildcard(), is_constant())

    def callback(self, pre, post, node_map):
        pad_width = [[int(pad) for pad in dim] for dim in post.attrs.pad_width]

        transpose_axes = []

        non_zero_dims = [idx for idx, pad in enumerate(pad_width) if pad != [0, 0]]
        available_zero_dims = [idx for idx, pad in enumerate(pad_width) if pad == [0, 0] and idx >= len(pad_width) - 2]
        zdi = 0

        assert len(non_zero_dims) <= 2

        for dim in non_zero_dims:
            if dim < len(pad_width) - 2:
                permute = list(range(len(pad_width)))
                permute[dim], permute[available_zero_dims[zdi]] = permute[available_zero_dims[zdi]], permute[dim]
                pad_width[dim], pad_width[available_zero_dims[zdi]] = pad_width[available_zero_dims[zdi]], pad_width[dim]
                zdi += 1
                transpose_axes.append(permute)

        if len(transpose_axes) == 0:
            return post

        arg = post.args[0]
        for axes in transpose_axes:
            arg = tvm.relay.transpose(arg, axes=axes)

        arg = tvm.relay.nn.pad(arg, pad_width=pad_width, pad_value=post.args[1])

        for axes in transpose_axes:
            arg = tvm.relay.transpose(arg, axes=axes)

        return arg

class PopulateStridedSliceAxes(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__()
        self.input_tensor = wildcard()

        self.pattern = is_op('strided_slice')(wildcard())

    def callback(self, pre, post, node_map):
        if pre.attrs.axes is not None:
            return post

        input_shape = [int(dim) for dim in pre.args[0].checked_type.shape]
        output_shape = [int(dim) for dim in pre.checked_type.shape]

        begin = [int(dim) for dim in pre.attrs.begin]
        end = [int(dim) for dim in pre.attrs.end]
        stride = [int(dim) for dim in pre.attrs.strides]

        act = post.args[0]
        for dim, (in_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
            if in_dim != out_dim:
                if pre.attrs.slice_mode == "size":
                    final_stride = None
                    final_end = (begin[dim] + end[dim], )
                else:
                    final_stride = (stride[dim],)
                    final_end = (end[dim], )
                act = tvm.relay.strided_slice(act, begin=(begin[dim],), end=final_end, strides=final_stride,axes=(dim,), slice_mode="end")

        return act

class ConvertAddToBiasAddAfterConv2d(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.input = wildcard()
        self.weight = wildcard()
        self.bias = wildcard()
        self.act = is_op('nn.conv2d')(self.input, self.weight)
        self.reshaped_bias = is_op('reshape')(self.bias)
        self.pattern = is_op('add')(self.act, self.reshaped_bias)

    def callback(self, pre, post, node_map):
        bias = node_map[self.bias][0]
        act = node_map[self.act][0]

        return tvm.relay.nn.bias_add(act, bias)

class EnsureKeepdims(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)

        self.pattern = is_op('sum')(wildcard()) | is_op('mean')(wildcard())

    def callback(self, pre, post, node_map):
        if post.attrs.keepdims == 0:
            act = post.args[0]
            if node_map[self.pattern][0].op.name == 'sum':
                reduce_ = tvm.relay.sum(
                    act,
                    axis=post.attrs.axis,
                    keepdims=True,
                )
            else:
                reduce_ = tvm.relay.mean(
                    act,
                    axis=post.attrs.axis,
                    keepdims=True,
                )
            result = tvm.relay.squeeze(reduce_, axis=post.attrs.axis)
            return result
        else:
            return post

class DecomposeBatchFlatten(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.pattern = is_op('nn.batch_flatten')(self.act)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        input_shape = list(pre.args[0].checked_type.shape)
        target_shape = [input_shape[0]] + [math.prod(input_shape[1:])]

        return tvm.relay.reshape(act, newshape=target_shape)

class ConvertExpandDimsToReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.pattern = is_op('expand_dims')(self.act)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        axis = int(pre.attrs.axis)
        num_new_axes = int(pre.attrs.num_newaxis)

        if not isinstance(pre.args[0], tvm.relay.expr.Var) and pre.args[0].op.name == "reshape":
            target_shape = list(pre.args[0].attrs.newshape)
        else:
            target_shape = list(pre.args[0].checked_type.shape)

        for i in range(num_new_axes):
            target_shape.insert(axis, 1)

        return tvm.relay.reshape(act, newshape=target_shape)

class SkipRedundantConcatenateSlice(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.concat = is_op("concatenate")(wildcard())
        self.pattern = is_op("strided_slice")(wildcard())
        # self.pattern = self.concat

    def callback(self, pre, post, node_map):
        if not self.concat.match(post.args[0]):
            return post

        slice_shape = [int(dim) for dim in post.checked_type.shape]
        slice_start = int(post.attrs.begin[0])
        slice_end = int(post.attrs.end[0])
        slice_axis = int(post.attrs.axes[0])
        if slice_axis < 0:
            slice_axis += len(post.checked_type.shape)
        slice_strides = int(post.attrs.axes[0])

        concat_axis = post.args[0].attrs.axis
        if concat_axis < 0:
            concat_axis += len(post.args[0].checked_type.shape)

        if concat_axis != slice_axis:
            return post

        # TODO
        if len(post.args[0].args[0].fields) != 2:
            return post

        left_shape = [int(dim) for dim in post.args[0].args[0].fields[0].checked_type.shape]
        right_shape = [int(dim) for dim in post.args[0].args[0].fields[1].checked_type.shape]

        if slice_start == 0:
            if slice_shape == left_shape:
                return post.args[0].args[0].fields[0]
        else:
            if left_shape[concat_axis] == slice_start and slice_shape == right_shape:
                return post.args[0].args[0].fields[1]

        return post

class DecomposeRepeat(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.pattern = is_op("repeat")(wildcard())
    
    def callback(self, pre, post, node_map):
        axis = int(post.attrs.axis)
        num_repeats = int(post.attrs.repeats)
        input_shape = list(pre.args[0].checked_type.shape)
        assert input_shape[axis] == 1, "Cannot decompose repeat to broadcast when dim != 1"
        output_shape = input_shape
        output_shape[axis] *= num_repeats

        result = tvm.relay.broadcast_to(post.args[0], output_shape)
        return result

def run_buda_compile_passes(relay_module, print_all=False):

    relay_module["main"] = rewrite(ConvertLayout(), relay_module["main"])
    logger.trace("After ConvertLayout")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(RemoveCast(), relay_module["main"])
    logger.trace("After RemoveCast")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeStack(), relay_module["main"])
    logger.trace("After DecomposeStack")
    logger.trace(relay_module.functions)

    relay_module = tvm.transform.Sequential([transform.DecomposeVariance()])(relay_module)
    logger.trace("After DecomposeVariance")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(ConvertArgmaxTakeToReduceMax(), relay_module["main"])
    logger.trace("After ConvertArgmaxTakeToReduceMax")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(AddSqueezeForArgmax(), relay_module["main"])
    logger.trace("After AddSqueezeForArgmax")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeEinsum(), relay_module["main"])
    logger.trace("After DecomposeEinsum")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeLayoutTransform(), relay_module["main"])
    logger.trace("After DecomposeLayoutTransform")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(LiftLinearSplit(), relay_module["main"])
    logger.trace("After LiftLinearSplit")
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

    relay_module["main"] = rewrite(ExplicateTranspose(), relay_module["main"])
    logger.trace("After ExplicateTranspose")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(ExplicateHSliceTranspose(), relay_module["main"])
    logger.trace("After ExplicateHSliceTranspose")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeConv1DToConv2D(), relay_module["main"])
    logger.trace("After DecomposeConv1DtoConv2D")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeMultiAxisMax(), relay_module["main"])
    logger.trace("After DecomposeMultiAxisMax")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeMultiAxisTranspose(), relay_module["main"])
    logger.trace("After DecomposeMultiAxisTranspose")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(EstimateWhereInCausalMask(), relay_module["main"])
    logger.trace("After EstimateWhereInCausalMask")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(LowerAdaptiveAvgPool(), relay_module["main"])
    logger.trace("After LowerAdaptiveAvgPool")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(LowerAdaptiveMaxPool(), relay_module["main"])
    logger.trace("After LowerAdaptiveMaxPool")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(EnsureKeepdims(), relay_module["main"])
    logger.trace("After DecomposeReduceSum")
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

    relay_module["main"] = rewrite(DecomposeMultiAxisBroadcast(), relay_module["main"])
    logger.trace("After DecomposeMultiAxisMean")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(RemoveRedundantTake(), relay_module["main"])
    logger.trace("After RemoveRedundantTake")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(RemoveRedundantReshape(), relay_module["main"])
    logger.trace("After RemoveRedundantReshape")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(TransposePad(), relay_module["main"])
    logger.trace("After TransposePad")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeMultiRangeTake(), relay_module["main"])
    logger.trace("After DecomposeMultiRangeTake")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(LowerTakeToStridedSlice(), relay_module["main"])
    logger.trace("After LowerTakeToStridedSlice")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(ConvertAddToBiasAddAfterConv2d(), relay_module["main"])
    logger.trace("After ConvertAddToBiasAddAfterConv2d")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(SkipRedundantConcatenateSlice(), relay_module["main"])
    logger.trace("After SkipRedundantConcatenateSlice")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeBatchFlatten(), relay_module["main"])
    logger.trace("After DecomposeBatchFlatten")
    logger.trace(relay_module.functions)

    relay_module["main"] = rewrite(DecomposeRepeat(), relay_module["main"])
    logger.trace("After DecomposeRepeat")
    logger.trace(relay_module.functions)

    return relay_module
