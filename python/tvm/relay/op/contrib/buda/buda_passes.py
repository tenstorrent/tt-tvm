import tvm

from tvm.relay import transform
from ....dataflow_pattern import wildcard, is_op
from tvm._ffi.base import TVMError

import numpy as np
import math
import numpy as np
from tvm.relay.dataflow_pattern import *

from loguru import logger
from .utils import *
from tvm.relay.op import _make

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

        if node_map[self.pattern][0].op.name == "nn.conv2d" and node_map[self.conv2d][0].attrs.data_layout == "NHWC" and node_map[self.conv2d][0].attrs.kernel_layout in ["HWIO", "HWOI"]:

            weight = node_map[self.conv2d][0].args[1]
            if node_map[self.conv2d][0].attrs.kernel_layout == "HWOI":
                channel_first_weight = tvm.relay.transpose(weight, axes=[2, 3, 0, 1])
            else:
                channel_first_weight = tvm.relay.transpose(weight, axes=[3, 2, 0, 1])

            new_conv2d = tvm.relay.op.nn.conv2d(
                act,
                channel_first_weight,
                strides=post.attrs.strides,
                padding=post.attrs.padding,
                groups=post.attrs.groups,
                channels=post.attrs.channels,
                kernel_size=post.attrs.kernel_size,
                data_layout="NHWC",
                kernel_layout="OIHW",
            )
            return new_conv2d
        elif node_map[self.pattern][0].op.name == "nn.global_max_pool2d" and node_map[self.globalmax_pool2d][0].attrs.layout == "NHWC":
            raise NotImplementedError
        elif node_map[self.pattern][0].op.name == "nn.global_avg_pool2d" and node_map[self.globalavg_pool2d][0].attrs.layout == "NHWC":
            raise NotImplementedError
        elif node_map[self.pattern][0].op.name == "nn.adaptive_max_pool2d" and node_map[self.adaptivemax_pool2d][0].attrs.layout == "NHWC":
            raise NotImplementedError
        elif node_map[self.pattern][0].op.name == "nn.adaptive_avg_pool2d" and node_map[self.adaptiveavg_pool2d][0].attrs.layout == "NHWC":
            raise NotImplementedError
        else:
            return post

class ResolveConvChannels(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.act = wildcard()
        self.weight = wildcard()
        
        t1 = is_op("transpose")(self.act)
        t2 = is_op("transpose")(t1)
        self.conv = is_op("nn.conv2d")(t2, self.weight) | is_op("nn.conv2d_transpose")(t2, self.weight) | is_op("nn.max_pool2d")(t2) | is_op("nn.avg_pool2d")(t2)
        t3 = is_op("transpose")(self.conv)
        self.pattern = is_op("transpose")(t3)
        
    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        weight = node_map[self.weight][0] if self.weight in node_map else None
        conv = node_map[self.conv][0]

        pool = False
        if conv.op.name == "nn.conv2d":
            op = tvm.relay.nn.conv2d
        elif conv.op.name == "nn.conv2d_transpose":
            op = tvm.relay.nn.conv2d_transpose
        elif conv.op.name == "nn.max_pool2d":
            op = tvm.relay.nn.max_pool2d
            pool = True
        else:
            op = tvm.relay.nn.avg_pool2d
            pool = True
        
        conv_layout = conv.attrs.data_layout if not pool else conv.attrs.layout
        
        data_layout = "NHWC" if conv_layout == "NCHW" else "NCHW"
        
        if not pool:
            return op(
                act,
                weight,
                strides=conv.attrs.strides,
                padding=conv.attrs.padding,
                groups=conv.attrs.groups,
                channels=conv.attrs.channels,
                kernel_size=conv.attrs.kernel_size,
                data_layout=data_layout,
                kernel_layout=conv.attrs.kernel_layout,
            )
        else:
            return op(
                act,
                pool_size=conv.attrs.pool_size,
                strides=conv.attrs.strides,
                padding=conv.attrs.padding,
                layout=data_layout,
                ceil_mode=conv.attrs.ceil_mode,
            )

class DecomposeDynamicResize2d(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True)
        self.input = wildcard()
        self.input2 = wildcard()
        self.input3 = wildcard()
        self.resize = is_op('dyn.image.resize2d')(self.input, self.input2, self.input3)
        self.pattern = wildcard()(self.resize) | wildcard()(wildcard(), self.resize) | wildcard()(self.resize, wildcard())

    def callback(self, pre, post, node_map):
        act = node_map[self.input][0]
        roi = tuple(node_map[self.input3][0].data.numpy())
        resize = node_map[self.resize][0]
        consumer = node_map[self.pattern][0]

        supported_ops = [
            "add",
            "subtract",
            "multiply",
        ]

        assert consumer.op.name in supported_ops, f"Do not support shape finding for op {consumer.op.name}"
        
        new_shape = list(consumer.checked_type.shape)
        new_resize = tvm.relay.image.resize2d(
            act,
            size=new_shape[-2:],
            roi=roi,
            layout="NCHW",
            method=resize.attrs.method,
            coordinate_transformation_mode=resize.attrs.coordinate_transformation_mode,
            rounding_method=resize.attrs.rounding_method,
            cubic_alpha=resize.attrs.cubic_alpha,
            cubic_exclude=resize.attrs.cubic_exclude,
            extrapolation_value=resize.attrs.extrapolation_value,
            out_dtype='float32',
        )

        args = list(consumer.args)
        resize_idx = args.index(resize)        
        args[resize_idx] = new_resize
        
        return tvm.relay.expr.Call(consumer.op, args)

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
        if post.attrs.axis == None:
            reduce_axes = [x for x in range(len(list(post.args[0].checked_type.shape)))]
        else:
            reduce_axes = list(post.attrs.axis)
        if len(reduce_axes) == 1:
            return post

        acts = node_map[self.act][0]
        keepdims = bool(post.attrs.keepdims)
        output_shape = list(pre.checked_type.shape)

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

        if len(transpose_axes) == 2:
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
        if post.attrs.axis == None:
            reduce_axes = [x for x in range(len(list(post.args[0].checked_type.shape)))]
        else:
            reduce_axes = list(post.attrs.axis)
        if len(reduce_axes) == 1:
            return post

        acts = node_map[self.act][0]

        keepdims = bool(post.attrs.keepdims)
        output_shape = list(pre.checked_type.shape)

        for axis in reduce_axes:
            acts = tvm.relay.mean(acts, axis=int(axis), keepdims=True)
        
        if keepdims == False:
            acts = tvm.relay.reshape(acts, newshape=output_shape)
        return acts    

class DecomposeMultiAxisSum(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.sum = is_op("sum")(self.act)
        self.pattern = self.sum

    def callback(self, pre, post, node_map):
        reduce_axes = list(post.attrs.axis)
        if len(reduce_axes) == 1:
            return post

        acts = node_map[self.act][0]

        keepdims = bool(post.attrs.keepdims)
        output_shape = list(pre.checked_type.shape)

        for axis in reduce_axes:
            acts = tvm.relay.sum(acts, axis=int(axis), keepdims=True)

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
        inp_shape = list(pre.args[0].checked_type.shape)
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
            act = node_map[self.act][0]
            weight = node_map[self.weight][0]

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
    """
    Lifting head split of the KQV matmul above linear/dense layer.
    
    This is done by hoisting the split above the linear/dense layer, and then
    doing the split on weights and bias of the linear/dense layer. 
    """
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        # Linear layer
        self.act = wildcard()
        self.weight = wildcard()
        self.bias = wildcard()
        self.dense = is_op("nn.dense")(self.act, self.weight)

        # I) Bias with reshape first variant
        self.reshape1 = is_op("reshape")(self.dense)
        self.add1 = is_op("add")(self.reshape1, self.bias)
        self.split1 = is_op('split')(self.add1)
        self.pattern1 = self.split1

        # II) Bias with add first variant
        self.add2 = is_op("add")(self.dense, self.bias)
        self.reshape2 = is_op("reshape")(self.add2)
        self.split2 = is_op('split')(self.reshape2)
        self.pattern2 = self.split2
        
        # III) No bias variant
        self.reshape3 = is_op("reshape")(self.dense)
        self.split3 = is_op('split')(self.reshape3)
        self.pattern3 = self.split3

        # Recognize any of the three patterns
        self.pattern = self.pattern1 | self.pattern2 | self.pattern3

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        assert pre.op.name == "split", "Split should be last op in pattern matcher"

        # Determine if bias is present or not based on pattern
        if self.pattern1.match(post) or self.pattern2.match(post):
            has_bias = True
        elif self.pattern3.match(post):
            has_bias = False
        else:
            assert False, "Invalid pattern match case, shouldn't happen"

        # Linear/Dense attributes
        weight_shape = pre_node_map[self.weight][0].checked_type.shape
        
        # Split attributes
        split_op_axis = pre.attrs.axis
        split_op_indices_or_sections = pre.attrs.indices_or_sections

        # Shape of split producer
        pre_split_shape = pre.args[0].checked_type.shape

        # If split is defined by number of sections, compute the indices
        if isinstance(split_op_indices_or_sections, tvm.tir.expr.IntImm):
            total_len = int(pre_split_shape[split_op_axis])
            section_len = total_len // int(split_op_indices_or_sections)
            split_op_indices_or_sections = list(range(section_len, total_len, section_len))
        else:
            split_op_indices_or_sections = [int(ios) for ios in split_op_indices_or_sections]
        split_op_indices_or_sections_num = len(split_op_indices_or_sections) + 1

        # Define new reshape shape for fractured dense paths
        newshape = list(pre_split_shape)
        newshape[split_op_axis] = -1

        # Weight should be transposed in dense/linear layer. Therefore, if split
        # is done along the last axis (C dim), we need to do the split over the
        # first one (R dim) instead
        if split_op_axis == len(pre_split_shape) - 1 or split_op_axis == -1:
            head_split_reshape_before_split_is_valid = bool(pre_split_shape[split_op_axis] == weight_shape[0])
            head_split_reshape_after_split_is_valid = bool(pre_split_shape[-1] * pre_split_shape[-2] == weight_shape[0])
            split_op_axis = 0

            assert head_split_reshape_before_split_is_valid or head_split_reshape_after_split_is_valid, "Invalid (not covered) split case for linear split uplift"
            
            # Use split op selection instead of indices for proper weights cut
            if head_split_reshape_after_split_is_valid:
                split_op_indices_or_sections = split_op_indices_or_sections_num

        # Split weights and biases
        weight = node_map[self.weight][0]
        split_weights = tvm.relay.split(weight, split_op_indices_or_sections, split_op_axis)
        bias = node_map[self.bias][0] if has_bias else None
        split_biases = tvm.relay.split(bias, split_op_indices_or_sections, -1) if has_bias else None

        # Defined by number of splits, create fractured dense paths with splitted
        # weights and biases (if applied). For example, this means that instead of 
        # having one dense layer, if there is a split of 3, we will have 3 new dense
        # layers with 1/3 of the weights and biases (if applied) each. At the end,
        # the outputs of these 3 dense layers will be concatenated to provide valid
        # KQV split results
        outputs = []
        act = node_map[self.act][0]
        for i in range(split_op_indices_or_sections_num):
            single_path_output = tvm.relay.nn.dense(act, tvm.relay.TupleGetItem(split_weights.tuple_value, i))
            single_path_output = tvm.relay.add(single_path_output, tvm.relay.TupleGetItem(split_biases.tuple_value, i)) if has_bias else single_path_output
            single_path_output = tvm.relay.reshape(single_path_output, newshape=newshape)
            outputs.append(single_path_output)
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
        act_shape = pre.tuple_value().op.args[0].checked_type.shape
        axis = split.attrs.axis
        if axis < 0:
            axis += len(act_shape)

        if isinstance(split.attrs.indices_or_sections, tvm.tir.expr.IntImm):
            sections = int(split.attrs.indices_or_sections)
            total_length = int(act_shape[axis])
            ios = list(range(total_length//sections, total_length, total_length//sections))
        else:
            ios = [int(dim) for dim in split.attrs.indices_or_sections]
        ios.append(act_shape[axis])

        begin = 0 if post.index == 0 else ios[post.index - 1]
        end = ios[post.index]

        # Check if this strided slice does nothing. If so just return act
        if end - begin == act.checked_type.shape[axis]:
            return act
        
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
        act_shape = list(pre.args[0].checked_type.shape)
        axis = pre.attrs.axis

        # Skip removal of takes which contain dynamic shapes
        if any([isinstance(dim, tvm.tir.expr.Any) for dim in act_shape]):
            return post

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

class PopulateReduceAxes(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__()
        self.pattern = is_op('sum')(wildcard())

    def callback(self, pre, post, node_map):
        if pre.attrs.axis is not None:
            return post

        ndims = len(pre.args[0].checked_type.shape)
        raxes = list(range(0, ndims, 1))

        return tvm.relay.sum(post.args[0], axis=raxes, keepdims=pre.attrs.keepdims)

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

class LowerCopyToNOP(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once, require_type=require_type)
        self.pattern = is_op("copy")(wildcard())
        

    def callback(self, pre, post, node_map):
        result = tvm.relay.identity(post.args[0])
        return post.args[0]

class ArgmaxAndMaxReconstruct(DFPatternCallback):
    def __init__(self, rewrite_once=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.softmax = is_op("nn.softmax")(self.input_tensor)
        
        self.argmax = is_op("argmax")(self.softmax)
        self.copy1 = is_op("copy")(self.argmax)
        self.copy2 = is_op("copy")(self.copy1)
        self.indices = is_constant()
        self.add = is_op("add")(self.indices, self.argmax)

        self.reshape = is_op("reshape")(self.softmax)
        self.take = is_op("take")(self.reshape, self.add)
        self.copy3 = is_op("copy")(self.take)
        self.copy4 = is_op("copy")(self.copy3)

        self.pattern = is_tuple([wildcard(), self.copy2, self.copy4])

    def callback(self, pre, post, node_map):
        act = node_map[self.input_tensor][0]
        softmax = tvm.relay.nn.softmax(act, node_map[self.softmax][0].attrs.axis)
        argmax = tvm.relay.argmax(
            softmax,
            axis=node_map[self.argmax][0].attrs.axis,
            keepdims=True,
            exclude=node_map[self.argmax][0].attrs.exclude,
            select_last_index=node_map[self.argmax][0].attrs.select_last_index,
        )
        argmax = tvm.relay.squeeze(argmax, axis=node_map[self.argmax][0].attrs.axis)
        maximum = tvm.relay.max(
            softmax,
            axis=node_map[self.argmax][0].attrs.axis,
            keepdims=True,
            exclude=node_map[self.argmax][0].attrs.exclude,
        )
        maximum = tvm.relay.squeeze(maximum, axis=node_map[self.argmax][0].attrs.axis)
        return tvm.relay.Tuple([post.fields[0], argmax, maximum])

class LowerTakeToStridedSlice(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.indices = is_constant()
        self.pattern = is_op("take")(self.input_tensor, self.indices)

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        
        # Skip removal of takes which contain dynamic shapes
        if any([isinstance(dim, tvm.tir.expr.Any) for dim in pre.checked_type.shape]):
            return post

        act = node_map[self.input_tensor][0]
        try:
            act_shape = list(pre_node_map[self.input_tensor][0].checked_type.shape)
        except ValueError as e:
            act_shape = list(pre_node_map[self.input_tensor][0].attrs.newshape)

        # If shape is not fully known, return
        for dim in act_shape:
            if isinstance(dim, tvm.tir.Any):
                return post

        try:
            indices = node_map[self.indices][0].data.numpy().flatten()
            start_value = indices[0]
            # Make sure indices are continuously ascending with stride of 1
            for idx, val in enumerate(indices):
                if start_value + idx != val:
                    raise ValueError
        except ValueError as v:
            return post

        axis = pre.attrs.axis
        if len(indices) == 1 and indices[0] == -1:
            index = act_shape[int(axis)] - 1
            strided_slice = tvm.relay.strided_slice(act, begin=(index, ), end=(index + 1,), strides=(1, ), axes=(axis, ))
        else:
            strided_slice = tvm.relay.strided_slice(act, begin=(indices[0], ), end=(indices[-1] + 1,), strides=(1, ), axes=(axis, ))

        reshape = tvm.relay.reshape(strided_slice, newshape=pre.checked_type.shape)
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
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        act = node_map[self.input_tensor][0]
        input_shape = list(pre_node_map[self.input_tensor][0].checked_type.shape)

        argmax_op = node_map[self.argmax][0]
        if argmax_op.op.name != "argmax":
            return post

        argmax_axis = int(argmax_op.attrs.axis[0])
        if argmax_axis != -1 and int(input_shape[0]) == 1:
            return post

        reshape_op = node_map[self.reshape][0]
        new_shape = list(reshape_op.attrs.newshape)
        if not (len(new_shape) == 1 and (int(new_shape[0]) == -1 or int(new_shape[0]) == 144512)):
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
            return post
        
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
        zero = tvm.relay.const(0.0, dtype=pre.checked_type.dtype)
        one = tvm.relay.const(1.0, dtype=pre.checked_type.dtype)
        inverse_causal_mask = tvm.relay.where(causal_mask, zero, one)
        
        value = post.args[2]
        mask = tvm.relay.multiply(inverse_causal_mask, value)

        act = post.args[1]
        return tvm.relay.add(act, mask)

class CastWhereConditionToBool(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)

        self.pattern = is_op("where")(wildcard(), wildcard(), wildcard()) 

    def callback(self, pre, post, node_map):
        cond = tvm.relay.cast(pre.args[0], "bool")
        return tvm.relay.where(cond, pre.args[1], pre.args[2])


class DecomposeNegative(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.in_a = wildcard()

        self.pattern = is_op('negative')(self.in_a)

    def callback(self, pre, post, node_map):
        negative_one = tvm.relay.const(-1.0, dtype=pre.checked_type.dtype)
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
            out = tvm.relay.reshape(out, newshape=pre.checked_type.shape)

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

        elif match_einsum_pattern("ibnd,jbnd->ijbn", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert len(srcA_shape) == 4 and len(srcB_shape) == 4
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(A, axes=[1, 2, 0, 3]) # bnid
            transpose_srcB = tvm.relay.transpose(B, axes=[1, 2, 3, 0]) # bdm
            new_shape_srcA = [int(srcA_shape[1]) * int(srcA_shape[2]), int(srcA_shape[0]), int(srcA_shape[3]),] # b*n id
            new_shape_srcB = [int(srcB_shape[1]) * int(srcB_shape[2]), int(srcB_shape[3]), int(srcB_shape[0]),] # b*n dj
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA) 
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB) 

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)
            result = tvm.relay.reshape(result, newshape=[int(srcA_shape[1]), int(srcA_shape[2]), int(srcA_shape[0]), int(srcB_shape[0]),]) #bnij

            return tvm.relay.transpose(result, axes=[2, 3, 0, 1]) # ijbn

        elif match_einsum_pattern("ijbn,jbnd->ibnd", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert len(srcA_shape) == 4 and len(srcB_shape) == 4
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(A, axes=[2, 3, 0, 1]) # bnij
            transpose_srcB = tvm.relay.transpose(B, axes=[1, 2, 0, 3]) # bnjd
            new_shape_srcA = [int(srcA_shape[2]) * int(srcA_shape[3]), int(srcA_shape[0]), int(srcA_shape[1]),] # b*n ij
            new_shape_srcB = [int(srcB_shape[1]) * int(srcB_shape[2]), int(srcB_shape[0]), int(srcB_shape[3]),] # b*n jd
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)
            result = tvm.relay.reshape(result, newshape=[int(srcA_shape[2]), int(srcA_shape[3]), int(srcA_shape[0]), int(srcB_shape[3]),]) #bnid

            return tvm.relay.transpose(result, axes=[2, 0, 1, 3]) # ijbn

        elif match_einsum_pattern("jbki,jfki->jkbf", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert len(srcA_shape) == 4 and len(srcB_shape) == 4
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(A, axes=[0, 2, 1, 3]) # jkbi
            transpose_srcB = tvm.relay.transpose(B, axes=[0, 2, 3, 1]) # jkif
            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[2]), int(srcA_shape[1]), int(srcA_shape[3]),] # j*k bi
            new_shape_srcB = [int(srcB_shape[0]) * int(srcB_shape[2]), int(srcB_shape[3]), int(srcB_shape[1]),] # j*k if
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)
            result = tvm.relay.reshape(result, newshape=[int(srcA_shape[0]), int(srcA_shape[2]), int(srcA_shape[1]), int(srcB_shape[1]),]) #jkbf

            return result

        elif match_einsum_pattern("f,bfde->fbde", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert len(srcA_shape) == 1 and len(srcB_shape) == 4
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            reshape_srcA = tvm.relay.reshape(A, newshape=[1, int(srcA_shape[0]), 1, 1]) # 1 f 1 1
            result = tvm.relay.multiply(reshape_srcA, B) # b f d e
            
            return tvm.relay.transpose(result, axes=[1, 0, 2, 3]) # fbde

        elif match_einsum_pattern("jikd,jkgi->jkdg", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert len(srcA_shape) == 4 and len(srcB_shape) == 4
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(A, axes=[0, 2, 3, 1]) # jkdi
            transpose_srcB = tvm.relay.transpose(B, axes=[0, 1, 3, 2]) # jkig
            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[2]), int(srcA_shape[3]), int(srcA_shape[1]),] # j*k bi
            new_shape_srcB = [int(srcB_shape[0]) * int(srcB_shape[1]), int(srcB_shape[3]), int(srcB_shape[2]),] # j*k if
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)
            result = tvm.relay.reshape(result, newshape=[int(srcA_shape[0]), int(srcA_shape[2]), int(srcA_shape[3]), int(srcB_shape[2]),]) #jkbf

            return result

        elif match_einsum_pattern("b i d, b j d -> b i j", equation):
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            result = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=True)
            return result
        elif match_einsum_pattern("b i j, b j d -> b i d", equation):
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]
            result = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=False)
            return result
        elif match_einsum_pattern("abcg,gf->abcf", equation):
            srcA = node_map[self.act][0][0]  # abcg
            srcB = node_map[self.act][0][1]  # gf
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == 4 and len(srcB_shape) == 2
            assert srcA_shape[-1] == srcB_shape[-2]

            srcA = tvm.relay.reshape(srcA, newshape=(srcA_shape[-4] * srcA_shape[-3], srcA_shape[-2], srcA_shape[-1]))  # a*b cg
            srcB = tvm.relay.reshape(srcB, newshape=(1, srcB_shape[-2], srcB_shape[-1]))  # 1gf
            res = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=False)  # a*b cf
            res = tvm.relay.reshape(res, newshape=(srcA_shape[-4], srcA_shape[-3], srcA_shape[-2], srcB_shape[-1]))  # abcf

            # import torch
            # a = torch.rand((1, 1, 32, 32))
            # b = torch.rand((32, 64))
            # torch_res = torch.einsum("abcg,gf->abcf", a, b)

            # from tvm.relay.frontend.common import infer_shape
            # from tvm.relay.frontend.common import infer_value
            # from tvm.relay.frontend.common import analysis
            # tvm_res = infer_value(res, {'args_0': tvm.nd.array(a), 'Const': tvm.nd.array(b)})

            # assert np.allclose(torch_res.numpy(), tvm_res.numpy())

            return res
        elif match_einsum_pattern("af,cfe->ace", equation):
            srcA = node_map[self.act][0][0]  # af
            srcB = node_map[self.act][0][1]  # cfe
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == 2 and len(srcB_shape) == 3
            assert srcA_shape[-1] == srcB_shape[-2]

            srcA = tvm.relay.reshape(srcA, newshape=(1, srcA_shape[-2], srcA_shape[-1])) # 1af
            bmm = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=False)  # cae
            res = tvm.relay.transpose(bmm, axes=[1, 0, 2]) # ace

            # import torch
            # a = torch.rand((32, 64))
            # b = torch.rand((1, 64, 32))
            # torch_res = torch.einsum("af,cfe->ace", a, b)

            # from tvm.relay.frontend.common import infer_shape
            # from tvm.relay.frontend.common import infer_value
            # from tvm.relay.frontend.common import analysis

            # a = a.unsqueeze(0)
            # b = b.transpose(1, 2)
            # tvm_res = infer_value(res, {'args_0': tvm.nd.array(a), 'args_1': tvm.nd.array(b)})

            # assert np.allclose(torch_res.numpy(), tvm_res.numpy())

            return res
        elif match_einsum_pattern("abcd,->abcd", equation):
            srcA = node_map[self.act][0][0]  # abcd
            srcB = node_map[self.act][0][1]  #
            
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
                        
            res = tvm.relay.multiply(srcA, srcB)
            
            # import torch
            # from tvm.relay.frontend.common import infer_value
            # a = torch.rand((1, 1, 768))
            # b = torch.rand((1, 1))
            # infer_value(res, {'args_0': tvm.nd.array(a), 'args_1': tvm.nd.array(b)})
            
            # a_val = infer_value(pre.args[0][0], {'args_0': tvm.nd.array(a), 'args_1': tvm.nd.array(b)})
            # b_val = infer_value(pre.args[0][1], {'args_2': tvm.nd.array(b)})
            
            # a_valt = torch.ones((1, 12, 1, 1))
            # b_valt = b.squeeze()
            # torch.einsum("abcd,->abcd", a_valt, b_valt)
                        
            return res
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
        output_shape = [int(dim) for dim in pre.checked_type.shape]

        assert post.attrs.layout == "NCHW"

        stride = [in_size // out_size for in_size, out_size in zip(input_shape[-2:], output_shape[-2:])]
        kernel = [in_size - (out_size - 1) * stride for in_size, out_size, stride in zip(input_shape[-2:], output_shape[-2:], stride)]
        padding = 0

        return tvm.relay.nn.avg_pool2d(
            post.args[0],
            pool_size=kernel,
            strides=stride,
            padding=padding,
            count_include_pad=True,
        )


class LowerAdaptiveMaxPool(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()
        self.pattern = is_op('nn.adaptive_max_pool2d')(wildcard())

    def callback(self, pre, post, node_map):
        input_shape = [int(dim) for dim in post.args[0].checked_type.shape]
        output_shape = [int(dim) for dim in pre.checked_type.shape]

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
        # Skip removal of squeeze which contain dynamic shapes
        if any([isinstance(dim, tvm.tir.expr.Any) for dim in pre.checked_type.shape]):
            return post
        
        return tvm.relay.reshape(post.args[0], newshape=pre.checked_type.shape)

class TransposePad(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.pattern = is_op('nn.pad')(wildcard(), is_constant())

    def callback(self, pre, post, node_map): 
        pad_width = [[int(pad) for pad in dim] for dim in post.attrs.pad_width] 
        non_zero_dims = [idx for idx, pad in enumerate(pad_width) if pad != [0, 0]] 
        assert len(non_zero_dims) <= 2

        pad_mode = post.attrs.pad_mode
        arg = post.args[0]
        arg = tvm.relay.nn.pad(arg, pad_width=pad_width, pad_value=post.args[1], pad_mode=pad_mode)
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
                    if len(stride) == len(input_shape):
                        final_stride = (stride[dim],) 
                    else:
                        assert len(stride) == 1
                        final_stride = (stride[0],) # Stride can be a length 1 list
                    final_end = (end[dim], )
                act = tvm.relay.strided_slice(act, begin=(begin[dim],), end=final_end, strides=final_stride,axes=(dim,), slice_mode="end")

        return act

class ConvertAddToBiasAddAfterConv2d(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.input = wildcard()
        self.weight = wildcard()
        self.bias = wildcard()
        self.act = is_op('nn.conv2d')(self.input, self.weight) | is_op('nn.conv2d_transpose')(self.input, self.weight)
        self.pattern = is_op('add')(self.act, self.bias)

    def callback(self, pre, post, node_map):
        bias = node_map[self.bias][0]
        act = node_map[self.act][0]
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        # Check if it's either a bias in form of var or a constant, or evan a reshape
        # which follows bias.
        if not (isinstance(bias, (tvm.relay.expr.Var, tvm.relay.expr.Constant)) \
            or bias.op.name == "reshape" and isinstance(bias.args[0], (tvm.relay.expr.Var, tvm.relay.expr.Constant))):
            return post

        # Skip reshape if follows bias
        if not isinstance(bias, (tvm.relay.expr.Var, tvm.relay.expr.Constant)) and isinstance(bias.args[0], (tvm.relay.expr.Var, tvm.relay.expr.Constant)) and bias.op.name == "reshape":
            bias = bias.args[0]
            bias_shape = list(pre_node_map[self.bias][0].args[0].checked_type.shape)
        else:
            bias_shape = list(pre_node_map[self.bias][0].checked_type.shape)

        if act.attrs.data_layout == "NHWC":
            single_dim = True
            for i in bias_shape[:-1]: single_dim = single_dim and i == 1
            if single_dim:
                bias = tvm.relay.reshape(bias, [bias_shape[-1]])
            return tvm.relay.nn.bias_add(act, bias, axis=-1)
        elif act.attrs.data_layout == "NCHW":
            single_dim = True
            for i in bias_shape[:-3] + bias_shape[-2:]: single_dim = single_dim and i == 1
            if single_dim and len(bias_shape) >= 3:
                bias = tvm.relay.reshape(bias, [bias_shape[-3]])
            return tvm.relay.nn.bias_add(act, bias)
        else:
            raise NotImplementedError(f"Unhandled data layout: {act.attrs.data_layout}")
    
class ConvertAddToBiasAddAfterConv2dTFWithChannelFirst(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.bias = wildcard()
        
        self.act = wildcard()
        self.t_act1 = is_op("transpose")(self.act)
        self.t_act2 = is_op("transpose")(self.t_act1)
        
        self.weight = wildcard()
        self.t_weight1 = is_op("transpose")(self.weight)
        self.t_weight2 = is_op("transpose")(self.t_weight1)
        self.t_weight3 = is_op("transpose")(self.t_weight2)
        
        self.conv = is_op('nn.conv2d')(self.t_act2, self.t_weight3)
        self.t_conv1 = is_op('transpose')(self.conv)
        self.t_conv2 = is_op('transpose')(self.t_conv1)
        
        self.add = is_op('add')(self.t_conv2, self.bias)
        
        self.relu = is_op('nn.relu')(self.add)
        self.t_relu1 = is_op('transpose')(self.relu)
        self.t_relu2 = is_op('transpose')(self.t_relu1)
        
        self.pattern = self.t_relu2
        
    def callback(self, pre, post, node_map):
        bias = node_map[self.bias][0]
        conv_act = node_map[self.conv][0]

        bias_add = tvm.relay.nn.bias_add(conv_act, bias)
        relu = tvm.relay.nn.relu(bias_add)
        
        return relu
    
class RemoveRedundantTranposesBetwenAvgPoolAndFlatteningReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.avg_pool = is_op('nn.avg_pool2d')(self.act)
        self.t_ap1 = is_op('transpose')(self.avg_pool)
        self.t_ap2 = is_op('transpose')(self.t_ap1)
        self.reshape = is_op('reshape')(self.t_ap2)
        
        self.pattern = self.reshape

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]

        avg_pool = node_map[self.avg_pool][0]
        t1 = node_map[self.t_ap1][0]
        t2 = node_map[self.t_ap2][0]
        flatten_reshape = node_map[self.reshape][0]

        avg_pool_shape = list(avg_pool.checked_type.shape)
        t1_shape = list(t1.checked_type.shape)
        t2_shape = list(t2.checked_type.shape)
        flatten_reshape_shape = list(flatten_reshape.checked_type.shape)

        if not (avg_pool_shape == t1_shape and avg_pool_shape[-1] == t2_shape[-3] and flatten_reshape_shape == [1, avg_pool_shape[-1] * avg_pool_shape[-2] * avg_pool_shape[-3]]):
            return post

        new_avg_pool = tvm.relay.nn.avg_pool2d(
            act,
            pool_size=avg_pool.attrs['pool_size'],
            strides=avg_pool.attrs['strides'],
            dilation=avg_pool.attrs['dilation'],
            padding=avg_pool.attrs['padding'],
            layout=avg_pool.attrs['layout'],
            out_layout=avg_pool.attrs['out_layout'],
            ceil_mode=avg_pool.attrs['ceil_mode'],
            count_include_pad=avg_pool.attrs['count_include_pad'],
        )
        flatten = tvm.relay.reshape(new_avg_pool, newshape=flatten_reshape_shape)

        return flatten


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
        if axis < 0:
            axis += 1 + len(pre.args[0].checked_type.shape)
        num_new_axes = int(pre.attrs.num_newaxis)
        
        if not isinstance(pre.args[0], tvm.relay.expr.Var) and pre.args[0].op.name == "reshape":
            target_shape = list(pre.args[0].attrs.newshape)
        else:
            target_shape = list(pre.args[0].checked_type.shape)

        # Cannot handle dynamic shapes
        for dim in target_shape:
            if isinstance(dim, tvm.tir.expr.Any):
                return post

        for i in range(num_new_axes):
            target_shape.insert(axis, 1)

        return tvm.relay.reshape(act, newshape=target_shape)


class DecomposeRepeat(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.pattern = is_op("repeat")(wildcard())
    
    def callback(self, pre, post, node_map):
        axis = int(post.attrs.axis)
        num_repeats = int(post.attrs.repeats)
        input_shape = list(pre.args[0].checked_type.shape)
        assert input_shape[axis] == 1, "Cannot decompose repeat to broadcast when input dim != 1"
        output_shape = input_shape
        output_shape[axis] *= num_repeats

        result = tvm.relay.broadcast_to(post.args[0], output_shape)
        return result


class ConvertGlobalAvgPool2dtoAvgPool2d(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.pattern = is_op("nn.global_avg_pool2d")(wildcard())
    
    def callback(self, pre, post, node_map):
        strides = (1, 1)
        pool_size = pre.args[0].checked_type.shape[2:]
        layout = post.attrs.layout
        act = post.args[0]

        avg_pool2d = tvm.relay.op.nn.avg_pool2d(
            act,
            pool_size=pool_size,
            strides=strides,
            layout=layout,
        )
        return avg_pool2d

class ConvertUpsampleToResize2d(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.pattern = is_op("nn.upsampling")(wildcard())
    
    def callback(self, pre, post, node_map):
        method = pre.attrs.method
        align_corners = pre.attrs.align_corners
        assert pre.attrs.layout == "NCHW", "Only support NCHW layout for upsample2d"

        target_shape = pre.checked_type.shape[-2:]

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        return tvm.relay.image.resize2d(
            post.args[0],
            size=target_shape,
            layout="NCHW",
            method=method,
            coordinate_transformation_mode=coord_trans,
            cubic_alpha=-0.75,
        )

class DecomposeMultiIndexAdvIndex(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.pattern = is_op("adv_index")(wildcard())
    
    def callback(self, pre, post, node_map):
        if len(pre.args[0].fields) == 2:
            return pre

        data = pre.args[0].fields[0]
        dims_to_drop = len(pre.args[0].fields) - 2
        assert all([int(dim) == 1 for dim in data.checked_type.shape[:dims_to_drop]]), "Dim to drop needs to be singleton"
        squeeze = tvm.relay.op.reshape(data, data.checked_type.shape[dims_to_drop:])
        index = tvm.relay.op.adv_index((squeeze, pre.args[0].fields[-1]))
        unsqueeze = tvm.relay.op.reshape(index, pre.checked_type.shape)

        return unsqueeze

class DecomposeErf(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.pattern = is_op("erf")(self.act)
    
    def callback(self, pre, post, node_map):
        act = post.args[0]
        act_sign = tvm.relay.op.sign(act)
        act_abs = tvm.relay.op.abs(act)

        # constants
        a1 = tvm.relay.expr.const(0.254829592)
        a2 = tvm.relay.expr.const(-0.284496736)
        a3 = tvm.relay.expr.const(1.421413741)
        a4 = tvm.relay.expr.const(-1.453152027)
        a5 = tvm.relay.expr.const(1.061405429)
        p = tvm.relay.expr.const(0.3275911)
        one = tvm.relay.expr.const(1.0)
        minus_one = tvm.relay.expr.const(-1.0)

        t = one / (one + p * act_abs)
        y = one - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * tvm.relay.op.exp((minus_one * act_abs) * act_abs)
        res = act_sign * y # erf(-x) = -erf(x)

        return res


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
        one = node_map[self.one][0].data.numpy()
        half = node_map[self.half][0].data.numpy()
        sqrt_half = node_map[self.sqrt_half][0]

        # Relay graph may use sqrt(1/2) outright, or take the recipricoral of sqrt(2)
        if isinstance(sqrt_half, tvm.relay.expr.Constant):
            sqrt_half = sqrt_half.data.numpy()
            root_two_multiplied = math.isclose(sqrt_half, 0.70710677, rel_tol=1e-6, abs_tol=1e-6)
        else:
            sqrt_half = sqrt_half.args[0].data.numpy()
            root_two_multiplied = math.isclose(sqrt_half, 1.4142135, rel_tol=1e-6, abs_tol=1e-6)

        one_added = math.isclose(one, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        half_multiplied = math.isclose(half, 0.5, rel_tol=1e-6, abs_tol=1e-6)
        
        
        if not (one_added and half_multiplied and root_two_multiplied):
            return post

        return tvm.relay.gelu(node_map[self.act][0])


class ReconstructOnnxGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.one_over_root_two = wildcard()
        self.one = is_constant()
        self.half_added = is_constant()

        times_root_two = is_op("multiply")(self.act, self.one_over_root_two)
        erf = is_op("erf")(times_root_two)
        add = is_op("add")(erf, self.one)
        mult2 = is_op("multiply")(self.act, add)
        gelu = is_op("multiply")(mult2, self.half_added)

        self.pattern = gelu

    def callback(self, pre, post, node_map):
        half_added = math.isclose(node_map[self.half_added][0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)
        one_added = math.isclose(node_map[self.one][0].data.numpy(), 1.0, rel_tol=1e-6, abs_tol=1e-6)

        sqrt_half = node_map[self.one_over_root_two][0]
        # Relay graph may use sqrt(1/2) outright, or take the recipricoral of sqrt(2)
        if isinstance(sqrt_half, tvm.relay.expr.Constant):
            sqrt_half = sqrt_half.data.numpy()
            root_two_multiplied = math.isclose(sqrt_half, 0.70710677, rel_tol=1e-6, abs_tol=1e-6)
        else:
            sqrt_half = sqrt_half.args[0].data.numpy()
            root_two_multiplied = math.isclose(sqrt_half, 1.4142135, rel_tol=1e-6, abs_tol=1e-6)

        if not (half_added and one_added and root_two_multiplied):
            return post

        return tvm.relay.gelu(node_map[self.act][0])


class ReconstructPyTorchGeluNew(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        square = is_op("multiply")(self.act, self.act)
        pow_3 = is_op("multiply")(square, self.act)
        self.c_0044715 = is_constant()
        times_const = is_op("multiply")(pow_3, self.c_0044715)
        addition = is_op("add")(self.act, times_const)
        self.root_two_over_pie = is_constant()
        times_root_2_over_pie = is_op("multiply")(addition, self.root_two_over_pie)
        tanh = is_op("tanh")(times_root_2_over_pie)
        self.one = is_constant()
        tanh_plus_one = is_op("add")(tanh, self.one)
        self.one_half = is_constant()
        divide_by_two = is_op("multiply")(self.act, self.one_half)
        new_gelu = is_op("multiply")(divide_by_two, tanh_plus_one)

        self.pattern = new_gelu

    def callback(self, pre, post, node_map):
        constnats_correct = True
        constnats_correct = constnats_correct and math.isclose(node_map[self.c_0044715][0].data.numpy(), 0.044715, rel_tol=1e-6, abs_tol=1e-6)
        constnats_correct = constnats_correct and math.isclose(node_map[self.root_two_over_pie][0].data.numpy(), math.sqrt(2.0 / math.pi), rel_tol=1e-6, abs_tol=1e-6)
        constnats_correct = constnats_correct and math.isclose(node_map[self.one][0].data.numpy(), 1, rel_tol=1e-6, abs_tol=1e-6)
        constnats_correct = constnats_correct and math.isclose(node_map[self.one_half][0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)

        if not constnats_correct:
            return post

        return tvm.relay.gelu(node_map[self.act][0], approximate="tanh")


class ReconstructPyTorchGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.one_over_root_two = is_constant()
        self.half_multiplied = is_constant()
        self.half_added = is_constant()

        times_root_two = is_op("multiply")(self.act, self.one_over_root_two)
        erf = is_op("erf")(times_root_two)
        times_half = is_op("multiply")(erf, self.half_multiplied)
        add = is_op("add")(self.half_added, times_half)
        gelu = is_op("multiply")(self.act, add)

        self.pattern = gelu

    def callback(self, pre, post, node_map):
        half_added = math.isclose(node_map[self.half_added][0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)
        half_multiplied = math.isclose(node_map[self.half_multiplied][0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)
        root_two_multiplied = math.isclose(node_map[self.one_over_root_two][0].data.numpy(), 0.70710677, rel_tol=1e-6, abs_tol=1e-6)

        if not (half_added and half_multiplied and root_two_multiplied):
            return post

        return tvm.relay.gelu(node_map[self.act][0])


class ReconstructJaxGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.sqrt_root = wildcard()
        self.one = is_constant()
        self.two = is_constant()

        reciprocal_root = is_op("reciprocal")(self.sqrt_root)
        times_root_two = is_op("multiply")(self.act, reciprocal_root)
        erf = is_op("erf")(times_root_two)
        add_one = is_op("add")(erf, self.one)
        times_act = is_op("multiply")(self.act, add_one)
        reciprocal_two = is_op("reciprocal")(self.two)
        times_half = is_op("multiply")(times_act, reciprocal_two)

        self.pattern = times_half

    def callback(self, pre, post, node_map):
        reciprocal_sqrt_root = math.isclose(node_map[self.sqrt_root][0].data.numpy(), 1.41421356, rel_tol=1e-6, abs_tol=1e-6)
        one_added = math.isclose(node_map[self.one][0].data.numpy(), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        reciprocal_two = math.isclose(node_map[self.two][0].data.numpy(), 2.0, rel_tol=1e-6, abs_tol=1e-6)

        if not (reciprocal_sqrt_root and one_added and reciprocal_two):
            return post

        return tvm.relay.gelu(node_map[self.act][0])


class SimplifyGroupNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.reshape0 = is_op("reshape")(self.act)
        self.mean = is_op("mean")(self.reshape0)
        self.sub = is_op("subtract")(self.reshape0, self.mean)
        self.var = is_op("variance")( self.reshape0, self.mean)
        self.add = is_op("add")(self.var, wildcard())
        self.sqrt = is_op("sqrt")(self.add)
        self.div = is_op("divide")(self.sub, self.sqrt)
        self.reshape1 = is_op("reshape")(self.div)
        self.pattern = self.reshape1

    def callback(self, pre, post, node_map):
        if len(node_map[self.reshape0][0].attrs.newshape) <= 4:
            # Dont need decomposition
            return post

        if len(node_map[self.reshape1][0].attrs.newshape) > 4:
            # Cannot decompose
            return post

        old_mean = node_map[self.mean][0]
        old_var = node_map[self.var][0]
        # Handle 5D/6D between the 2 reshapes
        arg = node_map[self.act][0]
        new_shape = node_map[self.reshape0][0].attrs.newshape
        new_shape = new_shape[:3] + [np.prod(new_shape[3:])]
        new_reshape0 = tvm.relay.reshape(arg, newshape=new_shape)
        new_mean = tvm.relay.mean(
            new_reshape0, 
            axis=[-2, -1], 
            keepdims=old_mean.attrs.keepdims,
            exclude=old_mean.attrs.exclude)
        new_sub = tvm.relay.subtract(new_reshape0, new_mean)
        new_var = _make._variance(
            new_reshape0, 
            new_mean,
            [-2, -1],
            old_var.attrs.keepdims,
            old_var.attrs.exclude,
            old_var.attrs.unbiased)
        new_add = tvm.relay.add(new_var, node_map[self.add][0].args[1])
        new_sqrt = tvm.relay.sqrt(new_add)
        new_div = tvm.relay.divide(new_sub, new_sqrt)
        new_reshape1 = tvm.relay.reshape(new_div, node_map[self.reshape1][0].attrs.newshape)

        return new_reshape1

class ReconstructPyTorchLayerNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.gamma = wildcard()
        self.beta = wildcard()
        self.eps = is_constant()

        self.mean_act = is_op("mean")(self.act)
        sub_0 = is_op("subtract")(self.act, self.mean_act)
        mul_0 = is_op("multiply")(sub_0, sub_0)
        var = is_op("mean")(mul_0)

        sum_denom = var.optional(lambda x: is_op("add")(x, self.eps))
        sub = is_op("subtract")(self.act, self.mean_act)
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

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = pre_node_map[self.act][0].checked_type.shape
        gamma_shape = list(pre_node_map[self.gamma][0].checked_type.shape)

        axis = pre_node_map[self.mean_act][0].attrs.axis
        assert len(axis) == 1, "TVM Layernorm only supports single dim"
        if axis[0] >= 0:
            layernorm_axis = int(axis[0] - len(act_shape))
        else:
            layernorm_axis = int(axis[0])

        if layernorm_axis != -1:
            return post

        if np.prod(gamma_shape) != act_shape[layernorm_axis]:
            return post

        if np.prod(pre_node_map[self.beta][0].checked_type.shape) != act_shape[layernorm_axis]:
            return post

        return tvm.relay.layernorm(act, gamma, beta, eps, layernorm_axis)


class ReconstructTFLayerNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.gamma = wildcard()
        self.beta = wildcard()
        self.eps = is_constant()

        self.mean_act = is_op("mean")(self.act)
        sub_0 = is_op("subtract")(self.act, self.mean_act)
        mul_0 = is_op("multiply")(sub_0, sub_0)
        var = is_op("mean")(mul_0)

        sum_denom = var.optional(lambda x: is_op("add")(x, self.eps))
        denom = is_op("sqrt")(sum_denom)
        recp = is_op("reciprocal")(denom)

        weight = is_op("multiply")(self.gamma, recp)
        mean_part = is_op("multiply")(self.mean_act, weight)
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

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = pre_node_map[self.act][0].checked_type.shape
        gamma_shape = pre_node_map[self.gamma][0].checked_type.shape
        beta_shape = pre_node_map[self.beta][0].checked_type.shape

        if len(gamma_shape) > 1 and sum([1 if int(x) != 1 else 0 for x in list(gamma_shape)]) == 1:
            # Count the number of dims thats not 1
            gamma_shape = (np.prod([int(x) for x in gamma_shape]),)
            gamma = tvm.relay.reshape(gamma, newshape=gamma_shape)
        else:
            assert len(gamma_shape) == 1, "TVM Layernorm only supports single dim"

        if len(beta_shape) > 1 and sum([1 if int(x) != 1 else 0 for x in list(beta_shape)]) == 1:
            # Count the number of dims thats not 1
            beta_shape = (np.prod([int(x) for x in beta_shape]),)
            beta = tvm.relay.reshape(beta, newshape=gamma_shape)
        else:
            assert len(beta_shape) == 1, "TVM Layernorm only supports single dim"

        axis = pre_node_map[self.mean_act][0].attrs.axis
        assert len(axis) == 1, "TVM Layernorm only supports single dim"
        if axis[0] >= 0:
            layernorm_axis = int(axis[0] - len(act_shape))
        else:
            layernorm_axis = int(axis[0])

        if layernorm_axis != -1:
            return post

        if np.prod(gamma_shape) != act_shape[layernorm_axis]:
            return post

        if np.prod(pre_node_map[self.beta][0].checked_type.shape) != act_shape[layernorm_axis]:
            return post

        return tvm.relay.layernorm(act, gamma, beta, eps, layernorm_axis)


class ReconstructJaxLayerNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        
        # Calculate mean and var
        ## Mean2
        mean2_act_mult = is_op("multiply")(self.act, self.act)
        mean2_act_sum = is_op("sum")(mean2_act_mult)
        mean2_act_reshape = is_op("reshape")(mean2_act_sum)
        mean2_dim_shape_const = is_constant() # self.axis_val
        mean2_div = is_op("reciprocal")(mean2_dim_shape_const)
        mean2 = is_op("multiply")(mean2_act_reshape, mean2_div)
        
        ## Mean
        mean_act_sum = is_op("sum")(self.act)
        mean_act_reshape = is_op("reshape")(mean_act_sum)
        self.axis_val = is_constant()
        mean_div = is_op("reciprocal")(self.axis_val)
        self.mean = is_op("multiply")(mean_act_reshape, mean_div)
        abs_mean = is_op("multiply")(self.mean, self.mean)
        
        ## Var
        mean2_sub_abs_mean = is_op("subtract")(mean2, abs_mean)
        zero_const = is_constant()
        self.var = is_op("maximum")(zero_const, mean2_sub_abs_mean)
        
        # Normalize
        ## Activation subtract mean
        mean_prep_transpose = is_op("transpose")(self.mean)
        mean_prep_reshape = is_op("reshape")(mean_prep_transpose)
        y_act_sub_mean = is_op("subtract")(self.act, mean_prep_reshape)
        
        ## Activation scale
        var_prep_transpose = is_op("transpose")(self.var)
        var_prep_reshape = is_op("reshape")(var_prep_transpose)
        self.eps = is_constant()
        var_add_epsilon = is_op("add")(var_prep_reshape, self.eps)
        mul_sqrt = is_op("sqrt")(var_add_epsilon)
        mul_reciprocal = is_op("reciprocal")(mul_sqrt)
        self.gamma = is_constant()
        scale_mul_mul = is_op("multiply")(self.gamma, mul_reciprocal)
        y_scale_mul = is_op("multiply")(scale_mul_mul, y_act_sub_mean)
        
        ## Bias add
        self.beta = is_constant()
        y_bias_add = is_op("add")(y_scale_mul, self.beta)
        
        # Set pattern
        self.pattern = y_bias_add

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        gamma = node_map[self.gamma][0]
        beta = node_map[self.beta][0]

        try:
            eps = node_map[self.eps][0].data.numpy().item()
        except TVMError: # Does not have epsilon addition
            eps = 0

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = pre_node_map[self.act][0].checked_type.shape
        gamma_shape = pre_node_map[self.gamma][0].checked_type.shape
        beta_shape = pre_node_map[self.beta][0].checked_type.shape

        if len(gamma_shape) > 1 and sum([1 if int(x) != 1 else 0 for x in list(gamma_shape)]) == 1:
            # Count the number of dims thats not 1
            gamma_shape = (np.prod([int(x) for x in gamma_shape]),)
            gamma = tvm.relay.reshape(gamma, newshape=gamma_shape)
        else:
            assert len(gamma_shape) == 1, "TVM Layernorm only supports single dim"

        if len(beta_shape) > 1 and sum([1 if int(x) != 1 else 0 for x in list(beta_shape)]) == 1:
            # Count the number of dims thats not 1
            beta_shape = (np.prod([int(x) for x in beta_shape]),)
            beta = tvm.relay.reshape(beta, newshape=gamma_shape)
        else:
            assert len(beta_shape) == 1, "TVM Layernorm only supports single dim"

        axis = -1
        axis_val = int(pre_node_map[self.axis_val][0].data.numpy())
        for i, v in reversed(list(enumerate(act_shape))):
            if v == axis_val:
                axis = i - len(act_shape)
                break
        
        if axis >= 0:
            layernorm_axis = int(axis - len(act_shape))
        else:
            layernorm_axis = int(axis)

        if layernorm_axis != -1:
            return post

        if np.prod(gamma_shape) != act_shape[layernorm_axis]:
            return post

        if np.prod(pre_node_map[self.beta][0].checked_type.shape) != act_shape[layernorm_axis]:
            return post

        return tvm.relay.layernorm(act, gamma, beta, eps, layernorm_axis)

    
class RepositionQNormScalarMultiplier(DFPatternCallback):
    """
    Reposition Q norm which follows QK matmul for self-attention in order to 
    avoid unsupported reshape and transpose ops (initially spot in Bert implemented 
    in Jax).
    """
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.q_bias = is_constant()
        self.q_bias_add = is_op("add")(wildcard(), self.q_bias)
        self.split_head_reshape = is_op("reshape")(self.q_bias_add)

        self.q_norm_last_dim_shape = is_constant()
        self.q_norm_reciprocal = is_op("reciprocal")(self.q_norm_last_dim_shape)
        
        self.scalar_multiplier = is_op("multiply")(self.split_head_reshape, self.q_norm_reciprocal)
        self.multiplier_transpose = is_op("transpose")(self.scalar_multiplier)
        self.multiplier_reshape = is_op("reshape")(self.multiplier_transpose)
        self.k_hidden_states = wildcard()
        self.batch_matmul = is_op("nn.batch_matmul")(self.multiplier_reshape, self.k_hidden_states)
        
        self.pattern = self.batch_matmul

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        
        q_bias_add = node_map[self.q_bias_add][0]
        q_norm_reciprocal = node_map[self.q_norm_reciprocal][0]
        k_hidden_states = node_map[self.k_hidden_states][0]
        
        split_head_reshape_attribute = pre_node_map[self.split_head_reshape][0].attrs.newshape
        transpose_reshape_multiplier_attribute = pre_node_map[self.multiplier_transpose][0].attrs.axes
        reshape_transposed_multiplier_attribute = pre_node_map[self.multiplier_reshape][0].attrs.newshape
        repositioned_batch_matmul_transpose_a = pre_node_map[self.batch_matmul][0].attrs.transpose_a
        repositioned_batch_matmul_transpose_b = pre_node_map[self.batch_matmul][0].attrs.transpose_b
        
        bias_add_norm_const_multiply = tvm.relay.multiply(q_bias_add, q_norm_reciprocal)
        reshape_multiplier = tvm.relay.reshape(bias_add_norm_const_multiply, newshape=split_head_reshape_attribute)
        transpose_multiplier = tvm.relay.transpose(reshape_multiplier, axes=transpose_reshape_multiplier_attribute)
        reshape_transposed_multiplier = tvm.relay.reshape(transpose_multiplier, newshape=reshape_transposed_multiplier_attribute)
        batch_matmul_with_reposition = tvm.relay.nn.batch_matmul(reshape_transposed_multiplier, k_hidden_states, 
                                                                 transpose_a=repositioned_batch_matmul_transpose_a, 
                                                                 transpose_b=repositioned_batch_matmul_transpose_b)
        
        return batch_matmul_with_reposition
    
    
class ReconstructQKVMatmulToEnableFurtherHstackOverTransposeZ(DFPatternCallback):
    """
    Reconstruction of the batch matmul used to multiply QK states and V states. In sum,
    this pass transposes and reorders inputs of the batch matmul in order to create 
    more appropriate shape where hstack can be used, instead of previously existing
    transpose on Z dim (currently not supported). 
    """
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.v_bias = is_constant()
        self.v_bias_add = is_op("add")(wildcard(), self.v_bias)
        self.split_head_reshape = is_op("reshape")(self.v_bias_add)
        self.split_head_transpose_z = is_op("transpose")(self.split_head_reshape)
        self.split_head_transpose_rc = is_op("transpose")(self.split_head_transpose_z)
        self.reshape_squeeze = is_op("reshape")(self.split_head_transpose_rc)
        
        self.qk_hidden_states = wildcard()
        self.batch_matmul = is_op("nn.batch_matmul")(self.reshape_squeeze, self.qk_hidden_states)
        self.bmm_reshape = is_op("reshape")(self.batch_matmul)
        self.bmm_transpose_z = is_op("transpose")(self.bmm_reshape)
        self.bmm_transpose_rc = is_op("transpose")(self.bmm_transpose_z)
        self.final_reshape = is_op("reshape")(self.bmm_transpose_rc)
        
        self.pattern = self.final_reshape

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        
        reshape_squeeze = node_map[self.reshape_squeeze][0]
        qk_hidden_states = node_map[self.qk_hidden_states][0]
        
        orig_srcA_shape = pre_node_map[self.reshape_squeeze][0].checked_type.shape
        orig_srcB_shape = pre_node_map[self.qk_hidden_states][0].checked_type.shape
        
        if len(orig_srcA_shape) != 3 or len(orig_srcB_shape) != 3:
            logger.warning(f"Invalid shape lengths for ReconstructQKVMatmulToEnableFurtherHstackOverTransposeZ pass")
            return post
        
        transpose_v_states = tvm.relay.transpose(reshape_squeeze, axes=[0, 2, 1])
        transpose_qk_states = tvm.relay.transpose(qk_hidden_states, axes=[0, 2, 1])
        reordered_batch_matmul = tvm.relay.nn.batch_matmul(transpose_qk_states, transpose_v_states, transpose_a=False, transpose_b=False)
        new_bmm_reshape = tvm.relay.reshape(reordered_batch_matmul, newshape=[1, orig_srcA_shape[-3], orig_srcB_shape[-1], orig_srcA_shape[-2]])
        new_bmm_transpose_zr = tvm.relay.transpose(new_bmm_reshape, axes=[0, 2, 1, 3])
        new_final_reshape = tvm.relay.reshape(new_bmm_transpose_zr, newshape=[1, orig_srcB_shape[-2], orig_srcA_shape[-3] * orig_srcA_shape[-2]])
        
        return new_final_reshape


class CombineReshapes(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True)

        self.act = wildcard()
        self.rs1 = is_op("reshape")(self.act)
        self.pattern = is_op("reshape")(self.rs1)

    def callback(self, pre, post, node_map):

        act = node_map[self.act][0]
        final_shape = pre.checked_type.shape

        return tvm.relay.reshape(act, final_shape)


class DecompEinsumWithWTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act1 = wildcard()
        self.act2 = wildcard()
        self.tup = is_tuple([self.act1, self.act2])
        self.einsum = is_op("einsum")(self.tup).has_attr({"equation": "abfd,f->fabd"})
        self.transpose = is_op("transpose")(self.einsum).has_attr({"axes": [1, 0, 3, 2]})

        self.pattern = self.transpose

    def callback(self, pre, post, node_map):
        act1 = node_map[self.act1][0]
        act2 = node_map[self.act2][0]
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act2_shape = list(pre_node_map[self.act2][0].checked_type.shape)
        new_shape = [1, 1,] + act2_shape + [1,]
        act2_reshape = tvm.relay.reshape(act2, newshape=new_shape)
        mul = tvm.relay.multiply(act1, act2_reshape)
        transpose_yz = tvm.relay.transpose(mul, axes=[0,2,1,3])
        transpose_xy = tvm.relay.transpose(transpose_yz, axes=[0,1,3,2])

        return transpose_xy

class DecompWTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act1 = wildcard()
        self.act2 = wildcard()
        self.transpose_before = is_op("transpose")(self.act1).has_attr({"axes": [2, 0, 1, 3]})
        self.mul = is_op("multiply")(self.transpose_before, self.act2)
        self.final_transpose = is_op("transpose")(self.mul)
        self.pattern = self.final_transpose

    def callback(self, pre, post, node_map):
        act1 = node_map[self.act1][0]
        act2 = node_map[self.act2][0]

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act2_shape = list(pre_node_map[self.act2][0].checked_type.shape)
        new_shape = [1, 1,] + [act2_shape[0]] + [1,]
        act2_reshape = tvm.relay.reshape(act2, newshape=new_shape)
        mul = tvm.relay.multiply(act1, act2_reshape)
        transpose_yz = tvm.relay.transpose(mul, axes=[0,2,1,3])
        final_transpose_axes = list(node_map[self.final_transpose][0].attrs.axes)
        if final_transpose_axes[2:] == [2, 3]:
            return transpose_yz
        elif final_transpose_axes[2:] == [3, 2]:
            transpose_xy = tvm.relay.transpose(transpose_yz, axes=[0,1,3,2])
        else:
            return post

        return transpose_xy

class RemoveRedundantReshapeTransposeReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.reshape_1 = is_op("reshape")(self.act)
        self.transpose = is_op("transpose")(self.reshape_1,)
        self.reshape_2 = is_op("reshape")(self.transpose)
        self.pattern = self.reshape_2

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        input_shape = pre_node_map[self.act][0].checked_type.shape
        reshape2 = node_map[self.reshape_2][0]
        final_shape = reshape2.attrs.newshape

        if list(final_shape) == list(input_shape)[-2:]:
            return tvm.relay.reshape(node_map[self.act][0], newshape=final_shape)

        if list(final_shape) == [1] + list(input_shape):
            return tvm.relay.reshape(node_map[self.act][0], newshape=final_shape)

        return post
    
class AttemptRemoveStackWDim(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True)
        self.acts = wildcard()
        self.stack = is_op("stack")(self.acts)
        self.t1 = is_op("transpose")(self.stack,)
        self.reshape = is_op("reshape")(self.t1)
        self.pattern = is_op("transpose")(self.reshape)
        
    def callback(self, pre: Expr, post: Expr, node_map: tvm.ir.container.Map) -> Expr:
        acts = node_map[self.acts][0]
        stack = node_map[self.stack][0]
        t1 = node_map[self.t1][0]
        reshape = node_map[self.reshape][0]
        t2 = node_map[self.pattern][0]

        if all(len(stack.args[0].fields[i].checked_type.shape) == 3 for i in range(len(stack.args[0].fields))) \
                and len(stack.checked_type.shape) > 3 \
                and stack.checked_type.shape[-4] > 1 \
                and t2.checked_type.shape[-2] == stack.checked_type.shape[-4]:
            
            reshaped_inps = []
            for arg in stack.args[0].fields:
                newdim = 1
                for d in arg.checked_type.shape:
                    newdim *= d
                
                if arg.checked_type.shape[-2] == 1:
                    arg = tvm.relay.transpose(arg, axes=[1, 0, 2])
                    
                flattened = tvm.relay.reshape(arg, newshape=[1, 1, newdim])
                reshaped_inps.append(flattened)
            
            return tvm.relay.concatenate(reshaped_inps, -2)
        
        return super().callback(pre, post, node_map)

class ReplicatePyBudaReshapeTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.transpose_1 = is_op("transpose")(self.act)
        self.transpose_2 = is_op("transpose")(self.transpose_1,)
        self.reshape = is_op("reshape")(self.transpose_2)
        self.pattern = self.reshape

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = pre_node_map[self.act][0].checked_type.shape
        t1 = node_map[self.transpose_1][0]
        t1_axes = [int(dim) for dim in t1.attrs.axes]
        t2 = node_map[self.transpose_2][0]
        t2_axes = [int(dim) for dim in t2.attrs.axes]
        
        reshape = node_map[self.reshape][0]
        newshape = [int(dim) for dim in reshape.attrs.newshape]
        input_shape = [int(dim) for dim in t2.checked_type.shape]
        if len(input_shape) < 4:
            return post
        eos = [input_shape[0], input_shape[1] * input_shape[2], input_shape[3]]
        if t1_axes == [0, 2, 1, 3] and t2_axes == [0, 1, 3, 2] and newshape == eos:
            newshape = [1, int(act_shape[0]), eos[2], eos[1]]
            r1 = tvm.relay.reshape(node_map[self.act][0], newshape)
            t1 = tvm.relay.transpose(r1, axes=[0, 1, 3, 2])
            squeeze = tvm.relay.reshape(t1, newshape=[int(act_shape[0]), eos[1], eos[2]])
            return squeeze
        else:
            return post
class CommuteIndexPastReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.index = is_op("strided_slice")(self.act)
        self.bias = wildcard()
        self.reshape_bias = is_op("reshape")(self.bias)
        self.add = is_op("add")(self.index, self.reshape_bias)
        self.reshape0 = is_op("reshape")(self.add)
        self.transpose0 = is_op("transpose")(self.reshape0).has_attr({"axes" : [0, 1, 3, 2]})
        self.reshape1 = is_op("reshape")(self.transpose0)
        self.pattern = self.reshape1


    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        index_attrs = node_map[self.index][0].attrs
        if len(index_attrs.begin) != 1 or len(index_attrs.end) != 1:
            return post
        bias = node_map[self.bias][0]

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = list(pre_node_map[self.act][0].checked_type.shape)
        if len(act_shape) != 4:
            return post

        bias_shape = list(pre_node_map[self.bias][0].checked_type.shape)
        if len(bias_shape) != 1:
            return post

        reshape0_target = list(node_map[self.reshape0][0].attrs.newshape)
        if reshape0_target != [1, 1, index_attrs.end[0] - index_attrs.begin[0], act_shape[-2] * act_shape[-1]]:
            return post

        reshape1_target = list(node_map[self.reshape1][0].attrs.newshape)
        if reshape1_target != [1,act_shape[-2] * act_shape[-1], index_attrs.end[0] - index_attrs.begin[0]]:
            return post
    
        new_reshape0 = tvm.relay.reshape(act, newshape=[1, 1, act_shape[-3], act_shape[-2] * act_shape[-1]])
        new_transpose = tvm.relay.transpose(new_reshape0, axes=[0, 1, 3, 2])
        new_reshape1 = tvm.relay.reshape(new_transpose, newshape=[1,act_shape[-2] * act_shape[-1], act_shape[-3]])
        new_index = tvm.relay.strided_slice(new_reshape1, begin=index_attrs.begin, end=index_attrs.end, strides=index_attrs.strides, axes=[2])

        new_bias_reshape = tvm.relay.reshape(bias, newshape=[1, 1, bias_shape[0]])
        new_add = tvm.relay.add(new_index, new_bias_reshape)
        return new_add


class DecomposeScatterND(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        # ScatterND op with all inputs
        self.data = wildcard()
        self.indices = wildcard()
        self.updates = wildcard()
        self.mode = wildcard()
        self.scatter_nd = is_op("scatter_nd")(self.data, self.indices, self.updates)
        
        self.pattern = self.scatter_nd

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        
        # Arguments & Op
        data = node_map[self.data][0]
        indices = node_map[self.indices][0]
        updates = node_map[self.updates][0]
        
        # Handle only update mode (replacement instead of accumulation)
        if not pre_node_map[self.scatter_nd][0].attrs.mode == "update":
            return post
        
        # Handle only 2D tensors
        if len(pre_node_map[self.data][0].checked_type.shape) != 2 or len(pre_node_map[self.indices][0].checked_type.shape) != 2 or len(pre_node_map[self.updates][0].checked_type.shape) != 2:
            return post
        
        # Debug values
        # import torch
        # from tvm.relay.frontend.common import analysis
        # from tvm.relay.frontend.common import infer_shape
        # from tvm.relay.frontend.common import infer_value

        # analysis.free_vars(data)
        # data_shape = infer_shape(data)
        # a = torch.arange(0, np.prod(data_shape), dtype=torch.float32).view(data_shape) / 18
        # a = tvm.nd.array(a)
        # analysis.free_vars(indices)
        # indices_shape = infer_shape(indices)
        # b = torch.rand(indices_shape, dtype=torch.float32).view(indices_shape)
        # b = tvm.nd.array(b)
        
        # Scatter ND using where op
        out = tvm.relay.where(indices, updates, data)
        
        # infer_shape(out)
        # infer_value(out, {'input_act': a, 'threshold_1': b})
        
        return out
        
        
class ConvertIsNaN(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        # IsNaN op with all inputs
        self.data = wildcard()
        self.isnan = is_op("isnan")(self.data)
        
        self.pattern = self.isnan

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        
        data = pre_node_map[self.data][0]
        
        cond = tvm.relay.equal(data, tvm.relay.const(np.nan, dtype="float32"))
        where = tvm.relay.where(cond, tvm.relay.const(True), tvm.relay.const(False))
        
        return where
    
    
class RemoveRedundantBinaryStacks(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.tuple = is_tuple([self.act])
        self.stack = is_op("stack")(self.tuple)
        self.reshape = is_op("reshape")(self.stack)

        self.pattern = self.reshape
        
    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        act = node_map[self.act][0]
        stack = pre_node_map[self.stack][0]
        reshape = pre_node_map[self.reshape][0]
        
        if len(stack.args) != 1 and len(reshape.args) != 1:
            return post

        stack_field_input_shape = [int(i) for i in stack.args[0].checked_type.fields[0].shape]
        reshape_shape = [int(i) for i in reshape.attrs.newshape]

        if stack_field_input_shape != reshape_shape:
            return post
        
        # Creates duplicates
        # from tvm.relay.frontend.common import infer_type
        # return infer_type(act.fields[0])
        
        # Also, creates duplicates
        # mod = tvm.ir.IRModule.from_expr(act.fields[0])
        # mod = transform.InferType()(mod)
        # return mod['main'].body
        
        return act


class RemoveStopFusionAnnotationNodes(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.stop_fusion_node = is_op("annotation.stop_fusion")(self.act)
        
        self.pattern = self.stop_fusion_node

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        
        return act


class Enforce1DOutputForArgwhereOp(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.argwhere = is_op("argwhere")(self.act)
        self.transpose = is_op("transpose")(self.argwhere)
        
        self.pattern = self.transpose
        
    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        act = node_map[self.act][0]
        act_shape = pre_node_map[self.act][0].checked_type.shape
        transpose_shape = pre_node_map[self.transpose][0].checked_type.shape
        
        if len(act_shape) == 2 and int(act_shape[0]) == 1 and len(transpose_shape) == 2 and int(transpose_shape[0]) == 2:
            out = tvm.relay.squeeze(act, axis=[0])
            out = tvm.relay.argwhere(out)
            out = tvm.relay.transpose(out, axes=[1, 0])
            out = tvm.relay.squeeze(out, axis=[0])
            return out

        return post


class BroadcastScatterValuesToMatchIndices(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.indices = wildcard()
        self.updates = wildcard()
        self.scatter = is_op("scatter")(self.act, self.indices, self.updates)
        
        self.pattern = self.scatter
        
    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        act = node_map[self.act][0]
        indices = node_map[self.indices][0]
        updates = node_map[self.updates][0]
        updates_shape = pre_node_map[self.updates][0].checked_type.shape
        scatter = node_map[self.scatter][0]

        if len(updates_shape) == 1 and isinstance(updates_shape[0], tvm.tir.expr.IntImm) and int(updates_shape[0]) == 1:
            # Match dtype for bcast
            updates = tvm.relay.cast(updates, indices.checked_type.dtype)
            updates = tvm.relay.broadcast_to_like(updates, indices)
            # Match dtype for scatter
            updates = tvm.relay.cast(updates, act.checked_type.dtype)
            out = tvm.relay.scatter(act, indices, updates, int(scatter.attrs.axis))
            return out

        return post

class InverseMaskGen(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.zero = wildcard()
        self.mask = wildcard()
        self.mask_gen = is_op("equal")(self.mask, self.zero)
        
        self.pattern = self.mask_gen
        
    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        if not isinstance(pre_node_map[self.zero][0], tvm.relay.Constant) or pre_node_map[self.zero][0].data.numpy() != 0:
            return post

        casted = tvm.relay.cast(pre_node_map[self.mask][0], "bool")
        casted = tvm.relay.cast(casted, "int32")
        subtract =  tvm.relay.subtract(tvm.relay.const(1, "int32"), casted)
        booled = tvm.relay.cast(subtract, "bool")
        return booled
                
        

class DecomposeVariance(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.variance = is_op("variance")(wildcard(), wildcard())
        
        self.pattern = self.variance

    def callback(self, pre, post, node_map):
        mean = post.args[1]
        sub = tvm.relay.subtract(post.args[0], mean)
        mul = tvm.relay.multiply(sub, sub)
        var = tvm.relay.mean(mul, axis=post.attrs.axis, keepdims=post.attrs.keepdims, exclude=post.attrs.exclude)

        return var


class DecomposePRelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.prelu = is_op("nn.prelu")(wildcard(), wildcard())
        
        self.pattern = self.prelu

    def callback(self, pre, post, node_map):
        relu_ = tvm.relay.nn.relu(post.args[0])
        neg_one = tvm.relay.Constant(tvm.nd.array(np.array([-1,], dtype="float32")))
        neg_ = tvm.relay.multiply(post.args[0], neg_one)
        neg_relu_ = tvm.relay.nn.relu(neg_)
        mul_ = tvm.relay.multiply(post.args[1], neg_relu_)
        mul_2 = tvm.relay.multiply(mul_, neg_one)
        out = tvm.relay.add(relu_, mul_2)
        return out


class DecomposeNonZeroPadtoConcat(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.pad = is_op("nn.pad")(wildcard(), wildcard())
        
        self.pattern = self.pad

    def callback(self, pre, post, node_map):
        act = post.args[0]
        pad_width = post.attrs.pad_width
        pad_value = post.args[1].data.numpy()
        if pad_value == 0:
            return post

        pad_shape = list(act.checked_type.shape)
        pad_shape = [int(x) for x in pad_shape]
        for i, item in enumerate(pad_width):
            before = int(item[0])
            after = int(item[1])
            if before == 0 and after == 0:
                continue
    
            if before != 0:
                current_pad_shape = pad_shape.copy()
                current_pad_shape[i] = before
                const = tvm.relay.const(np.ones(current_pad_shape) * pad_value, dtype=act.checked_type.dtype)
                act = tvm.relay.concatenate([const, act], axis=i)
                pad_shape = [x + y for x, y in zip(pad_shape, current_pad_shape)]

            if after != 0:
                current_pad_shape = pad_shape.copy()
                current_pad_shape[i] = after
                const = tvm.relay.const(np.ones(current_pad_shape) * pad_value, dtype=act.checked_type.dtype)
                act = tvm.relay.concatenate([act, const], axis=i)
                pad_shape = [x + y for x, y in zip(pad_shape, current_pad_shape)]

        return act


def _get_callback_name(callback):
    if isinstance(callback, DFPatternCallback):
        return type(callback).__name__
    elif isinstance(callback, tvm.transform.Pass):
        return callback.info.name
    else:
        raise NotImplementedError(f"Type of callback ({type(callback)}) not implemented")


def _run_pattern_callback(relay_module, callback, callback_name):
    if isinstance(callback, DFPatternCallback):
        relay_module['main'] = rewrite(callback, relay_module['main'])
    elif isinstance(callback, tvm.transform.Pass):
        relay_module = tvm.transform.Sequential([callback])(relay_module)
    else:
        raise NotImplementedError(f"Type of callback ({type(callback)}) not implemented")

    logger.trace(f"After {callback_name}")
    logger.trace(relay_module.functions)

    return relay_module


def run_pattern_callbacks(relay_module, callbacks, params=None, inputs=None, target=None, framework_outputs=None, verify_cfg=None):
    
    run_verify = verify_cfg and params and inputs and target and framework_outputs and verify_cfg.verify_each_buda_pass
    if verify_cfg and verify_cfg.verify_each_buda_pass and not run_verify:
        logger.warning(f"Cannot verify relay module after buda passes because one of (params, inputs, target, golden_outputs, veirfy_cfg) is None")

    for callback in callbacks:
        callback_name = _get_callback_name(callback)
        try:
            relay_module = _run_pattern_callback(relay_module, callback, callback_name)
        except Exception as ex:
            logger.error(f"Failed on \"{callback_name}\" TVM callback")
            raise ex
        if run_verify:
            logger.trace(f"Verifying {callback_name}")
            tvm.relay.op.contrib.buda.buda.verify_tvm_compile(relay_module, params, inputs, target, framework_outputs, callback_name, verify_cfg)
    
    return relay_module


def run_buda_compile_passes(relay_module, params=None, inputs=None, target=None, framework_outputs=None, verify_cfg=None):
    return run_pattern_callbacks(
        relay_module,
        [
            ConvertLayout(),
            ResolveConvChannels(),
            DecomposeDynamicResize2d(),
            DecomposePRelu(),
            # RemoveCast(),
            DecomposeStack(),
            SimplifyGroupNorm(),
            DecomposeVariance(),
            ArgmaxAndMaxReconstruct(),
            ConvertArgmaxTakeToReduceMax(),
            AddSqueezeForArgmax(),
            DecompEinsumWithWTranspose(),
            DecompWTranspose(),
            DecomposeEinsum(),
            DecomposeLayoutTransform(),
            LiftLinearSplit(),
            LowerSplitToStridedSlice(),
            DenseWeightTranspose(),
            DecomposePower(),
            DecomposeNegative(),
            DecomposeRsqrt(),
            InvertDivide(),
            ExplicateTranspose(),
            ExplicateHSliceTranspose(),
            DecomposeConv1DToConv2D(),
            PopulateReduceAxes(),
            DecomposeMultiAxisMax(),
            DecomposeMultiAxisTranspose(),
            EstimateWhereInCausalMask(),
            CastWhereConditionToBool(),
            LowerAdaptiveAvgPool(),
            LowerAdaptiveMaxPool(),
            EnsureKeepdims(),
            LowerSqueezeToReshape(),
            PopulateTransposeAxes(),
            PopulateStridedSliceAxes(),
            ConvertExpandDimsToReshape(),
            DecomposeMultiAxisMean(),
            DecomposeMultiAxisSum(),
            DecomposeMultiAxisBroadcast(),
            RemoveRedundantTake(),
            RemoveRedundantReshape(),
            LowerCopyToNOP(),
            TransposePad(),
            DecomposeNonZeroPadtoConcat(),
            DecomposeMultiRangeTake(),
            LowerTakeToStridedSlice(),
            ConvertAddToBiasAddAfterConv2d(),
            DecomposeBatchFlatten(),
            DecomposeRepeat(),
            ConvertGlobalAvgPool2dtoAvgPool2d(),
            ConvertUpsampleToResize2d(),
            DecomposeMultiIndexAdvIndex(),
            # DecomposeErf(),
            ReconstructTFGelu(),
            ReconstructOnnxGelu(),
            ReconstructPyTorchGeluNew(),
            # ReconstructPyTorchGelu(),
            ReconstructJaxGelu(),
            # ReconstructPyTorchLayerNorm(),
            ReconstructTFLayerNorm(),
            RepositionQNormScalarMultiplier(),
            ReconstructQKVMatmulToEnableFurtherHstackOverTransposeZ(),
            CombineReshapes(),
            ReconstructJaxLayerNorm(),
            RemoveRedundantTranposesBetwenAvgPoolAndFlatteningReshape(),
            RemoveRedundantReshapeTransposeReshape(),
            ReplicatePyBudaReshapeTranspose(),
            CommuteIndexPastReshape(),
            AttemptRemoveStackWDim(),
            # RemoveRedundantBinaryStacks(),
            DecomposeScatterND(),
            ConvertIsNaN(),
            RemoveStopFusionAnnotationNodes(),
            Enforce1DOutputForArgwhereOp(),
            BroadcastScatterValuesToMatchIndices(),
            InverseMaskGen(),
        ],
        params=params,
        inputs=inputs,
        target=target,
        framework_outputs=framework_outputs,
        verify_cfg=verify_cfg
    )
