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
from tvm.relay.op.transform import broadcast_to
from tvm.target.compilation_config import make_compilation_config
from ....dataflow_pattern import wildcard, is_op
from ..register import register_pattern_table
from .reportify import dump_graph
from .buda_passes import run_buda_compile_passes
from .relay_passes import run_relay_compile_passes
from .utils import *

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
_register_external_op_helper("nn.matmul")
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
_register_external_op_helper("abs")
_register_external_op_helper("argmax")
_register_external_op_helper("cos")
_register_external_op_helper("sin")
_register_external_op_helper("nn.pad")
_register_external_op_helper("max")
_register_external_op_helper("broadcast_to")
_register_external_op_helper("sum")
_register_external_op_helper("power")
_register_external_op_helper("ones")
_register_external_op_helper("zeros")
_register_external_op_helper("tanh")
_register_external_op_helper("scatter")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("nn.max_pool1d")
_register_external_op_helper("take")


def nn_layernorm_to_buda_layernorm():
    act = wildcard()
    return is_op("layernorm")

def dense_to_matmul():
    data = wildcard()
    weight = wildcard()
    weight_t = is_op('transpose')(weight)
    return is_op('nn.dense')(data, weight_t)

def reshape_transpose_to_hslice():
    act = wildcard()
    act_r = is_op('reshape')(act)
    return is_op('transpose')(act_r)
    
def transpose_reshape_to_hstack():
    act = wildcard()
    act_t = is_op("transpose")(act)
    return is_op("reshape")(act_t)

def reshape_to_vstack():
    act = wildcard()
    return is_op('reshape')(act)

def reshape_to_vslice():
    act = wildcard()
    return is_op('reshape')(act)

def stack_reshape_reshape_to_binary_stack():
    act = is_tuple(None)
    stack = is_op("stack")(act)
    rshp = is_op("reshape")(stack)
    return is_op("reshape")(rshp)

def decompose_concat_input_tuple():
    act = is_tuple(None)
    return is_op("concatenate")(act)


def merge_conv2d_with_bias():
    input = wildcard()
    weight = wildcard()
    bias = wildcard()

    conv2d = is_op('nn.conv2d')(input, weight)
    bias_add = is_op('nn.bias_add')(conv2d, bias)

    return bias_add

@register_pattern_table("buda")
def pattern_table():
    matmul = ("buda.matmul", dense_to_matmul())
    hslice = ("buda.hslice", reshape_transpose_to_hslice(), is_reshape_transpose_hslice)
    hstack = ("buda.hstack", transpose_reshape_to_hstack(), is_transpose_reshape_hstack)
    vstack = ("buda.vstack", reshape_to_vstack(), is_reshape_vstack)
    vslice = ("buda.vslice", reshape_to_vslice(), is_reshape_vslice)
    layernorm = ("buda.layernorm", nn_layernorm_to_buda_layernorm())
    binary_stack = ("buda.binary_stack", stack_reshape_reshape_to_binary_stack(), is_stack_reshape_reshape_to_binary_stack)
    concatenate = ("buda.concatenate", decompose_concat_input_tuple())
    buda_conv2d_with_bias = ("buda.buda_conv2d_with_bias", merge_conv2d_with_bias())

    buda_patterns = [hstack, hslice, vstack, vslice, matmul, binary_stack, concatenate, buda_conv2d_with_bias]

    return buda_patterns




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

class ReconstructPyTorchGeluNew(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
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

        return tvm.relay.gelu(node_map[self.act][0])

class ReconstructPyTorchGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
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
        axis = None
        # Find the last dimension of the specific size
        for i, dim in enumerate(reversed(act_shape)):
            if dim == gamma.checked_type.shape[0]:
                axis = (i * -1) - 1 # i == 0 means axis = -1
                break

        assert axis is not None, "Cannot find an axis in input activation that matches weight shape"

        # Also supports padded shapes (e.g. (32, 1, 1))
        gamma_shape = list(gamma.checked_type.shape)
        gamma_shape.pop(axis)
        is_padded = all(dim == 1 for dim in gamma_shape)
        
        assert len(gamma.checked_type.shape) == 1 or is_padded, "TVM Layernorm only supports single dim layernorm"

        return tvm.relay.layernorm(act, gamma, beta, eps, axis)

class ReconstructTFLayerNorm(DFPatternCallback):
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

        assert len(gamma.type_annotation.shape) == 1, "TVM Layernorm only supports single dim layernorm"
        axis = None
        # Find the last dimension of the specific size
        for i, dim in enumerate(reversed(act_shape)):
            if dim == gamma.type_annotation.shape[0]:
                axis = (i * -1) - 1 # i == 0 means axis = -1
                break

        assert axis is not None, "Cannot find an axis in input activation that matches weight shape"

        return tvm.relay.layernorm(act, gamma, beta, eps, axis)

class UpdateConstants(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.params = {}
        self.const_idx = 0
        self.pattern = is_constant()
        self.function_name = ""

    def callback(self, pre, post, node_map):
        if self.function_name not in self.params:
            self.params[self.function_name] = {}

        self.params[self.function_name][self.const_idx] = post.data
        self.const_idx += 1
        return post

class AddNopsToPassthrough(ExprMutator):
    def __init__(self):
        super().__init__()
        self.output_vars =  []

    def visit_var(self, var):
        if var in self.output_vars:
            target_shape = list(var.checked_type.shape)
            return tvm.relay.reshape(var, newshape=target_shape)
        else:
            return var

    def visit_call(self, call):
        new_op = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return tvm.relay.Call(new_op, new_args, call.attrs)

    def visit_global_var(self, gvar):
        return gvar

    def visit_op(self, op):
        return op

    def visit_function(self, fn):
        if isinstance(fn.body, tvm.relay.expr.Tuple):
            outputs = [output for output in fn.body]
        else:
            outputs = [fn.body]

        self.output_vars.extend([output for output in outputs if isinstance(output, tvm.relay.Var)])
        new_body = self.visit(fn.body)
        return tvm.relay.Function(list(fn.params), new_body, fn.ret_type, fn.type_params, fn.attrs)


class AllowUnsupportedOps(ExprMutator):
    def __init__(self, check_only=False):
        super().__init__()
        self.check_only = check_only

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op):
            if self.check_only:
                assert False, f"Operator {call.op} is not supported"
            elif call.op.get_attr("target.buda") is None:
                def _func_wrapper(expr):
                    return True
                tvm.ir.register_op_attr(call.op.name, "target.buda", _func_wrapper)

        return super().visit_call(call)


def reconstruct_ops_for_buda(mod):
    print_all = False

    logger.trace("reconstruct_ops_for_buda:: At Entry")
    logger.trace(mod.functions)

    mod["main"] = rewrite(ReconstructPyTorchGelu(), mod["main"])
    logger.trace("After ReconstructPyTorchGelu")
    logger.trace(mod.functions)

    mod["main"] = rewrite(ReconstructPyTorchGeluNew(), mod["main"])
    logger.trace("After ReconstructPyTorchGeluNew")
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


def compile_for_buda(relay_module, graph_name, target='llvm', params=None):

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

        dump_graph(relay_module, graph_name, "before_compiling")

        relay_module = run_relay_compile_passes(relay_module)
        dump_graph(relay_module, graph_name, "after_relay_passes")
        compiled_relay_module = run_buda_compile_passes(relay_module)
        dump_graph(compiled_relay_module, graph_name, "after_buda_passes")

    return compiled_relay_module, params


def partition_for_buda(mod, graph_name, allow_unsupported=False):
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

        mod["main"] = AddNopsToPassthrough().visit(mod["main"])
        logger.trace("After AddNopsToPassthrough")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.InferType()])(mod)
        logger.trace("After InferType")
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

        mod["main"] = AllowUnsupportedOps(check_only=True).visit(mod["main"])
        logger.trace("After AllowUnsupportedOps")
        logger.trace(mod.functions)

        if not isinstance(mod["main"].body, tvm.relay.expr.Tuple):
            main_body_call_node = [mod["main"].body]
        else:
            main_body_call_node = mod["main"].body

        for item in main_body_call_node:
            if isinstance(item, tvm.relay.expr.Call):
                for arg in item.args:
                    if isinstance(arg, tvm.relay.expr.Var):
                        continue
                    assert isinstance(arg.op, tvm.ir.expr.GlobalVar), f"Operator {arg.op.name} is unsupported"
                    assert arg.op in mod.global_var_map_.values(), mod["main"]

        assert len(mod.global_var_map_) > 1, f"No buda compatible graph can be generated"

        constant_updator = UpdateConstants()

        for i in range(1, len(mod.global_var_map_), 1):
            constant_updator.function_name = mod.get_global_vars()[i].name_hint
            rewrite(constant_updator, mod[mod.get_global_vars()[i]])
        params = constant_updator.params

    dump_graph(mod, graph_name, "after_buda_partition")
        
    return mod, params
