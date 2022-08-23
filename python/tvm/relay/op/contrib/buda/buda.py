import logging
import torch

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.ir.type import TupleType
from tvm.ir.tensor_type import TensorType
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
from tvm._ffi.base import TVMError
from tvm.ir.transform import PassContext
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name, BuildModule
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

import math
import numpy as np
from tvm.relay.dataflow_pattern import *

from loguru import logger

import networkx as nx

def _register_external_op_helper_pytorch(op_name, supported=True):
    op = tvm.ir.op.Op.get(op_name)
    if op.has_attr("target.pybuda_cpudevice"):
        op.reset_attr("target.pybuda_cpudevice")

    @tvm.ir.register_op_attr(op_name, "target.pybuda_cpudevice")
    def _func_wrapper(expr):
        from pybuda.config import _get_global_compiler_config
        compiler_cfg = _get_global_compiler_config()
        return compiler_cfg.enable_tvm_cpu_fallback
    return _func_wrapper


def initialize_pybuda_cpudevice_ops(mod):
    ResetOpAttributes().visit(mod["main"])
    _register_external_op_helper_pytorch("embedding")
    _register_external_op_helper_pytorch("equal")
    _register_external_op_helper_pytorch("nn.log_softmax")

def _register_external_op_helper_pybuda(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.pybuda")
    def _func_wrapper(expr):
        return supported
    return _func_wrapper

_register_external_op_helper_pybuda("abs")
_register_external_op_helper_pybuda("add")
_register_external_op_helper_pybuda("adv_index")
_register_external_op_helper_pybuda("argmax")
_register_external_op_helper_pybuda("broadcast_to")
_register_external_op_helper_pybuda("cast")
_register_external_op_helper_pybuda("clip")
_register_external_op_helper_pybuda("cos")
_register_external_op_helper_pybuda("cumsum")
_register_external_op_helper_pybuda("exp")
_register_external_op_helper_pybuda("gelu")
_register_external_op_helper_pybuda("image.resize2d")
_register_external_op_helper_pybuda("layernorm")
_register_external_op_helper_pybuda("log")
_register_external_op_helper_pybuda("logical_not")
_register_external_op_helper_pybuda("max")
_register_external_op_helper_pybuda("maximum")
_register_external_op_helper_pybuda("mean")
_register_external_op_helper_pybuda("multiply")
_register_external_op_helper_pybuda("nn.avg_pool2d")
_register_external_op_helper_pybuda("nn.batch_matmul")
_register_external_op_helper_pybuda("nn.conv2d_transpose")
_register_external_op_helper_pybuda("nn.conv2d")
_register_external_op_helper_pybuda("nn.dense")
_register_external_op_helper_pybuda("nn.dropout")
_register_external_op_helper_pybuda("nn.leaky_relu")
_register_external_op_helper_pybuda("nn.matmul")
_register_external_op_helper_pybuda("nn.max_pool1d")
_register_external_op_helper_pybuda("nn.max_pool2d")
_register_external_op_helper_pybuda("nn.pad")
_register_external_op_helper_pybuda("nn.relu")
_register_external_op_helper_pybuda("nn.softmax")
_register_external_op_helper_pybuda("ones")
_register_external_op_helper_pybuda("power")
_register_external_op_helper_pybuda("reciprocal")
_register_external_op_helper_pybuda("reshape")
_register_external_op_helper_pybuda("scatter_add")
_register_external_op_helper_pybuda("scatter")
_register_external_op_helper_pybuda("sigmoid")
_register_external_op_helper_pybuda("sin")
_register_external_op_helper_pybuda("sqrt")
_register_external_op_helper_pybuda("stack")
_register_external_op_helper_pybuda("strided_slice")
_register_external_op_helper_pybuda("subtract")
_register_external_op_helper_pybuda("sum")
_register_external_op_helper_pybuda("tanh")
_register_external_op_helper_pybuda("transpose")
_register_external_op_helper_pybuda("where")
_register_external_op_helper_pybuda("zeros")


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

def transpose_reshape_reshape_to_hstack():
    act = wildcard()
    act_t = is_op("transpose")(act)
    rshp = is_op("reshape")(act_t)
    return is_op("reshape")(rshp)

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

def decompose_adv_index_input_tuple():
    act = wildcard()
    indices = wildcard()
    input = is_tuple([act, indices])

    return is_op("adv_index")(input)

def dropout_tuple_get_item():
    act = wildcard()
    dropout = is_op("nn.dropout")(act)
    return is_tuple_get_item(dropout)

def merge_conv2d_with_bias():
    input = wildcard()
    weight = wildcard()
    bias = wildcard()

    conv2d = is_op('nn.conv2d')(input, weight)
    bias_add = is_op('nn.bias_add')(conv2d, bias)

    return bias_add

@register_pattern_table("pybuda")
def pattern_table():
    matmul = ("pybuda.matmul", dense_to_matmul())
    hslice = ("pybuda.hslice", reshape_transpose_to_hslice(), is_reshape_transpose_hslice)
    hstack = [
        ("pybuda.hstack", transpose_reshape_to_hstack(), is_transpose_reshape_hstack), 
        ("pybuda.hstack", transpose_reshape_reshape_to_hstack(), is_transpose_reshape_reshape_hstack)
        ]
    vstack = ("pybuda.vstack", reshape_to_vstack(), is_reshape_vstack)
    vslice = ("pybuda.vslice", reshape_to_vslice(), is_reshape_vslice)
    layernorm = ("pybuda.layernorm", nn_layernorm_to_buda_layernorm())
    binary_stack = ("pybuda.binary_stack", stack_reshape_reshape_to_binary_stack(), is_stack_reshape_reshape_to_binary_stack)
    concatenate = ("pybuda.concatenate", decompose_concat_input_tuple())
    adv_index = ("pybuda.adv_index", decompose_adv_index_input_tuple())
    buda_conv2d_with_bias = ("pybuda.buda_conv2d_with_bias", merge_conv2d_with_bias())
    dropout = ("pybuda.dropout", dropout_tuple_get_item())

    buda_patterns = [*hstack, hslice, vstack, vslice, matmul, binary_stack, concatenate, buda_conv2d_with_bias, adv_index, dropout]

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

class ReconstructPyTorchLayerNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
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

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = pre_node_map[self.act][0].checked_type.shape
        gamma_shape = list(pre_node_map[self.gamma][0].checked_type.shape)

        axis = None
        # Find the last dimension of the specific size
        for i, dim in enumerate(reversed(act_shape)):
            if dim == gamma_shape[0]:
                axis = (i * -1) - 1 # i == 0 means axis = -1
                break

        assert axis is not None, "Cannot find an axis in input activation that matches weight shape"

        # Also supports padded shapes (e.g. (32, 1, 1))
        gamma_shape.pop(axis)
        is_padded = all(dim == 1 for dim in gamma_shape)

        gamma_shape = pre_node_map[self.gamma][0].checked_type.shape        
        assert len(gamma_shape) == 1 or is_padded, "TVM Layernorm only supports single dim layernorm"

        return tvm.relay.layernorm(act, gamma, beta, eps, axis)

class ReconstructTFLayerNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
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

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = pre_node_map[self.act][0].checked_type.shape
        gamma_shape = pre_node_map[self.gamma][0].checked_type.shape
        beta_shape = pre_node_map[self.beta][0].checked_type.shape

        if len(gamma_shape) > 1 and sum([1 if int(x) != 1 else 0 for x in list(gamma_shape)]) == 1:
            # Count the number of dims thats not 1
            gamma_shape = (np.prod([int(x) for x in gamma_shape]),)
            gamma = tvm.relay.reshape(gamma, newshape=gamma_shape)
        else:
            assert len(gamma_shape) == 1, "TVM Layernorm only supports single dim layernorm"

        if len(beta_shape) > 1 and sum([1 if int(x) != 1 else 0 for x in list(beta_shape)]) == 1:
            # Count the number of dims thats not 1
            beta_shape = (np.prod([int(x) for x in beta_shape]),)
            beta = tvm.relay.reshape(beta, newshape=gamma_shape)
        else:
            assert len(beta_shape) == 1, "TVM Layernorm only supports single dim layernorm"

        axis = None
        # Find the last dimension of the specific size
        for i, dim in enumerate(reversed(act_shape)):
            if dim == gamma_shape[0]:
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

def _always_true(expr):
    return True

class AllowUnsupportedOps(ExprMutator):
    def __init__(self, check_only=False):
        super().__init__()
        self.check_only = check_only

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op):
            if self.check_only:
                assert False, f"Operator {call.op} is not supported"
            elif call.op.get_attr("target.pybuda") is None:
                tvm.ir.register_op_attr(call.op.name, "target.pybuda", _always_true)

        return super().visit_call(call)

class ResetOpAttributes(ExprVisitor):
    def __init__(self):
        super().__init__()

    def visit_op(self, op):
        if op.has_attr("target.pybuda_cpudevice"):
            op.reset_attr("target.pybuda_cpudevice")
        return super().visit_op(op)

def node_hash(node):
    if hasattr(node, "op"):
        if isinstance(node.op, tvm.relay.function.Function):
            node_descriptor = ("Function", False)
        else:
            node_descriptor = (node.op, False)
    elif isinstance(node, tvm.relay.expr.Var):
        node_descriptor = (node.name_hint, True)
    else:
        node_descriptor = (type(node), False)
    return (tvm.ir.structural_hash(node, map_free_vars=True), node_descriptor)

class NodeIndexer():
    def __init__(self):
        self.counters = ["function","call","let","var","global_var","if","tuple","tuple_getitem","constant"]
        self.increment = 100
        self.index = 0
        self.node_map = {}
    
    def reset_index(self):
        self.index = 0
    
    def count_visit(self, visitee, node=None):
        assert visitee in self.counters
        index = self.counters.index(visitee) + 1
        self.index += index * self.increment
        if node is not None:
            self.node_map[self.index] = node
        return self.index

class NodeReMapper(ExprVisitor):
    def __init__(self, node_list, node_map):
        self.node_indexer = NodeIndexer()
        self.node_map = node_map
        self.node_list = node_list
        super().__init__()

    def visit_call(self, call):
        index = self.node_indexer.count_visit("call")
        node = node_hash(call)
        if node != self.node_map[index] and self.node_map[index] in self.node_list:
            self.node_list.remove(self.node_map[index])
            self.node_list.add(node)
    
class DetermineTarget(ExprMutator):
    def __init__(self, graph, fallback_nodes):
        super().__init__()
        self.users_of_unsupported_ops = 0
        self.graph = graph
        self.nodes_to_cpu_eval = set()
        self.nodes_to_cpu_eval = self.nodes_to_cpu_eval | fallback_nodes
        self.graph_changed = True
        for node in fallback_nodes:
            ancestors = nx.ancestors(graph, node)
            descendants = nx.descendants(graph, node)
            if len(ancestors) > len(descendants):
                self.nodes_to_cpu_eval = self.nodes_to_cpu_eval | descendants
            else:
                self.nodes_to_cpu_eval = self.nodes_to_cpu_eval | ancestors

    def visit_call(self, call):
        def _cpu_eval(expr):
            cpu_eval = node_hash(expr) in self.nodes_to_cpu_eval
            if cpu_eval:
                logger.info(f"{expr.op.name} will be executed on CPU")
            return cpu_eval

        if node_hash(call) in self.nodes_to_cpu_eval:
            if isinstance(call.op, tvm.relay.function.Function):
                self.graph_changed = True
                new_attrs = {k: (v if k != "Composite" else v.replace("pybuda", "pybuda_cpudevice")) for (k, v) in call.op.attrs.items()}
                new_fn = call.op.with_attr(new_attrs)
                return tvm.relay.expr.Call(new_fn, call.args)
            else:
                try:
                    tvm.ir.register_op_attr(call.op.name, "target.pybuda_cpudevice", _cpu_eval, level=5)
                except:
                    pass

        if isinstance(call.op, tvm.ir.op.Op) and call.op.get_attr("target.pybuda") is not None:
            # for non-unary ops, if one of the args is unsupported, and only has one output, do the op on CPU, up to a total of
            # max_users_of_unsupported_ops ops to reduce data movement
            non_weight_args = [arg for arg in call.args if not isinstance(arg, tvm.relay.expr.Var)]
            if len(non_weight_args) > 1:
                for arg in call.args:
                    output_nodes = self.graph.out_degree(node_hash(arg))
                    if isinstance(arg, tvm.relay.expr.Call) and isinstance(arg.op, tvm.ir.op.Op) and arg.op.get_attr("target.pybuda") is None and output_nodes == 1:
                        self.nodes_to_cpu_eval.add(node_hash(call))
                        try:
                            tvm.ir.register_op_attr(call.op.name, "target.pybuda_cpudevice", _cpu_eval, level=5)
                        except:
                            pass
                        break

        return super().visit_call(call)


    # def visit_function(self, fn):
    #     import pdb; pdb.set_trace()
    #     return super().visit_function(fn)
    

def add_shared_weights_to_fallback(graph, fallback_nodes, input_names):
    added_nodes = set()
    input_nodes = [node for node in graph.nodes if node[1][1] and node[1][0] in input_names]
    topo_graph = list(nx.topological_sort(graph))
    for fallback_node in fallback_nodes:
        for ancestor in nx.ancestors(graph, fallback_node):
            name, maybe_param = ancestor[1]
            if not maybe_param or name in input_names:
                continue
            for output_node in graph.successors(ancestor):
                # if the output node or any of its discendants is not the fallback node
                if output_node != fallback_node and fallback_node not in nx.descendants(graph, output_node):
                    index = 0
                    nodes_to_check = [output_node]
                    while index < len(nodes_to_check):
                        node = nodes_to_check[index]
                        added_nodes.add(node)
                        if any(an for an in nx.ancestors(graph, node) if an in input_nodes):
                            break
                        nodes_to_check.extend(graph.successors(node))
                        index += 1

    return added_nodes | fallback_nodes

class ConstructDiGraph(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.graph = nx.MultiDiGraph()
        self.fallback_nodes = set()
        self.node_indexer = NodeIndexer()

    def register_args(self, parent, parent_node):
        for arg in parent.args:
            self.graph.add_edge(node_hash(arg), parent_node)

    def visit_call(self, call):
        node = node_hash(call)
        self.node_indexer.count_visit("call", node)
        if isinstance(call.op, tvm.ir.op.Op) and call.op.get_attr("target.pybuda_cpudevice") is not None:
            self.fallback_nodes.add(node)

        self.register_args(call, node)
        # Make sure CPU output shape starts with 1
        if (
            isinstance(call.op, tvm.ir.op.Op) 
            and call.op.name == "reshape"
            and call.checked_type.shape[0] == 1
            and isinstance(call.args[0], tvm.relay.expr.Call)
            and isinstance(call.args[0].op, tvm.ir.op.Op)
            and call.args[0].op.name == "embedding"
            and call.args[0].op.get_attr("target.pybuda_cpudevice") is not None
        ):
            self.fallback_nodes.add(node)

        return super().visit_call(call)
    
    def visit_tuple_getitem(self, t):
        node = node_hash(t)
        self.register_args(t.tuple_value, node)
        return super().visit_tuple_getitem(t)

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

    mod["main"] = rewrite(ReconstructOnnxGelu(), mod["main"])
    logger.trace("After ReconstructOnnxGelu")
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

def get_relay_output(mod, params, inputs, target):
    # Build and Run Relay modules with inputs as (key : tensor) pair
    # Then, inputs dont need to be in the same order as 'mod' defines.
    ret_type = mod["main"].checked_type.ret_type
    lib = relay.build(mod, target=target, params=params)
    m = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    m.run(**inputs)
    
    def _unflatten(flat_iter, cur_type):
        import tvm.relay.ty as _ty
        if isinstance(cur_type, _ty.TensorType):
            return next(flat_iter)
        if isinstance(cur_type, _ty.TupleType):
            fields = []
            for field_type in cur_type.fields:
                field = _unflatten(flat_iter, field_type)
                fields.append(field)
            return fields
        raise ValueError("Return type", ret_type, "contains unsupported type", cur_type)

    flattened = []
    import tvm.runtime.ndarray as _nd
    for i in range(m.get_num_outputs()):
        flattened.append(m.get_output(i).copyto(_nd.cpu(0)))
    relay_outputs = _unflatten(iter(flattened), ret_type)

    if not isinstance(relay_outputs, (list, tuple)):
        relay_outputs = [relay_outputs]
    relay_outputs = [x.numpy() for x in flattened]

    return relay_outputs


def verify_outputs(framework_outputs, relay_outputs, compile_location, rtol=1e-02, atol=1e-04, pcc=None):
    allowed_to_fail = False
    if len(framework_outputs) != len(relay_outputs):
        logger.error(f"Different number of outputs. Framework: {len(framework_outputs)}, TVM: {len(relay_outputs)} after {compile_location}")

    for i, (fr_out, tvm_out) in enumerate(zip(framework_outputs, relay_outputs)):

        if pcc is None:
            ok = np.allclose(fr_out, tvm_out, rtol=rtol, atol=atol, equal_nan=True)
        else:
            pcc_value = np.min(np.ma.corrcoef(np.ma.masked_invalid(fr_out.flatten()), np.ma.masked_invalid(tvm_out.flatten())))
            if isinstance(pcc_value, np.ma.core.MaskedConstant):
                pcc_value = 1.0
            ok = pcc_value >= pcc

        if not ok:
            logger.error(f"Tensor mismatch on output {i} between framework and TVM after {compile_location}.")
            logger.trace(f"Framework: (shape = {fr_out.shape}")
            logger.trace(fr_out)
            logger.trace(f"TVM: (shape = {tvm_out.shape}")
            logger.trace(tvm_out)
            logger.info("Max ATOL Delta: " + "{:.3e}".format(np.max(np.abs(np.tensor(fr_out - tvm_out))).item()) + ", atol=" +  "{}".format(atol))
            logger.info("Max RTOL Delta: " + "{:.3e}".format(np.max(np.abs(np.tensor(fr_out - tvm_out))/np.tensor(tvm_out)).item()) + ", rtol=" + "{}".format(rtol))
            if pcc is not None:
                logger.info(f"PCC got={pcc_value}, required={pcc}")
            if not allowed_to_fail:
                raise RuntimeError

    logger.info(f"Verified TVM Relay outputs against framework outputs after {compile_location}")

def verify_tvm_compile(mod, params, inputs, target, framework_outputs, compile_location, verify_cfg=None):
    relay_outputs = get_relay_output(mod, params, inputs, target)

    # Verify compile passes (original relay passes + buda passes)
    if verify_cfg:
        verify_outputs(framework_outputs, relay_outputs, compile_location, rtol=verify_cfg.rtol, atol=verify_cfg.atol, pcc=verify_cfg.pcc)
    else:
        verify_outputs(framework_outputs, relay_outputs, compile_location)


def compile_for_buda(relay_module, graph_name, target='llvm', params=None, inputs=None, framework_outputs=None, verify_cfg=None):

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

    tophub_context = tvm.autotvm.utils.EmptyContext()

    with tophub_context, tvm.transform.PassContext(opt_level=5):
        bld_mod = BuildModule()
        if params:
            bld_mod._set_params(params)

        logger.trace("Before Compiling")
        logger.trace(relay_module.functions)

        dump_graph(relay_module, graph_name, "before_compiling")

        relay_module = run_relay_compile_passes(relay_module)
        dump_graph(relay_module, graph_name, "after_relay_passes")
        compiled_relay_module = run_buda_compile_passes(relay_module, params, inputs, target, framework_outputs, verify_cfg)
        dump_graph(compiled_relay_module, graph_name, "after_buda_passes")

    return compiled_relay_module, params

class FlattenInputs(ExprMutator):

    def __init__(self, flattenend_name_map):
        super().__init__()
        self.flattened_name_map = flattenend_name_map
        self.input_tuples= []
        self.new_params = []

    def visit_tuple_getitem(self, op):
        if op.tuple_value in self.input_tuples:
            tup_index = self.tuple_indices[self.input_tuples.index(op.tuple_value)]
            return self.visit(self.old_param_map[tup_index][op.index])
        
        new_op = tvm.relay.TupleGetItem(self.visit(op.tuple_value), op.index)
        return new_op

    def visit_call(self, call):
        new_op = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return tvm.relay.Call(new_op, new_args, call.attrs)

    def visit_function(self, fn):
        new_body = fn.body
        self.old_param_map = {}
        self.tuple_indices = []

        param_num = 0

        for i in range(len(fn.params)):
            if isinstance(fn.params[i].type_annotation, TupleType):
                self.input_tuples.append(fn.params[i])
                inputs = fn.params[i].type_annotation.fields
                new_params = []
                
                for j in range(len(inputs)):
                    input = inputs[j]
                    new_params.append(tvm.relay.Var(self.flattened_name_map[fn.params[i].name_hint][j], input))
                    param_num += 1

                    if i not in self.old_param_map:
                        self.tuple_indices.append(i)
                        self.old_param_map[i] = {}
                    self.old_param_map[i][j] = new_params[-1]
   
                self.new_params += new_params
                
            else:
                self.new_params.append(fn.params[i])
                self.old_param_map[i] = self.new_params[-1]
                

        new_body = self.visit(fn.body)
        return tvm.relay.Function(self.new_params, new_body, fn.ret_type, fn.type_params, fn.attrs)

def flatten_inputs(mod, flattened_inputs, flattened_name_map):

    # flattens inputs in IR
    mod["main"] = FlattenInputs(flattened_name_map).visit(mod["main"])
    logger.trace("After FlattenInputs")
    logger.trace(mod.functions)

    return mod
    
def partition_for_buda(mod, graph_name, compiler_cfg, input_names=[]):
    initialize_pybuda_cpudevice_ops(mod)
    
    with tvm.transform.PassContext(opt_level=5):
        logger.trace("partition_for_buda:: At Entry")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.InferType()])(mod)
        logger.trace("After InferType")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.MergeComposite(pattern_table())])(mod)
        logger.trace("After MergeComposite")
        logger.trace(mod.functions)

        if compiler_cfg.enable_tvm_unsupported_ops:
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

        if compiler_cfg.enable_tvm_cpu_fallback:
            import time
            logger.debug(f"Running cpu fallback compilation")
            logger.debug(f"Constructing digraph...")
            start = time.time()
            graph_constructor = ConstructDiGraph()
            graph_constructor.visit(mod["main"])
            logger.debug(f"Done, took: {(time.time() - start):.2f} s")
            # dot = nx.nx_pydot.to_pydot(graph_constructor.graph)
            # print(dot)

            logger.debug(f"Finding and adding shared weights...")
            start = time.time()
            fallback_nodes = add_shared_weights_to_fallback(graph_constructor.graph, graph_constructor.fallback_nodes, input_names)
            logger.debug(f"Done, took: {(time.time() - start):.2f} s")
            logger.debug(f"Determining target for ops...")
            terget_determiner = DetermineTarget(graph_constructor.graph, fallback_nodes)
            new_mod = None
            start = time.time()
            while terget_determiner.graph_changed:
                logger.debug(".")
                terget_determiner.graph_changed = False
                mod["main"] = terget_determiner.visit(mod["main"])
            logger.debug(f"Done, took: {(time.time() - start):.2f} s")
            logger.debug(f"Remapping nodes...")
            start = time.time()
            node_remapper = NodeReMapper(terget_determiner.nodes_to_cpu_eval, graph_constructor.node_indexer.node_map)
            node_remapper.visit(mod["main"])
            terget_determiner.nodes_to_cpu_eval = node_remapper.node_list
            logger.trace("After DetermineTarget")
            logger.trace(mod.functions)
            logger.debug(f"Done, took: {(time.time() - start):.2f} s")

        mod = tvm.transform.Sequential([transform.AnnotateTarget(["pybuda_cpudevice", "pybuda"])])(mod)
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
                    if isinstance(arg, tvm.relay.expr.Tuple):
                        continue
                    assert isinstance(arg.op, tvm.ir.expr.GlobalVar), f"Operator {arg.op.name} is unsupported"
                    assert arg.op in mod.global_var_map_.values(), mod["main"]

        assert len(mod.global_var_map_) > 1, f"No buda compatible graph can be generated"

        constant_updator = UpdateConstants()

        for i in range(len(mod.global_var_map_)):
            function_name = mod.get_global_vars()[i].name_hint
            ResetOpAttributes().visit(mod[mod.get_global_vars()[i]])
            if function_name == "main":
                continue
            constant_updator.function_name = function_name
            rewrite(constant_updator, mod[mod.get_global_vars()[i]])
        params = constant_updator.params

    dump_graph(mod, graph_name, "after_buda_partition")
        
    return mod, params
