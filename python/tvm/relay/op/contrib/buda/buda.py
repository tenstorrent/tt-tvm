import logging
import torch

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.ir.type import TupleType
from tvm.ir.tensor_type import TensorType
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
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

def _register_external_op_helper_pytorch(op_name, compiler_cfg, supported=True):
    op = tvm.ir.op.Op.get(op_name)
    if op.has_attr("target.pybuda_cpudevice"):
        op.reset_attr("target.pybuda_cpudevice")

    @tvm.ir.register_op_attr(op_name, "target.pybuda_cpudevice")
    def _func_wrapper(expr):
        return compiler_cfg.enable_tvm_cpu_fallback
    return _func_wrapper

def initialize_pybuda_cpudevice_ops(mod, compiler_cfg):
    ResetOpAttributes().visit(mod["main"])
    for op in compiler_cfg.cpu_fallback_ops:
        _register_external_op_helper_pytorch(op, compiler_cfg)
    _register_external_op_helper_pytorch("equal", compiler_cfg)
    _register_external_op_helper_pytorch("nn.log_softmax", compiler_cfg)
    _register_external_op_helper_pytorch("scatter_add", compiler_cfg)

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
    return is_op("reshape")(stack)

def concat_reshape_reshape_to_binary_stack():
    x1 = is_op("reshape")(wildcard())
    x2 = is_op("reshape")(wildcard())
    act = is_tuple([x1, x2])
    concat = is_op("concatenate")(act)
    return is_op("reshape")(concat)

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

def merge_conv2d_transpose_with_bias():
    input = wildcard()
    weight = wildcard()
    bias = wildcard()

    conv2d = is_op('nn.conv2d_transpose')(input, weight)
    bias_add = is_op('nn.bias_add')(conv2d, bias)

    return bias_add

def channel_last_resize():
    input = wildcard()

    transpose_input_0 = is_op("transpose")(input).has_attr({"axes": [0, 3, 2, 1]})
    transpose_input_1 = is_op("transpose")(transpose_input_0).has_attr({"axes": [0, 1, 3, 2]})

    resize = is_op("image.resize2d")(transpose_input_1).has_attr({"layout":"NCHW"})

    transpose_result_0 = is_op("transpose")(resize).has_attr({"axes": [0, 2, 1, 3]})
    transpose_result_1 = is_op("transpose")(transpose_result_0).has_attr({"axes": [0, 1, 3, 2]})

    return transpose_result_1

@register_pattern_table("pybuda")
def pattern_table():
    matmul = ("pybuda.matmul", dense_to_matmul())
    hslice = ("pybuda.hslice", reshape_transpose_to_hslice(), is_reshape_transpose_hslice)
    hstack = [
        ("pybuda.hstack", transpose_reshape_to_hstack(), is_transpose_reshape_hstack), 
        ("pybuda.hstack", transpose_reshape_reshape_to_hstack(), is_transpose_reshape_reshape_hstack)
        ]
    binary_stack = [
        ("pybuda.binary_stack", stack_reshape_reshape_to_binary_stack(), is_stack_reshape_reshape_to_binary_stack),
        ("pybuda.binary_stack", concat_reshape_reshape_to_binary_stack(), is_concat_reshape_reshape_to_binary_stack),
    ]
    adv_index = ("pybuda.adv_index", decompose_adv_index_input_tuple())
    buda_conv2d_with_bias = ("pybuda.buda_conv2d_with_bias", merge_conv2d_with_bias())
    buda_conv2d_transpose_with_bias = ("pybuda.buda_conv2d_transpose_with_bias", merge_conv2d_transpose_with_bias())
    dropout = ("pybuda.dropout", dropout_tuple_get_item())

    # channel_last_maxpool2d = ("pybuda.channel_last_maxpool", channel_last_maxpool())
    channel_last_resize2d = ("pybuda.channel_last_resize2d", channel_last_resize())
    buda_patterns = [
        *hstack, 
        *binary_stack, 
        hslice, 
        matmul, 
        buda_conv2d_with_bias,
        buda_conv2d_transpose_with_bias, 
        adv_index, 
        dropout]

    return buda_patterns

# TM CPU Fallback ops of interest. Ones that are valuable 
# to be included as additional fallback nodes
tm_cpu_fallback_ops_of_interest = [
    "adv_index",
    "arange",
    "broadcast_to_like",
    "broadcast_to",
    "cast_like",
    "cast",
    "collapse_sum_like",
    "collapse_sum_to",
    "concatenate",
    "contrib_reverse_reshape",
    "embedding",
    "expand_dims",
    "full_like",
    "full",
    "gather_nd",
    "gather",
    "matrix_set_diag",
    "meshgrid",
    "one_hot",
    "reinterpret",
    "repeat",
    "reshape_like",
    "reshape",
    "reverse_sequence",
    "reverse",
    "sequence_mask",
    "slice_like",
    "sparse_to_dense",
    "split",
    "squeeze",
    "stack",
    "strided_slice",
    "take",
    "tile",
    "unravel_index",
    "where",
    # PyBuda
    "pybuda.adv_index",
    "pybuda.binary_stack",
    "pybuda.hslice",
    "pybuda.hstack",
]

# TM CPU Fallback ops which should not be included in fallback
# itself. Those ops often are decomposed into some sort of matmuls
# which should be forced to be executed on TT device mostly.
tm_cpu_fallback_ops_to_not_include = [
    "nn.adaptive_avg_pool1d",
    "nn.adaptive_avg_pool2d",
    "nn.adaptive_avg_pool3d",
    "nn.adaptive_max_pool1d",
    "nn.adaptive_max_pool2d",
    "nn.adaptive_max_pool3d",
    "nn.avg_pool1d",
    "nn.avg_pool2d",
    "nn.avg_pool3d",
    "nn.batch_matmul",
    "nn.conv1d_transpose",
    "nn.conv1d",
    "nn.conv2d_transpose",
    "nn.conv2d",
    "nn.conv3d_transpose",
    "nn.conv3d",
    "nn.dense",
    "nn.global_avg_pool2d",
    "nn.global_max_pool2d",
    "nn.matmul",
    "nn.max_pool1d",
    "nn.max_pool2d",
    "nn.max_pool3d",
    "nn.sparse_conv2d",
    "nn.sparse_dense",
    "nn.upsampling",
    "nn.upsampling3d",
    # PyBuda
    "pybuda.buda_conv2d_transpose_with_bias",
    "pybuda.buda_conv2d_with_bias",
    "pybuda.matmul",
]

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
            return tvm.relay.identity(var)
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

class IdentityFunctionUnraveller(ExprMutator):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def visit_call(self, call):
        if not hasattr(call.op, 'checked_type'):
            return super().visit_call(call)

        if isinstance(call.op.checked_type, tvm.relay.FuncType):
            function = self.mod[call.op.name_hint]
            if len(function.params) == 1 and function.body == function.params[0]:
                return super().visit(call.args[0])
                
        return super().visit_call(call)

class CheckFallbackOps(ExprVisitor):
    def __init__(self, cpu_fallback_ops):
        super().__init__()
        self.cpu_fallback_ops = cpu_fallback_ops
        self.has_fallback_ops = False
        
    def visit_op(self, op): 
        if op.name in self.cpu_fallback_ops:
            self.has_fallback_ops = True 
        return super().visit_op(op)
    
   
class UnwrapPyBudaOpsForCPUFallback(ExprMutator):
    def __init__(self):
        super().__init__()

    def visit_call(self, call):
        if isinstance(call.op, tvm.relay.function.Function):
            if "Composite" in call.op.attrs and call.op.attrs["Composite"] == "pybuda_cpudevice.matmul":
                if isinstance(call.args[0], tvm.relay.expr.Call) and call.args[0].op.name == "reshape":
                    arg0 = call.args[0].args[0]
                else:
                    arg0 = call.args[0]
                if isinstance(call.args[1], tvm.relay.expr.Var):
                    arg1 = call.args[1]
                    # If transpose is part of a function, and therefore not picked up, add it explicitly
                    if isinstance(call.op.body.args[1], tvm.relay.expr.Call) and call.op.body.args[1].op.name == "transpose":
                        arg1 = tvm.relay.transpose(arg1, axes=call.op.body.args[1].attrs['axes'])
                else:
                    arg1 = call.args[1].args[0]

                logger.info("Fixed linear")
                return tvm.relay.nn.dense(arg0, arg1)
            
            if "Composite" in call.op.attrs and call.op.attrs["Composite"] == "pybuda_cpudevice.hslice":
                # Get hslice input
                if isinstance(call.args[0], tvm.relay.expr.Call):
                    arg0 = call.args[0]
                else:
                    return super().visit_call(call)

                # Extract reshape and transpose attributes
                if isinstance(call.op.body, tvm.relay.expr.Call) and call.op.body.op.name == "transpose" and \
                isinstance(call.op.body.args[0], tvm.relay.expr.Call) and call.op.body.args[0].op.name == "reshape":
                    transpose = call.op.body
                    transpose_axes = transpose.attrs.axes
                    reshape = call.op.body.args[0]
                    reshape_new_shape = reshape.attrs.newshape
                else:
                    return super().visit_call(call)
                
                # Unwrap hslice function into composite ops
                ops = tvm.relay.reshape(arg0, reshape_new_shape)
                ops = tvm.relay.transpose(ops, transpose_axes)

                logger.info("Fixed hslice")
                return ops

        return super().visit_call(call)


class ResetOpAttributes(ExprVisitor):
    def __init__(self):
        super().__init__()

    def visit_op(self, op):
        if op.has_attr("target.pybuda_cpudevice"):
            op.reset_attr("target.pybuda_cpudevice")
        return super().visit_op(op)

def node_hash(node):
    """Generate unique TVM node with hash and metadata.
    
    TVM node hash is unique identifier of TVM node in the specific
    graph phase. As TVM doesn't have unique node IDs this is used. 
    Also, as way of generating TVM hash can depend on op attributes 
    and position in the graph, it's common that these unique hashes 
    change with each TVM optimization pass.
    
    Besides hash, this function also populates some meta-data related
    to the node. Here is explanation regarding each of them:
    1. Node name (depends on node type, op type, etc.)
    2. Is it variable or not (params)

    Args:
        node (relay.expr.Call): Visiting node

    Returns:
        tuple: Unique node identifier
    """
    if hasattr(node, "op"):
        if isinstance(node.op, tvm.relay.function.Function):
            node_descriptor = (node.op.attrs["Composite"], False)
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
        return super().visit_call(call)
    
class DetermineTarget(ExprMutator):
    def __init__(self, graph, fallback_nodes, compiler_cfg):
        super().__init__()
        self.users_of_unsupported_ops = 0
        self.graph = graph
        self.nodes_to_cpu_eval = set()
        self.nodes_to_cpu_eval = self.nodes_to_cpu_eval | fallback_nodes
        self.graph_changed = True
        self.modify_graph = False
        self.compiler_cfg = compiler_cfg
        for node in fallback_nodes:
            ancestors = nx.ancestors(graph, node)
            descendants = nx.descendants(graph, node)
            if len(ancestors) > len(descendants):
                self.nodes_to_cpu_eval = self.nodes_to_cpu_eval | descendants
            else:
                if self.compiler_cfg.enable_tm_cpu_fallback:
                    continue
                self.nodes_to_cpu_eval = self.nodes_to_cpu_eval | ancestors

    def visit_call(self, call):
        def _cpu_eval(expr):
            node = node_hash(expr)
            cpu_eval = node in self.nodes_to_cpu_eval
            if cpu_eval:
                logger.info(f"{expr.op.name} will be executed on CPU")
            return cpu_eval

        if self.modify_graph:
            if node_hash(call) in self.nodes_to_cpu_eval:
                if isinstance(call.op, tvm.relay.function.Function):
                    self.graph_changed = True
                    new_attrs = {k: (v if k != "Composite" else v.replace("pybuda", "pybuda_cpudevice")) for (k, v) in call.op.attrs.items()}
                    new_fn = call.op.with_attr(new_attrs)
                    logger.info(f"Changing graph")
                    return super().visit_call(tvm.relay.expr.Call(new_fn, call.args))
        elif node_hash(call) in self.nodes_to_cpu_eval and not isinstance(call.op, tvm.relay.function.Function) :
            try:
                tvm.ir.register_op_attr(call.op.name, "target.pybuda_cpudevice", _cpu_eval, level=5)
            except:
                pass

        # For non-unary ops, if one of the args is predefined for fallback and only has one output, do that op on CPU too to reduce data movement
        elif isinstance(call.op, tvm.ir.op.Op) and any([True if hasattr(arg, "op") and hasattr(arg.op, "name") and arg.op.name in self.compiler_cfg.cpu_fallback_ops else False for arg in call.args]):
            non_weight_args = [arg for arg in call.args if not isinstance(arg, tvm.relay.expr.Var)]
            if len(non_weight_args) > 1:
                call_node_ancestors = nx.ancestors(self.graph, node_hash(call))
                for arg_index, arg in enumerate(call.args):
                    output_nodes = self.graph.out_degree(node_hash(arg))
                    if isinstance(arg, tvm.relay.expr.Call) and isinstance(arg.op, tvm.ir.op.Op) and arg.op.get_attr("target.pybuda_cpudevice") is not None and output_nodes == 1:
                        arg_ancestors = nx.ancestors(self.graph, node_hash(arg))
                        arg_ancestors.add(node_hash(arg))
                        non_arg_ancestors = call_node_ancestors - arg_ancestors
                        contains_unsupported = any([ancestor in self.nodes_to_cpu_eval for ancestor in non_arg_ancestors])
                        if not contains_unsupported:
                            break

                        self.nodes_to_cpu_eval.add(node_hash(call))
                        self.nodes_to_cpu_eval = self.nodes_to_cpu_eval | call_node_ancestors
                        try:
                            tvm.ir.register_op_attr(call.op.name, "target.pybuda_cpudevice", _cpu_eval, level=5)
                        except:
                            pass
                        break

        return super().visit_call(call)


    # def visit_function(self, fn):
    #     return super().visit_function(fn)
    

def add_shared_weights_to_fallback(graph, fallback_nodes, input_names):
    added_nodes = set()
    input_nodes = [node for node in graph.nodes if node[1][1] and node[1][0] in input_names]
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


def extend_fallback_with_tm_ops(graph, fallback_nodes, max_depth, ops_of_interest, ops_to_avoid):
    logger.trace("Checking for fallback nodes based on perf...")
    
    # Gether all DiGraph output nodes (includes subgraphs too)
    output_nodes = [u for u, deg in graph.out_degree() if not deg]
    
    if len(output_nodes) != 1:
        # Subgraphs are often TVM functions represented as standalone graph, so we'll ignore them
        # as they are referenced in the original DiGraph
        logger.warning("Extended TM CPU Fallback: Multiple output nodes in DiGraph, using one with most ancestor nodes.")

    # Use graph which has most nodes (highly probability to be considered)
    # as main (most of subgraphs are wrapped TVM Functions with couple of ops)
    max_ancestors = 0
    output_node = None
    for out_n in output_nodes:
        ancestors = nx.ancestors(graph, out_n)
        if max_ancestors < len(ancestors):
            max_ancestors = len(ancestors)
            output_node = out_n
            
    # Initialize DiGraph attributes
    for node in graph.nodes():
        graph.nodes[node]['exec_on_cpu'] = False
        graph.nodes[node]['path_suitable_for_fallback'] = True
            
    # Traverse all paths until certain depth and mark valid candidates
    # as CPU fallback nodes
    tm_fallback_traverse(graph, output_node, max_depth, ops_of_interest, ops_to_avoid, 0)
    
    # Optional for debugging purposes
    print_extended_tm_fallback_graph = False
    if print_extended_tm_fallback_graph:
        # Color DiGraph for easier debugging. Meaning of each color:
        # - green - op of interest, but not on suitable fallback path
        # - blue - op is on suitable fallback path, but doesn't contain op of interest on path
        # - red - op is not suitable fallback path as it contains invalid op as one of its descendants
        # - purple - op is suitable for CPU execution as its on valid path and has ops of interests as descendants
        for node in graph.nodes():
            if graph.nodes[node]['exec_on_cpu']:
                graph.nodes[node]['color'] = 'green'
                
            if graph.nodes[node]['path_suitable_for_fallback']:
                graph.nodes[node]['color'] = 'blue'
            else:
                graph.nodes[node]['color'] = 'red'
                
            if graph.nodes[node]['exec_on_cpu'] and graph.nodes[node]['path_suitable_for_fallback']:
                graph.nodes[node]['color'] = 'purple'
        
        # Visualize DiGraph
        # 
        # Useful online visualizer: https://dreampuf.github.io/GraphvizOnline
        dot_graph = nx.nx_pydot.to_pydot(graph)
        print(dot_graph)
    
    # Gather additional CPU fallback nodes
    additional_fallback_ops = set()
    for node in graph.nodes():
        if not (graph.nodes[node]['exec_on_cpu'] and graph.nodes[node]['path_suitable_for_fallback']):
            continue
        
        op_name = str(node[1][0])
        logger.trace("Additional descendant for CPU: {}".format(op_name))
        
        additional_fallback_ops.add(node)
        
    return additional_fallback_ops | fallback_nodes

def tm_fallback_traverse(graph, current_node, max_depth, ops_of_interest, ops_to_avoid, current_depth):
    # Traversal depth exit condition
    if current_depth >= max_depth:
        logger.trace("Stopping traverse for given path as max allowed depth is reached")
        return
    current_depth += 1

    # Invalid paths exit condition
    for node, descendants in nx.bfs_successors(graph, current_node):
        valid_descendants_fallback_path = True
        if not descendants:
            continue

        for descendant in descendants:
            valid_descendants_fallback_path &= graph.nodes[descendant]["path_suitable_for_fallback"]
            
        if not valid_descendants_fallback_path:
            logger.trace("Stopping traverse for given path all descendant paths are invalid for fallback")
            return

    logger.trace("Current: {}".format(current_node[1][0]))
    predecessors = graph.predecessors(current_node)
    for predecessor in predecessors:
        node_desc = predecessor[1]
        op = node_desc[0]
        op_name = str(op)
        logger.trace("Predecessor: {}".format(op_name))
            
        # Handle ops to avoid
        if op_name in ops_to_avoid:
            graph.nodes[predecessor]["path_suitable_for_fallback"] &= False
        graph.nodes[predecessor]["path_suitable_for_fallback"] &= graph.nodes[current_node]["path_suitable_for_fallback"]
        
        # Handle conv based ops to avoid
        if op_name in ops_to_avoid and "conv" in op_name and "bias" not in op_name:
            graph.nodes[current_node]["exec_on_cpu"] = False
            graph.nodes[current_node]["path_suitable_for_fallback"] &= False
        
        # Handle ops of interest
        if op_name in ops_of_interest:
            graph.nodes[predecessor]["exec_on_cpu"] = True
            
            # Handle descendants if path is suitable for fallback
            for node, children in nx.bfs_successors(graph, predecessor):
                logger.trace("Correcting descendants for: {}".format(node))
                for child in children:
                    logger.trace("Descendant: {}".format(child))

                    if graph.nodes[child]['exec_on_cpu']:
                        logger.trace("Child ({}) already executes on CPU, breaking further traverse".format(child))
                        break

                    if graph.nodes[child]['path_suitable_for_fallback']:
                        logger.trace("Child ({}) is corrected to be executed on CPU".format(child))
                        graph.nodes[child]['exec_on_cpu'] = True
                    
        tm_fallback_traverse(graph, predecessor, max_depth, ops_of_interest, ops_to_avoid, current_depth)
        
    logger.trace("Finishing traverse for given path on depth: {}".format(current_depth))

class ConstructDiGraph(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.graph = nx.MultiDiGraph()
        self.fallback_nodes = set()
        self.node_indexer = NodeIndexer()
        self.names_used = {}

    def register_args(self, parent, parent_node):
        for arg in parent.args:
            if isinstance(arg, tvm.relay.expr.Var):
                pass
    
            self.graph.add_edge(node_hash(arg), parent_node)

    def visit_call(self, call):
        node = node_hash(call)
        self.node_indexer.count_visit("call", node)
        print(call.checked_type.__class__)
        if isinstance(call.op, tvm.ir.op.Op) and call.op.get_attr("target.pybuda_cpudevice") is not None:
            self.fallback_nodes.add(node)
            logger.info(f"Adding: {call.op} to fallback")
        elif (
            isinstance(call.op, tvm.relay.function.Function) 
            and isinstance(call.op.body, tvm.relay.expr.TupleGetItem)
            and call.op.body.tuple_value.op.get_attr("target.pybuda_cpudevice") is not None
        ):
            self.fallback_nodes.add(node)
            logger.info(f"Adding: {call.op.body.op} to fallback")

        elif (
            isinstance(call.op, tvm.relay.function.Function) 
            and not isinstance(call.op.body, tvm.relay.expr.TupleGetItem)
            and call.op.body.op.get_attr("target.pybuda_cpudevice") is not None
        ):
            self.fallback_nodes.add(node)
            logger.info(f"Adding: {call.op.body.op} to fallback")

        elif (
            isinstance(call.op, tvm.ir.op.Op) 
            and isinstance(call.checked_type, tvm.ir.tensor_type.TensorType)
            and len(call.checked_type.shape) > 4
        ):
            self.fallback_nodes.add(node)
            logger.info(f"Adding: {call.op} to fallback")

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
            logger.info(f"Adding: {call.op} to fallback")

        return super().visit_call(call)
    
    def visit_tuple_getitem(self, t):
        node = node_hash(t)
        self.register_args(t.tuple_value, node)
        return super().visit_tuple_getitem(t)

    def visit_tuple(self, t):
        tuple_node = node_hash(t)
        for producer in t.fields:
            producer_node = node_hash(producer)
            self.graph.add_edge(producer_node, tuple_node)
        return super().visit_tuple(t)


def get_relay_output(mod, params, inputs, target):
    # Build and Run Relay modules with inputs as (key : tensor) pair
    # Then, inputs dont need to be in the same order as 'mod' defines.
    ret_type = mod["main"].checked_type.ret_type
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build_module.build(mod, target=target, params=params)
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
        if fr_out.shape != tvm_out.shape:
            logger.error(f"Different shapes for outputs. Framework: {fr_out.shape}, TVM: {tvm_out.shape} after {compile_location}")

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
            logger.info("Max ATOL Delta: " + "{:.3e}".format(np.max(np.abs((fr_out - tvm_out))).item()) + ", atol=" +  "{}".format(atol))
            logger.info("Max RTOL Delta: " + "{:.3e}".format(np.max(np.abs((fr_out - tvm_out))/tvm_out).item()) + ", rtol=" + "{}".format(rtol))
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


def fallback_on_cpu(mod, compiler_cfg, input_names):
    import time
    
    logger.debug(f"Running cpu fallback compilation")
    logger.debug(f"Checking if the graph has any cpu-fallback ops...")
    start = time.time() 
    check_fallback_ops = CheckFallbackOps(compiler_cfg.cpu_fallback_ops)
    check_fallback_ops.visit(mod["main"])
    logger.debug(f"Done, took: {(time.time() - start):.2f} s")
    
    if check_fallback_ops.has_fallback_ops or compiler_cfg.enable_tm_cpu_fallback:
        logger.debug(f"Constructing digraph...")
        start = time.time()
        graph_constructor = ConstructDiGraph()
        graph_constructor.visit(mod["main"])
        logger.debug(f"Done, took: {(time.time() - start):.2f} s")
        
        # Visualize DiGraph
        #
        # Useful online visualizer: https://dreampuf.github.io/GraphvizOnline
        # dot_graph = nx.nx_pydot.to_pydot(graph_constructor.graph)
        # print(dot_graph)

        logger.debug(f"Finding and adding shared weights...")
        start = time.time()
        fallback_nodes = add_shared_weights_to_fallback(graph_constructor.graph, graph_constructor.fallback_nodes, input_names)
        
        # Extend fallback with valid TM ops from the end of the graph
        if compiler_cfg.enable_tm_cpu_fallback:
            fallback_nodes = extend_fallback_with_tm_ops(
                graph_constructor.graph, fallback_nodes,
                compiler_cfg.tm_cpu_fallback_max_depth,
                tm_cpu_fallback_ops_of_interest,
                tm_cpu_fallback_ops_to_not_include)
        
        logger.debug(f"Done, took: {(time.time() - start):.2f} s")
        logger.debug(f"Determining target for ops...")
        terget_determiner = DetermineTarget(graph_constructor.graph, fallback_nodes, compiler_cfg)
        new_mod = None
        start = time.time()
        terget_determiner.modify_graph = False
        mod["main"] = terget_determiner.visit(mod["main"])
        terget_determiner.modify_graph = True
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
        

        
    
def partition_for_buda(mod, graph_name, compiler_cfg, input_names=[]):
    initialize_pybuda_cpudevice_ops(mod, compiler_cfg)

    with tvm.transform.PassContext(opt_level=5):
        logger.trace("partition_for_buda:: At Entry")
        logger.trace(mod.functions)
        
        mod = tvm.transform.Sequential([transform.InferType()])(mod)
        logger.trace("After InferType")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.MergeComposite(pattern_table())])(mod)
        logger.trace("After MergeComposite")
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
            fallback_on_cpu(mod, compiler_cfg, input_names)

        mod = tvm.transform.Sequential([transform.AnnotateTarget(["pybuda_cpudevice", "pybuda"])])(mod)
        logger.trace("After AnnotateTarget")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.MergeCompilerRegions()])(mod)
        logger.trace("After MergeCompilerRegions")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.PartitionGraph(bind_constants=True)])(mod)
        logger.trace("After PartitionGraph")
        logger.trace(mod.functions)

        for k, v in mod.global_var_map_.items():
            if "cpudevice" in k:
                mod[v] = UnwrapPyBudaOpsForCPUFallback().visit(mod[v])
        logger.trace("After UnwrapPyBudaOpsForCPUFallback")
        logger.trace(mod.functions)

        # Unravel f(x) = x functions
        # Note that this applies only to tvm function calls where the input is returned as the output.
        # This will not change any identity op calls
        mod["main"] = IdentityFunctionUnraveller(mod).visit(mod["main"])
        logger.trace("After IdentityFunctionUnraveller")
        logger.trace(mod.functions)

        if not isinstance(mod["main"].body, tvm.relay.expr.Tuple):
            main_body_call_node = [mod["main"].body]
        else:
            main_body_call_node = mod["main"].body

        unsupported_op_names = []
        for item in main_body_call_node:
            if isinstance(item, tvm.relay.expr.Call):
                for arg in item.args:
                    if isinstance(arg, tvm.relay.expr.Var):
                        continue
                    if isinstance(arg, tvm.relay.expr.Tuple):
                        continue
                    if isinstance(arg, tvm.relay.expr.TupleGetItem):
                        continue
                    # if isinstance(arg, tvm.relay.expr.TupleGetItem):
                    #     # arg = arg.tuple_value().op
                    #     continue
                    
                    if not isinstance(arg.op, tvm.ir.expr.GlobalVar):
                        unsupported_op_names.append(arg.op.name)

                    assert arg.op in mod.global_var_map_.values(), mod["main"]
        
        if len(unsupported_op_names) > 0:
            print("Operators: " + str(unsupported_op_names) + " are unsupported.")
            assert False

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
        
    # Convert NaN attributes to Zeros
    for partition_key, partition_val in params.items():
        for param_key, param_val in params[partition_key].items():
            param_val_np = param_val.asnumpy()
            if np.isnan(param_val_np).all():
                zero_mtx = tvm.nd.array(np.zeros(param_val_np.shape).astype(param_val.dtype))
                params[partition_key][param_key] = zero_mtx

    dump_graph(mod, graph_name, "after_buda_partition")
        
    return mod, params
