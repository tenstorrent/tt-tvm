# SPDX-FileCopyrightText: © 2019-2023 The Apache Software Foundation © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=import-self, too-many-lines, len-as-condition, no-else-return, unused-variable, too-many-nested-blocks
# pylint: disable=consider-iterating-dictionary, invalid-name, unused-argument, unused-variable, broad-except
# pylint: disable=import-outside-toplevel, simplifiable-if-expression, cell-var-from-loop, unnecessary-lambda
# pylint: disable=missing-function-docstring, redefined-builtin, use-implicit-booleaness-not-comparison
"""PT: PyTorch frontend."""
import functools
import itertools
import math
import re
import sys
import logging
from typing import OrderedDict
import os

import numpy as np
import tvm
from tvm.ir import IRModule
from tvm.topi.utils import get_const_tuple

from .. import analysis as _analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .. import qnn, transform
from ..expr_functor import ExprMutator
from ..loops import while_loop
from ..prelude import Prelude, StaticTensorArrayOps
from ..ty import Any, TensorType, TupleType
from . import qnn_torch
from .common import AttrCvt, get_relay_op, gru_cell, logger, rnn_cell
from .common import infer_type as _infer_type
from .common import infer_shape as _infer_shape
from .common import infer_value as _infer_value
from .common import infer_value_simulated as _infer_value_simulated
from .common import lstm_cell, try_infer_value, unbind, fold_constant
from .common import set_span
from .pytorch_utils import is_version_greater_than, getattr_attr_name

from loguru import logger
__all__ = ["from_pytorch"]


# This returns a "subgraph" which puts variables whenever
# the type is known. It also records things to map the input
# nodes to the extracted graph's nodes.
# As Python objects are not round-trippable through C++, and
# our type annotations only live in Python, we need to map
# the nodes we get in visiting to the nodes
# we used to construct the graph (they are the same in C++,
# match each other in dictionary lookups, but are not the same
# in Python) by using the hint dictionary filled as
# {node: node for node in nodes} to get the type annotations.
# https://discuss.tvm.apache.org/t/round-tripping-objects-through-the-ffi/8440
class _TypeFinder(ExprMutator):
    def __init__(self, types):
        super().__init__()
        self.counter = 0
        self.vars = {}
        self.types = types
        self.leave = set()  # some variables are not inputs

    def visit_let(self, let):
        self.leave.add(let.var)
        return super().visit_let(let)

    def visit_function(self, fn):
        self.leave.update(fn.params)
        return super().visit_function(fn)

    def visit(self, expr):
        if expr in self.leave:
            return super().visit(expr)
        if expr in self.vars:
            return self.vars[expr]
        if isinstance(expr, tvm.relay.Var):
            self.vars[expr] = expr
            return expr
        if expr in self.types:
            ty = self.types[expr]
            v = tvm.relay.var(f"_{self.counter}", type_annotation=ty)
            self.counter += 1
            self.vars[expr] = v
            return v
        v = super().visit(expr)
        return v

class _CondEvaluator(ExprMutator):
    def __init__(self, types):
        super().__init__()

    def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        if call.op.name == "equal":
            if isinstance(new_args[0], _expr.Constant) and isinstance(new_args[1], _expr.Constant):
                eq = new_args[0].data.numpy() == new_args[1].data.numpy()
                return _wrap_const(eq)
        return call

def _should_construct_dynamic_list(list_construct_node):
    # if this list is element-accessed or modified at runtime, generate List ADT
    def inplace_add_to_add(op_name):
        if op_name == "aten::add_":
            return "aten::add"
        else:
            return op_name

    uses = _get_uses(list_construct_node)

    for loop_use in filter(lambda use: use.user.kind() == "prim::Loop", uses):
        block_input_index = loop_use.offset - 1
        block = list(loop_use.user.blocks())[0]
        list_loop_var = list(block.inputs())[block_input_index]
        uses += _get_uses(list_loop_var.node())

    op_names = map(inplace_add_to_add, set(use.user.kind() for use in uses))

    list_ops = set(["aten::add", "aten::__getitem__"])
    intersect = list_ops.intersection(op_names)

    if len(intersect) > 0 and intersect != set(["aten::add"]):
        return True

    # if add op outputs list, it is dynamic so we need to construct List ADT
    for use in filter(lambda use: use.user.kind() in ["aten::add", "aten::add_"], uses):
        output_type = _get_node_type(use.user)
        if output_type == "ListType":
            return True

    return False


def _is_int_seq(seq):
    # TODO (t-vi): handle non-int constants? (like numpy.intXX)
    return len(seq) > 0 and all([isinstance(i, int) for i in seq])

# operator implementation
class PyTorchOpConverter:
    """A helper class for holding PyTorch op converters."""

    def __init__(self, prelude, default_dtype, use_parser_friendly_name=False):
        self.prelude = prelude
        self.default_dtype = default_dtype
        self.create_convert_map()
        self.types = {}  # map from nodes to (Relay) type annotations
        self.source_map = {}  # map from graph node to its source name
        self.op_type_dict = {}  # map from op type to its presenting order
        self.current_op = []  # stack for recording current processing op
        self.use_parser_friendly_name = use_parser_friendly_name

    # this incrementally infers the type, see the comments on the type visitor
    # above.
    def infer_type(self, node, mod=None):
        """An incremental method to infer the type of a node in the relay graph."""

        if node in self.types:
            return self.types[node]
        if isinstance(node, tvm.relay.Var):
            return node.type_annotation

        tf = _TypeFinder(types=self.types)
        new_node = tf.visit(node)
        fn = _function.Function(list(tf.vars.values()), new_node)
        new_mod = IRModule({"main": fn})
        if mod is not None:
            new_mod.update(mod)
        new_mod = transform.RemoveUnusedFunctions()(new_mod)
        new_mod = transform.InferType()(new_mod)
        entry = new_mod["main"]
        ty = entry.body.checked_type
        self.types[node] = ty
        return self.types[node]

    def infer_type_with_prelude(self, val):
        body = self.infer_type(val, self.prelude.mod)
        return body

    # list ADT utilities
    def convert_to_list_adt(self, py_lst):
        elem_tys = [self.infer_type_with_prelude(elem) for elem in py_lst]
        msg = "List elements should have identical types"
        assert all(map(lambda ty: ty == elem_tys[0], elem_tys)), msg

        # get_type returns type_name, ctor1, ..., ctorN
        # 1 is nil
        _, cons, nil = self.prelude.mod.get_type("List")
        adt_lst = nil()
        for elem in reversed(py_lst):
            adt_lst = cons(elem, adt_lst)
        return adt_lst

    def map_tensor_array_constructor(self, adt_lst, shape):
        static_tensor_array_ops = StaticTensorArrayOps(self.prelude, "float32", shape)
        static_tensor_array_ops.register()
        tensor_create = self.prelude.get_tensor_ctor_static("tensor_constructor", "float32", shape)
        return self.prelude.map(tensor_create, adt_lst)

    def convert_to_tensor_array(self, adt_lst):
        _, cons, nil = self.prelude.mod.get_type("List")
        if self.prelude.length(adt_lst) == 0:
            return nil()

        checked_type = self.infer_type_with_prelude(self.prelude.hd(adt_lst))
        shape = checked_type.shape
        tensor_array = self.map_tensor_array_constructor(adt_lst, shape)
        return tensor_array, tuple(shape)

    def infer_shape(self, inputs, mod=None):
        """A method to get the output type of an intermediate node in the graph."""
        typ = self.infer_type(inputs, mod=mod)
        if hasattr(typ, "shape"):
            # Regular operator that outputs tensors
            return get_const_tuple(typ.shape)
        # The return type is not a tensor, for example List
        return typ

    def infer_shape_with_prelude(self, inputs):
        return self.infer_shape(inputs, mod=self.prelude.mod)

    def is_empty_shape(self, shape):
        rank = len(shape)
        if rank:
            is_empty = False
            for i in range(rank):
                if shape[i] == 0:
                    is_empty = True
                    break
            return is_empty
        else:
            return True

    def record_output_type(self, output):
        if isinstance(output, tuple):
            cleaned_output = [o for o in output if o is not None]
            types = self.infer_type_with_prelude(_expr.Tuple(cleaned_output))
            for o, t in zip(cleaned_output, types.fields):
                self.types[o] = t
        elif isinstance(output, _expr.Expr):
            self.infer_type_with_prelude(output)
        # it can also happen that the type is int or so

    def pytorch_promote_types(self, inputs, dtypes):
        """This promotes TVM inputs with TVM dtypes passed like PyTorch would"""
        actual_dtypes = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, _expr.Expr):
                idt = self.infer_type(inp).dtype
                actual_dtypes.append(idt)
            else:
                actual_dtypes.append(dtypes[i])
        dtypes = actual_dtypes
        tensor_dtypes = [dt for inp, dt in zip(inputs, dtypes) if not np.isscalar(inp)]
        non_tensor_inputs = [inp for inp in inputs if np.isscalar(inp)]
        result_type = _pytorch_result_type(tensor_dtypes, non_tensor_inputs)
        results = []
        for inp, dt in zip(inputs, dtypes):
            if np.isscalar(inp):
                results.append(_expr.const(inp, dtype=result_type))
            elif dt == result_type:
                results.append(inp)
            else:
                results.append(_op.cast(inp, result_type))
        return results

    def is_quantized_tensor(self, data):
        # If a quantized Torch module is saved and loaded back, dtype will be dropped
        # Since dtypes from Torch tensors are not reliable in such cases, we use
        # Relay's type inference result to decide if an input tensor is quantized
        ty = self.infer_type_with_prelude(data)
        return ty.dtype == "uint8"

    # Operator implementations
    def make_elemwise(self, name):
        def elemwise(inputs, input_types):
            if name == "divide":
                # https://pytorch.org/docs/stable/generated/torch.div.html#torch.div
                # None - default behavior. Performs no rounding and, if both input and
                # other are integer types, promotes the inputs to the default scalar type.
                if all(["int" in input_type for input_type in input_types[:2]]):
                    input_types[:2] = ["float32"] * 2
                    cast_inputs = []
                    for inp in inputs[:2]:
                        if np.isscalar(inp):
                            cast_inputs.append(_expr.const(inp, dtype="float32"))
                        else:
                            cast_inputs.append(_op.cast(inp, "float32"))
                    inputs[:2] = cast_inputs

            data0, data1 = self.pytorch_promote_types(inputs[:2], input_types[:2])
            return get_relay_op(name)(data0, data1)

        return elemwise

    def min_max_common(self, name_elemwise, name_reduce, inputs, input_types):
        if len(inputs) == 1:
            data = self.pytorch_promote_types(inputs[:1], input_types[:1])
            return get_relay_op(name_reduce)(data[0])
        elif len(inputs) >= 2 and isinstance(inputs[1], (list, int)):
            data = self.pytorch_promote_types(inputs[:1], input_types[:1])
            dim = inputs[1]
            keepdims = inputs[2] if len(inputs) > 2 else False
            # also return dummy indices
            return get_relay_op(name_reduce)(data[0], axis=dim, keepdims=keepdims), None
        else:
            data0, data1 = self.pytorch_promote_types(inputs[:2], input_types[:2])
            return get_relay_op(name_elemwise)(data0, data1)

    def max(self, inputs, input_types):
        return self.min_max_common("maximum", "max", inputs, input_types)

    def min(self, inputs, input_types):
        return self.min_max_common("minimum", "min", inputs, input_types)

    def maximum(self, inputs, input_types):
        data0, data1 = self.pytorch_promote_types(inputs[:2], input_types[:2])
        return _op.maximum(data0, data1)

    def minimum(self, inputs, input_types):
        data0, data1 = self.pytorch_promote_types(inputs[:2], input_types[:2])
        return _op.minimum(data0, data1)

    def make_unary(self, name):
        def unary(inputs, input_types):
            # this is just to ensure tensor input
            (data,) = self.pytorch_promote_types(inputs[:1], input_types[:1])
            return get_relay_op(name)(data)

        return unary

    def log1p(self, inputs, input_types):
        # 1_plus_log x = log(x + 1)
        (dtype,) = input_types
        one = _expr.const(1, dtype=dtype)
        return _op.log(inputs[0] + one)

    def square(self, inputs, input_types):
        (dtype,) = input_types
        return _op.power(inputs[0], _expr.const(2, dtype))

    def lerp(self, inputs, input_types):
        if len(inputs) != 3:
            msg = f"Wrong number of arguments ({len(inputs)}) to parse."
            raise AssertionError(msg)

        start = inputs[0]
        end = inputs[1]
        weight = inputs[2]
        return start + weight * (end - start)

    def arange(self, inputs, input_types):
        def _get_value(val, dtype):
            # dtype is a tvm dtype
            if isinstance(val, _expr.Expr):
                # since "arange" op will fill expr into its attribute
                # invoke set_span here to prevent expr-rewritten occurrs in span-filling stage
                source_name = self.source_map[self.current_op[-1]]
                inp = set_span(_op.cast(val, dtype), source_name)
                ret, _ = try_infer_value(inp, lambda ret: _expr.const(ret, dtype))
            else:
                ret = _create_typed_const(val, dtype)
            return ret

        def _get_type(val, inp_type):
            if isinstance(val, _expr.Expr):
                dtype = str(self.infer_type(val))
                return dtype
            return inp_type

        # PyTorch arange uses the following type semantics:
        # - if a dtype is given, start, stop, step are converted to that dtype
        # - if no dtype is given and all args are integral, dtype is int64
        # - if no dtype is given and there is a float arg, dtype is float32
        if len(inputs) in {5, 6, 7}:
            # inputs look like [_,_,_,dtype,layout,device,requires_grad]
            # therefore dtype_idx is always the length of inputs minus 4
            dtype_idx = len(inputs) - 4
            types = [_get_type(inputs[i], input_types[i]) for i in range(dtype_idx)]
            if inputs[dtype_idx] is not None:
                dtype = _convert_dtype_value(inputs[dtype_idx])
            elif any([t.startswith("float") for t in types]):
                dtype = "float32"
            else:
                dtype = "int64"

            # - if len(inputs) == 5, inputs = [stop, dtype, ...]
            # - if len(inputs) == 6, inputs = [start, stop, dtype, ...]
            # - if len(inputs) == 7, inputs = [start, stop, step, dtype, ...]
            start = _get_value(inputs[0], dtype) if len(inputs) > 5 else _expr.const(0, dtype)
            stop = _get_value(inputs[1 if len(inputs) > 5 else 0], dtype)
            step = _get_value(inputs[2], dtype) if len(inputs) > 6 else _expr.const(1, dtype)
        else:
            msg = f"Unknown number of arguments ({len(inputs)}) to parse."
            raise AssertionError(msg)

        return _op.transform.arange(start=start, stop=stop, step=step, dtype=dtype)

    def squeeze(self, inputs, input_types):
        data = inputs[0]
        if len(inputs) == 1:
            axis = None
        else:
            # TODO (t-vi): why is the cast to int needed? similarly elsewhere
            inputs = [inputs[1]] if not isinstance(inputs[1], list) else inputs[1]
            axis = [int(v) for v in inputs]

        return _op.transform.squeeze(data, axis)

    def unsqueeze(self, inputs, input_types):
        data = inputs[0]
        axis = inputs[1]

        return _op.transform.expand_dims(data, int(axis), 1)

    def concatenate(self, inputs, input_types):
        def tensor_array_concat(lst, axis):
            assert axis == 0, "Tensor array concat supported only for axis 0"
            tensor_array, shape = self.convert_to_tensor_array(lst)
            concat_shape = (Any(),) + shape[1:]
            concat = self.prelude.get_global_var_static("tensor_array_concat", "float32", shape)
            concatenated = concat(tensor_array)

            static_tensor_array_ops = StaticTensorArrayOps(self.prelude, "float32", concat_shape)
            static_tensor_array_ops.register()
            get_tensor = self.prelude.get_global_var_static(
                "tensor_get_data", "float32", concat_shape
            )
            return get_tensor(concatenated)

        data = inputs[0]
        axis = inputs[1]

        ref_dtype = _infer_type(data[0]).checked_type.dtype
        for i in range(1, len(data)):
            cur_dtype = _infer_type(data[i]).checked_type.dtype

            if ref_dtype != cur_dtype:
                data[i] = _op.cast(data[i], ref_dtype)

        if len(data) == 2 and len(_infer_shape(data[0])) != len(_infer_shape(data[1])) and (len(_infer_shape(data[0])) - len(_infer_shape(data[1]))) == 1:
            data[0] = _op.transform.squeeze(data[0], _op.const([0,]))

        if not isinstance(data, list):
            return tensor_array_concat(data, axis)

        if isinstance(data, _expr.Expr):
            data = [data]

        return _op.tensor.concatenate(data, int(axis))

    def slice(self, inputs, input_types):
        axis_dtype = "int64"
        index_size_limit = sys.maxsize
        data = inputs[0]
        dshape = self.infer_shape(data)
        ndim = len(dshape)
        dim = int(inputs[1])
        stride = inputs[4]

        target_begin, is_begin_const = try_infer_value(
            inputs[2], lambda ret: ret.astype(int).item(0)
        )
        target_end, is_end_const = try_infer_value(
            inputs[3], lambda ret: ret.astype(int).item(0)
        )

        # A fast path when slicing is nop.
        if (
            isinstance(target_begin, int)
            and isinstance(target_end, int)
            and target_begin == 0
            and target_end >= index_size_limit
            and stride == 1
        ):
            return data

        if target_begin is None and target_end is None:
            return data

        # Process begin
        begin = [0] * ndim

        if target_begin is not None:
            begin[dim] = target_begin

        if target_begin is not None and not isinstance(begin[dim], int):
            tmp = []
            for b in begin:
                if isinstance(b, int):
                    tmp.append(_op.expand_dims(_expr.const(b, axis_dtype), axis=0))
                else:
                    if len(self.infer_type(_op.cast(_op.expand_dims(b, axis=0), axis_dtype)).shape) > 1:
                        tmp.append(_op.cast(b, axis_dtype))
                    else:
                        tmp.append(_op.cast(_op.expand_dims(b, axis=0), axis_dtype))
            begin = _op.concatenate(tmp, axis=0)
            btype = self.infer_type(begin).dtype
            if str(btype) != axis_dtype:
                begin = _op.cast(begin, axis_dtype)

        # Process end
        if isinstance(target_end, int) and target_end >= index_size_limit:
            target_end = dshape[dim]

        if any([isinstance(d, tvm.tir.Any) for d in dshape]):
            end = _op.shape_of(data)
        else:
            end = dshape

        if isinstance(target_end, int):
            if isinstance(end, list):
                end[dim] = target_end
            else:
                all_static = True
                for i, shape_dim in enumerate(dshape):
                    if i != dim and isinstance(shape_dim, tvm.tir.Any):
                        all_static = False

                if all_static:
                    end = list(get_const_tuple(dshape))
                    end[dim] = target_end
                else:
                    target_end = _expr.const(target_end)
                    end = _op.scatter_elements(
                        end,
                        _op.expand_dims(_expr.const(dim), axis=0),
                        _op.expand_dims(target_end, axis=0),
                        axis=0,
                    )
        else:
            end = _op.cast(_op.shape_of(data), axis_dtype)
            if target_end is not None and not isinstance(target_end, tvm.tir.Any):
                ttype = self.infer_type(target_end).dtype
                if str(ttype) != axis_dtype:
                    target_end = _op.cast(target_end, axis_dtype)
                end = _op.scatter_elements(
                    end,
                    _op.expand_dims(_expr.const(dim), axis=0),
                    _op.expand_dims(target_end, axis=0),
                    axis=0,
                )

        if not isinstance(end, list):
            etype = self.infer_type(end).dtype
            if str(etype) != axis_dtype:
                end = _op.cast(end, axis_dtype)

        strides = [1] * ndim
        strides[dim] = stride

        return _op.transform.strided_slice(
            data, begin=begin, end=end, strides=strides, slice_mode="end"
        )

    def narrow(self, inputs, input_types):
        # Inputs are:
        # 0 - the tensor to narrow
        # 1 - the dimension along which to narrow
        # 2 - the starting dimension
        # 3 - the distance to the ending dimension
        # Lets find the ending dimension
        if inputs[2] < 0:
            input_shape = self.infer_shape(inputs[0])
            length = input_shape[inputs[1]]
            inputs[2] += length

        end = self.add(inputs[2:4], input_types[2:4])
        stride = 1
        slice_input = inputs[:3] + [end, stride]
        slice_types = input_types + ["int32"]
        return self.slice(slice_input, slice_types)

    def split(self, inputs, input_types):
        data = inputs[0]
        split_size = int(inputs[1])
        dim = int(inputs[2])

        split_index = split_size
        indices = []
        while split_index < self.infer_shape(data)[dim]:
            indices.append(split_index)
            split_index += split_size

        return _op.split(data, indices, dim)

    def split_with_sizes(self, inputs, input_types):
        data = inputs[0]
        sections = inputs[1]
        dim = int(inputs[2])

        if len(sections) == 1:
            # a special case used in torchvision detection models
            return _expr.TupleWrapper(_expr.Tuple([data]), 1)

        split_index = 0
        indices = []
        for i in range(len(sections) - 1):
            index, _ = try_infer_value(sections[i], lambda ret: int(ret))
            split_index += index
            indices.append(split_index)

        return _op.split(data, indices, dim)

    def tensor_split(self, inputs, input_types):
        # Reference: https://pytorch.org/docs/stable/generated/torch.tensor_split.html
        import torch

        if not isinstance(inputs[1], (int, list, tuple, torch.Tensor)):
            msg = (
                f"indices_or_sections type {type(inputs[1])} could not be parsed in "
                f"tensor_split op"
            )
            raise AssertionError(msg)

        if isinstance(inputs[1], torch.Tensor) and not (
            list(inputs[1].shape) == [] or list(inputs[1].shape) == 1
        ):
            msg = "indices_or_sections must be a zero-dimensional or one-dimensional long tensor"
            raise AssertionError(msg)

        if isinstance(inputs[1], int) or (
            isinstance(inputs[1], torch.Tensor) and list(inputs[1].shape) == []
        ):
            data = inputs[0]
            n = int(inputs[1])
            dim = int(inputs[2])

            split_size = int(self.infer_shape(data)[dim] / n)
            split_rest = int(self.infer_shape(data)[dim] % n)

            indices = []
            split_index = split_size
            if split_rest == 0:
                for i in range(n - 1):
                    indices.append(split_index)
                    split_index += split_size
            else:
                for i in range(split_rest):
                    indices.append(split_index + 1)
                    split_index = (i + 1) * (split_index + 1)
                for i in range(n - split_rest - 1):
                    split_index += split_size
                    indices.append(split_index)

            return _op.split(data, indices, dim)
        else:
            data = inputs[0]
            sections = inputs[1]
            dim = int(inputs[2])

            if isinstance(sections, tuple):
                sections = list(sections)
            elif isinstance(sections, torch.Tensor):
                sections = sections.cpu().numpy().tolist()

            return _op.split(data, sections, dim)

    def select(self, inputs, input_types):
        data = inputs[0]
        dim = int(inputs[1])
        index = _wrap_const(inputs[2])
        return _op.transform.take(data, index, axis=dim, mode="wrap")

    def take(self, inputs, input_types):
        data = inputs[0]
        indices = _op.cast(inputs[1], "int32")

        return _op.transform.take(data, indices=indices, mode="wrap")

    def topk(self, inputs, input_types):
        data = inputs[0]
        axis = int(inputs[2])
        is_ascend = not bool(inputs[3])
        sort = bool(inputs[4])

        if isinstance(inputs[1], _expr.Expr):
            k, _ = try_infer_value(inputs[1], lambda ret: ret.tolist())
        else:
            k = inputs[1]

        if not sort:
            msg = "Currently supports only sorted output for topk operator."
            raise AssertionError(msg)

        outs = _op.topk(data, k=k, axis=axis, is_ascend=is_ascend, ret_type="both", dtype="int64")

        return outs[0], outs[1]

    def reciprocal(self, inputs, input_types):
        data = inputs[0]
        return tvm.relay.reciprocal(data)

    def repeat(self, inputs, input_types):
        data = inputs[0]
        reps = []
        for r in inputs[1]:
            if isinstance(r, int):
                reps.append(r)
            else:
                reps.append(int(_infer_value(r, {}).numpy()))

        return _op.transform.tile(data, reps=reps)

    def repeat_interleave(self, inputs, input_types):
        data = inputs[0]
        if isinstance(inputs[1], int):
            repeats = inputs[1]
            axis = inputs[2]
        elif isinstance(inputs[1], _expr.Expr):
            if isinstance(inputs[1], _expr.Constant):
                repeats = int(inputs[1].data.numpy())
            else:
                repeats, _ = try_infer_value(inputs[1], lambda ret: ret.tolist())
            axis = inputs[2]
        else:
            msg = "Only repeat with one value as repeat is currently supported."
            raise AssertionError(msg)
        if axis is None:  # Flatten the data if no axis is given from torch
            data = _op.transform.reshape(data, [-1])
            axis = 0
        return _op.transform.repeat(data, repeats=repeats, axis=axis)

    def tensordot(self, input, input_types):
        dims = input[2:]
        x = input[0]
        y = input[1]
        xshape = self.infer_shape(x)
        yshape = self.infer_shape(y)

        # handle all types of inputs for `dims`
        if isinstance(dims, int):
            pairs = []
            for j in range(len(xshape)):
                for k in range(len(yshape)):
                    if xshape[j] == yshape[k]:
                        pairs.append((j, k))

            assert len(pairs) >= dims, f"Not enough matching dims, required = {dims}, given x = {x.shape}, y = {y.shape}"
            j = 0
            dim_count = dims
            dims = [[0,0], [0,0]]
            for pair in pairs[:dim_count]:
                dims[0][j] = pair[0]
                dims[1][j] = pair[1]
                j += 1

        elif isinstance(dims, (tuple, list)) and len(dims) == 1:
            dims = ((dims[0],), (dims[0],))
        elif isinstance(dims, (tuple, list)) and len(dims) > 1 and not isinstance(dims[0], (tuple, list)):
            dims = (dims, dims)

        # Handle negative dims
        dims = list(dims)
        dims[0] = list(dims[0])
        for j in range(len(dims[0])):
            if dims[0][j] < 0:
                dims[0][j] += len(xshape)

        dims[1] = list(dims[1])
        for j in range(len(dims[1])):
            if dims[1][j] < 0:
                dims[1][j] += len(yshape)

        dims[0] = tuple(dims[0])
        dims[1] = tuple(dims[1])
        dims = tuple(dims)

        assert len(dims[0]) == len(dims[1])
        for i in range(len(dims[0])):
            assert xshape[dims[0][i]] == yshape[dims[1][i]] or \
                xshape[dims[0][i]] == 1 or yshape[dims[1][i]] == 1

        alphabet = 'abcdefghijklmnopqrztuvwxyz'
        l = 0
        dim_to_char = OrderedDict()
        dim_to_char[0] = OrderedDict()
        dim_to_char[1] = OrderedDict()


        x_str = ""
        for i, j in enumerate(xshape):
            if i not in dim_to_char[0]:
                dim_to_char[0][i] = alphabet[l]
                l += 1
            x_str = x_str + dim_to_char[0][i]


        y_str = ""
        for i, j in enumerate(yshape):

            if i in dims[1]:
                if xshape[dims[0][dims[1].index(i)]] == j:
                    dim_to_char[1][i] = dim_to_char[0][dims[0][dims[1].index(i)]]
                else:
                    dim_to_char[1][i] = alphabet[l]
                    l += 1
            else:
                dim_to_char[1][i] = alphabet[l]
                l += 1

            y_str = y_str + dim_to_char[1][i]


        for dim in dims[0]:
            if dim in dim_to_char[0]:
                dim_to_char[0].pop(dim)

        for dim in dims[1]:
            if dim in dim_to_char[1]:
                dim_to_char[1].pop(dim)

        z_str = ""
        for k, v in dim_to_char[0].items():
            z_str = z_str + v

        for k, v in dim_to_char[1].items():
            z_str = z_str + v


        return _op.einsum([x, y], f"{x_str}, {y_str}->{z_str}")


    def addcdiv(self, inputs, input_types):
        data, t1, t2, c = self.pytorch_promote_types(inputs[:4], input_types[:4])
        return data + (c * (t1 / t2))

    def addcmul(self, inputs, input_types):
        data, t1, t2, c = self.pytorch_promote_types(inputs[:4], input_types[:4])
        return data + (c * (t1 * t2))

    def where(self, inputs, input_types):
        if len(inputs) == 1:
            return self.nonzero([inputs[0], True], input_types)

        cond = inputs[0]
        x, y = self.pytorch_promote_types(inputs[1:3], input_types[1:3])
        return _op.where(cond, x, y)

    def full_impl(self, data, fill_value, dtype):
        size = []
        need_reshape = False
        new_shape = []
        for dim in data:
            if isinstance(dim, _expr.Expr):
                if isinstance(dim, _expr.Constant):
                    dim = int(dim.data.numpy())
                    if isinstance(size, list):
                        size.append(dim)
                    new_shape.append(dim)
                else:
                    dim, success = try_infer_value(dim, lambda ret: int(ret), lambda: 0)
                    new_shape.append(dim)

                    if success:
                        if isinstance(size, list):
                            size.append(dim)
                    else:
                        size = None
                        need_reshape = True
            else:
                if isinstance(size, list):
                    size.append(dim)
                new_shape.append(dim)

        if size is None:
            tmp = []
            for dim in data:
                tmp.append(_op.cast(_op.expand_dims(dim, axis=0), "int64"))
            size = _op.concatenate(tmp, axis=0)

        if not isinstance(fill_value, _expr.Constant):
            if isinstance(fill_value, _expr.Expr):
                fill_value = _infer_value(fill_value, {})
            fill_value = _expr.const(fill_value, dtype=dtype)
        out = _op.full(fill_value, size, dtype=dtype)
        if need_reshape:
            out = _op.reshape(out, new_shape)
        return out

    def ones(self, inputs, input_types):
        data = inputs[0]

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = f"Data type {type(data)} could not be parsed in ones op"
            raise AssertionError(msg)

        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = self.default_dtype
        return self.full_impl(data, 1, dtype)

    def ones_like(self, inputs, input_types):
        data = inputs[0]
        out = _op.ones_like(data)

        # If the input and the output datatype is different, do a cast
        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = self.default_dtype
        if input_types[0] != dtype:
            out = _op.cast(out, dtype)

        return out

    def new_ones(self, inputs, input_types):
        size = inputs[1]

        import torch

        if not isinstance(size, (_expr.Expr, list, tuple, torch.Size, np.ndarray)):
            msg = f"Data type {type(size)} could not be parsed in ones op"
            raise AssertionError(msg)

        if inputs[2] is not None:
            dtype = _convert_dtype_value(inputs[2])
        else:
            dtype = input_types[0]
        return self.full_impl(size, 1, dtype)

    def zeros(self, inputs, input_types):
        data = inputs[0]

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = f"Data type {type(data)} could not be parsed in zeros op"
            raise AssertionError(msg)

        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = self.default_dtype
        return self.full_impl(data, 0, dtype)

    def zero_(self, inputs, input_types):
        data = inputs[0]
        shape = self.infer_shape(data)
        
        # Check if any shape dimension is symbolic (not an int)
        is_symbolic = any(not isinstance(dim, int) for dim in shape)
        if is_symbolic:
            return _op.full_like(data, fill_value=_expr.const(0))
        else:
            return self.full_impl(shape, 0, input_types[0])


    def zeros_like(self, inputs, input_types):
        data = inputs[0]
        out = _op.zeros_like(data)

        # If the input and the output datatype is different, do a cast
        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = self.default_dtype
        if input_types[0] not in dtype:
            out = _op.cast(out, dtype)

        return out


    def fill(self, inputs, input_type):
        data = inputs[0]
        fill_value = inputs[1]

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = "Data type %s could not be parsed in full op" % (type(data))
            raise AssertionError(msg)

        dtype = self.default_dtype

        return self.full_impl(self.infer_shape(data), fill_value, dtype)

    def full(self, inputs, input_types):
        data = inputs[0]
        fill_value = inputs[1]

        # Convert to scaler if provided values is TVM call expression and is not dependent on any inputs (i.e. constant)
        fill_value = _infer_value(fill_value, {}).numpy().item() if type(fill_value) == _expr.Call and len(_analysis.free_vars(fill_value)) == 0 else fill_value
        if isinstance(fill_value, tvm.relay.expr.TupleGetItem):
            fill_value = _infer_value(fill_value, {}).numpy().item()

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = f"Data type {type(data)} could not be parsed in full op"
            raise AssertionError(msg)

        if inputs[2] is not None:  # dtype given
            dtype = _convert_dtype_value(inputs[2])
        else:
            # if dtype is None, torch uses a global default set by torch.set_default_tensor_type()
            dtype = self.default_dtype

        return self.full_impl(data, fill_value, dtype)

    def full_like(self, inputs, input_types):
        data = inputs[0]
        fill_value = inputs[1]

        out = _op.full_like(data, _expr.const(fill_value))

        # If the input and the output datatype is different, do a cast
        if inputs[2] is not None:  # dtype given
            dtype = _convert_dtype_value(inputs[2])
        else:
            # if dtype is None, torch uses a global default set by torch.set_default_tensor_type()
            dtype = self.default_dtype
        if input_types[0] not in dtype:
            out = _op.cast(out, dtype)

        return out

    def new_full(self, inputs, input_types):
        data = inputs[1]
        fill_value = inputs[2]
        import torch

        if not isinstance(data, (_expr.Expr, list, tuple, torch.Size)):
            msg = f"Data type {type(data)} could not be parsed in full op"
            raise AssertionError(msg)

        if inputs[3] is not None:  # dtype given
            dtype = _convert_dtype_value(inputs[3])
        else:
            # if dtype is None, use the dtype of the input tensor
            dtype = self.infer_type(inputs[0])

        return self.full_impl(data, fill_value, dtype)

    def fill_(self, inputs, input_types):
        data = inputs[0]
        fill_value = inputs[1]
        if not isinstance(fill_value, (bool, int, float, complex)):
            fill_value = fold_constant(fill_value)
        return self.full_impl(self.infer_shape(data), fill_value, input_types[0])

    def linspace(self, inputs, input_types):
        start = inputs[0]
        stop = inputs[1]
        if isinstance(inputs[2],tvm.relay.expr.Constant) and inputs[2].data.shape == ():
            step = int(inputs[2].data.numpy())
        else:
            step = int(inputs[2])

        # Find the spacing between values as step
        if step != 1:
            step = (stop - start) / (step - 1)
            stop = stop + step
        else:
            stop = start + step

        if inputs[3] is None:
            import torch

            dtype = _convert_data_type(str(torch.get_default_dtype()))
        else:
            dtype = _convert_dtype_value(inputs[3])

        start = _create_typed_const(start, dtype)
        stop = _create_typed_const(stop, dtype)
        step = _create_typed_const(step, dtype)

        return _op.transform.arange(start=start, stop=stop, step=step, dtype=dtype)

    def relu(self, inputs, input_types):
        data = inputs[0]
        if self.is_quantized_tensor(data):
            assert len(inputs) == 3, "Input quant param not found in op inputs"
            input_zero_point = _expr.const(inputs[2], dtype="int32")
            return qnn_torch.quantized_relu(data, input_zero_point)
        return _op.nn.relu(data)

    def relu6(self, inputs, input_types):
        data = inputs[0]
        return _op.tensor.clip(data, 0.0, 6.0)

    def prelu(self, inputs, input_types):
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html#torch.nn.PReLU
        data = inputs[0]
        dim = self.get_dims(data)
        ndims = len(dim)
        axis = 0 if ndims == 1 else 1
        alpha = _op.broadcast_to(inputs[1], (dim[axis]))
        return _op.nn.prelu(data, alpha, axis)

    def leaky_relu(self, inputs, input_types):
        data = inputs[0]
        alpha = float(inputs[1])
        return _op.nn.leaky_relu(data, alpha)

    def elu(self, inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        disable_elu_handle_inf = bool(int(os.environ.get("PYBUDA_DISABLE_ELU_HANDLE_INF", 0)))
        if disable_elu_handle_inf:
            alpha = _expr.const(-float(inputs[1]), dtype=dtype)
            return alpha * _op.nn.relu(_expr.const(1, dtype=dtype) - _op.exp(data)) + _op.nn.relu(data)
        alpha = _expr.const(float(inputs[1]), dtype=dtype)
        data_shape = self.infer_shape(data)
        one_const = _expr.Constant(tvm.nd.array(np.ones(data_shape).astype(dtype)))
        zero_const = _expr.Constant(tvm.nd.array(np.zeros(data_shape).astype(dtype)))

        # Calculate the exponential of values less than 0 in 'data' to avoid 'inf' values for higher numbers.
        exp = _op.exp(_op.where(_op.less(data, zero_const), data, zero_const))

        # Exponential Linear Unit (ELU) function:
        #       ELU(data) =
        #           - If data > 0: data
        #           - If data <= 0: alpha * (exp(data) - 1)
        subtract = _op.subtract(exp, one_const)
        multiply = _op.multiply(alpha, subtract)
        return _op.where(_op.greater(data, zero_const), data, multiply)

    def celu(self, inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        alpha = _expr.const(float(inputs[1]), dtype=dtype)
        zero = _op.const(0, dtype)
        return alpha * _op.minimum(
            zero, _op.exp(data / alpha) - _expr.const(1, dtype=dtype)
        ) + _op.nn.relu(data)

    def gelu(self, inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        # gelu is data  * normcdf(data)
        # normcdf expressed as erf because we don't currently have that intrinsic
        # note that there is also a fastgelu variant approximating normcdf
        # with tanh and third order polynomials, but this is "true" gelu
        if len(inputs) == 2: # For pytorch 1.12
            if inputs[1] == 'none':
                return data * (
                    _expr.const(0.5, dtype=dtype)
                    + _op.erf(data * _expr.const(0.5**0.5, dtype=dtype)) * _expr.const(0.5, dtype=dtype)
                )
            elif inputs[1] == 'tanh':
                import math
                return (
                    (_op.tanh((data + (data * data * data * _expr.const(0.044715, dtype=dtype)))
                    * _expr.const(math.sqrt(2.0 / math.pi), dtype=dtype)) + _expr.const(1, dtype=dtype))
                    * (data * _expr.const(0.5, dtype=dtype))
                )

        else:
            return _op.gelu(data)
            # return data * (
            #     _expr.const(0.5, dtype=dtype)
            #     + _op.erf(data * _expr.const(0.5**0.5, dtype=dtype)) * _expr.const(0.5, dtype=dtype)
            # )

    def selu(self, inputs, input_types):
        data = inputs[0]
        # https://pytorch.org/docs/stable/nn.html#selu
        dtype = input_types[0]
        alpha = _expr.const(-1.6732632423543772848170429916717, dtype=dtype)
        gamma = _expr.const(1.0507009873554804934193349852946, dtype=dtype)
        return gamma * (
            alpha * _op.nn.relu(_expr.const(1.0, dtype=dtype) - _op.exp(data)) + _op.nn.relu(data)
        )

    def silu(self, inputs, input_types):
        data = inputs[0]
        return data * _op.tensor.sigmoid(data)

    def glu(self, inputs, input_types):
        """
        Applies the gated linear unit function GLU(a,b)= a * sigmoid(b)
        where a is the first half of the input matrices and b is the second half.
        Link: https://pytorch.org/docs/stable/generated/torch.nn.GLU.html
        """
        data = inputs[0]
        dim = inputs[1]
        relay_tup = _op.transform.split(data, 2, dim)
        return relay_tup[0] * _op.tensor.sigmoid(relay_tup[1])

    def log_sigmoid(self, inputs, input_types):
        data = inputs[0]
        mn = _op.minimum(_op.const(0, dtype=input_types[0]), data)
        z = _op.exp(-_op.abs(data))
        return mn - self.log1p([z], input_types)

    def cross_entropy_loss_with_logits(self, inputs, input_types):
        input = inputs[0]
        target = inputs[1]
        weights = inputs[2]
        reduction = inputs[3]
        ignore_index = inputs[4]
        label_smoothing = inputs[5]
        input_shape = self.infer_shape(input)
        target_shape = self.infer_shape(target)
        if input_shape != target_shape:
            if reduction == 0:
                reduction = "none"
            elif reduction == 1:
                reduction = "mean"
            else:
                reduction = "sum"
            num_class = self.infer_shape(input)[1]
            if weights is None:
                weights = _op.full(_expr.const(1), (num_class,), dtype=input_types[0])
            return _op.nn.nll_loss(
                _op.nn.log_softmax(input), target, weights, reduction, ignore_index
            )
        assert reduction == 1, "reduction not supported in cross_entropy_loss"
        assert ignore_index == -100, "ignore_index not supported in cross_entropy_loss"
        assert label_smoothing == 0.0, "label_smoothing not supported in cross_entropy_loss"
        assert weights is None, "weight not supported in cross_entropy_loss"
        return _op.nn.cross_entropy_with_logits(_op.nn.log_softmax(input), target)

    def l1_loss(self, inputs, input_types):
        assert len(inputs) == 3
        [predictions, targets, reduction] = inputs
        delta = _op.abs(_op.subtract(predictions, targets))
        if reduction == 0:
            # reduction = "none"
            return delta
        elif reduction == 1:
            # reduction = "mean"
            return _op.mean(delta)
        else:
            # reduction = "sum"
            return _op.sum(delta)

    def mse_loss(self, inputs, input_types):
        assert len(inputs) == 3
        [predictions, targets, reduction] = inputs
        delta = _op.subtract(predictions, targets)
        delta = _op.power(delta, _expr.const(2, input_types[0]))
        if reduction == 0:
            # reduction = "none"
            return delta
        elif reduction == 1:
            # reduction = "mean"
            return _op.mean(delta)
        else:
            # reduction = "sum"
            return _op.sum(delta)

    def hard_sigmoid(self, inputs, input_types):
        def _relu6(x):
            return _op.tensor.clip(x, 0.0, 6.0)

        def func(x):
            return _relu6(x + _expr.const(3.0)) / _expr.const(6.0)

        if self.is_quantized_tensor(inputs[0]):
            input_scale = _expr.const(inputs[1])
            input_zero_point = _expr.const(inputs[2])
            # PyTorch seems to use the following output qparams, but accuracy
            # is broken if we use this.
            # TODO(masahi): Revisit this parameter choice
            #
            # Taken from src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp
            # output_scale = _expr.const(0.00390625)  # 1.0 / 2^8
            # output_zero_point = _expr.const(-128)
            output_scale = input_scale
            output_zero_point = input_zero_point

            data = qnn.op.dequantize(inputs[0], input_scale, input_zero_point, axis=1)
            out = func(data)
            return qnn.op.quantize(out, output_scale, output_zero_point, out_dtype="uint8")

        return func(inputs[0])

    def hard_swish(self, inputs, input_types):
        data = inputs[0]
        return data * self.hard_sigmoid(inputs, input_types)

    def adaptive_avg_pool(self, op, inputs, input_types):
        data = inputs[0]
        output_size = self.convert_const_list(inputs[1])

        def func(x):
            return op(x, output_size=output_size)

        if self.is_quantized_tensor(data):
            return qnn_torch.apply_with_upcast(data, func)

        return func(data)

    def adaptive_max_pool(self, op, inputs, input_types):
        data = inputs[0]
        output_size = inputs[1]
        for i, item in enumerate(output_size):
            if isinstance(item, _expr.Constant):
                # convert Constant to int
                output_size[i] = item.data.numpy()[()]
        # returns dummy indices too
        return op(data, output_size=output_size), None

    @staticmethod
    def convert_const_list(data):
        if isinstance(data, (list, _expr.Tuple)):
            data = list(data)
            for i, _ in enumerate(data):
                if isinstance(data[i], _expr.Expr):
                    data[i] = int(_infer_value_simulated(data[i], {}).numpy())
        return data

    def maxpool_2d(self, inputs, input_types):
        data = inputs[0]

        pool_size = self.convert_const_list(inputs[1])
        strides = self.convert_const_list(inputs[2] if inputs[2] else pool_size)
        padding = self.convert_const_list(inputs[3])
        dilation = self.convert_const_list(inputs[4])
        ceil_mode = int(inputs[5])

        data_shape = self.infer_shape(data)
        if len(data_shape) == 3:
            return _op.nn.max_pool1d(
                data,
                pool_size=pool_size[-1],
                strides=strides[-1],
                dilation=dilation[-1],
                padding=padding[-1],
                layout="NCW",
                ceil_mode=ceil_mode,
            )
        else:
            return _op.nn.max_pool2d(
                data,
                pool_size=pool_size,
                strides=strides,
                dilation=dilation,
                padding=padding,
                layout="NCHW",
                ceil_mode=ceil_mode,
            )

    def maxpool_2d_with_indices(self, inputs, input_types):
        # returns dummy indices too
        return self.maxpool_2d(inputs, input_types), None

    def maxpool_1d(self, inputs, input_types):
        data = inputs[0]

        pool_size = inputs[1]
        strides = inputs[2] if inputs[2] else pool_size
        padding = inputs[3]
        dilation = inputs[4]
        ceil_mode = int(inputs[5])

        return _op.nn.max_pool1d(
            data,
            pool_size=pool_size,
            strides=strides,
            dilation=dilation,
            padding=padding,
            layout="NCW",
            ceil_mode=ceil_mode,
        )

    def maxpool_3d(self, inputs, input_types):
        data = inputs[0]

        need_squeeze = False
        if len(self.get_dims(data)) == 4:
            need_squeeze = True
            data = _op.expand_dims(data, 0)
        pool_size = inputs[1]
        strides = inputs[2] if inputs[2] else pool_size
        padding = inputs[3]
        dilation = inputs[4]
        ceil_mode = int(inputs[5])

        res = _op.nn.max_pool3d(
            data,
            pool_size=pool_size,
            strides=strides,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
        )
        return res if not need_squeeze else _op.squeeze(res, [0])

    def hardtanh(self, inputs, input_types):
        a = inputs[0]
        tanh_min = float(inputs[1])
        tanh_max = float(inputs[2])
        return _op.tensor.clip(a, tanh_min, tanh_max)

    def conv2d(self, inputs, input_types):

        inputs[3] = self.convert_const_list(inputs[3])
        inputs[4] = self.convert_const_list(inputs[4])
        inputs[5] = self.convert_const_list(inputs[5])

        # Add no output padding and move groups from inputs[6] to inputs[8]
        inputs.append([0, 0])
        inputs.append(inputs[6])
        inputs[6] = 0
        return self.convolution(inputs, input_types)

    def convolution(self, inputs, input_types):
        # Use transpose or normal
        use_transpose = True if inputs[6] == 1 else False

        data = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        strides = tuple(inputs[3])
        padding = tuple(inputs[4])
        dilation = tuple(inputs[5])

        if isinstance(weight, _expr.Expr):
            inferred_shape = self.infer_shape(weight)
            weight_shape = []
            for infer in inferred_shape:
                weight_shape.append(infer)
        else:
            msg = f"Data type {type(weight)} could not be parsed in conv op"
            raise AssertionError(msg)

        groups = int(inputs[8])

        if use_transpose:
            channels = weight_shape[1] * groups
            in_channels = weight_shape[0]
        else:
            channels = weight_shape[0]
            in_channels = weight_shape[1]

        # Check if this is depth wise convolution
        # We need to reshape weight so that Relay could recognize this is depth wise
        # weight_shape[1] is always in_channels // groups
        # For depthwise, in_channels == groups, so weight_shape[1] == 1
        # If groups > 1 but weight_shape[1] != 1, this is group convolution
        if groups > 1 and in_channels == 1:
            channel_multiplier = channels // groups
            new_weight_shape = (groups, channel_multiplier) + tuple(weight_shape[2:])
            weight = _op.transform.reshape(weight, new_weight_shape)

        kernel_size = weight_shape[2:]
        use_bias = isinstance(bias, _expr.Expr)

        # We are trying to invoke various relay operations through a single conv_op variable.
        # However the function signatures for some operations have additional attributes so we
        # pass these in along with the standard ones.
        additional_arguments = dict()

        if use_transpose:
            if len(kernel_size) == 3:
                conv_op = _op.nn.conv3d_transpose
            elif len(kernel_size) == 2:
                conv_op = _op.nn.conv2d_transpose
            else:
                conv_op = _op.nn.conv1d_transpose
            output_padding = tuple(inputs[7])
            additional_arguments["output_padding"] = output_padding

        else:
            if len(kernel_size) == 3:
                conv_op = _op.nn.conv3d
            elif len(kernel_size) == 2:
                conv_op = _op.nn.conv2d
            else:
                conv_op = _op.nn.conv1d

        if len(kernel_size) == 3:
            data_layout = "NCDHW"
            kernel_layout = "OIDHW"
            if use_transpose:
                # Transposed convolutions have IODHW layout.
                kernel_layout = "IODHW"
        elif len(kernel_size) == 2:
            data_layout = "NCHW"
            kernel_layout = "OIHW"
            if use_transpose:
                # Transposed convolutions have IOHW layout.
                kernel_layout = "IOHW"
        else:
            data_layout = "NCW"
            kernel_layout = "OIW"
            if use_transpose:
                # Transposed convolutions have IOW layout.
                kernel_layout = "IOW"

        # Conv1d does not currently support grouped convolution so we convert it to conv2d
        is_grouped_conv1d = False
        if groups > 1 and len(kernel_size) == 1 and not use_transpose:
            is_grouped_conv1d = True
            conv_op = _op.nn.conv2d
            kernel_size = [1] + kernel_size
            strides = (1,) + strides
            padding = (0,) + padding
            dilation = (1,) + dilation
            data = _op.expand_dims(data, axis=2)
            weight = _op.expand_dims(weight, axis=2)
            data_layout = "NCHW"
            kernel_layout = "OIHW"

        conv_out = conv_op(
            data,
            weight,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            channels=channels,
            kernel_size=kernel_size,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_layout="",
            out_dtype="",
            **additional_arguments,
        )
        if use_bias:
            res = _op.nn.bias_add(conv_out, bias)
        else:
            res = conv_out
        if is_grouped_conv1d:
            # Because we conducted grouped conv1d convolution through conv2d we must
            # squeeze the output to get the correct result.
            res = _op.squeeze(res, axis=[2])
        return res

    def softmax(self, inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        if isinstance(axis, str):
            axis = int(axis)

        return _op.nn.softmax(data, axis=axis)

    def threshold(self, inputs, input_types):
        data = inputs[0]
        threshold_f = float(inputs[1])
        threshold_ = _op.full_like(inputs[0], fill_value=_expr.const(threshold_f))
        value_f = float(inputs[2])
        value = _op.full_like(inputs[0], fill_value=_expr.const(value_f))
        return _op.where(_op.greater(data, threshold_), data, value)

    def contiguous(self, inputs, input_types):
        return inputs[0]

    def batch_norm(self, inputs, input_types):
        data = inputs[0]
        data_type = input_types[0]
        channels = self.infer_shape(data)

        scale = isinstance(inputs[1], _expr.Expr)
        if scale:
            gamma = inputs[1]
        else:
            gamma = _create_typed_const(np.ones([int(channels[1])]), data_type)

        center = isinstance(inputs[2], _expr.Expr)
        if center:
            beta = inputs[2]
        else:
            beta = _create_typed_const(np.zeros([int(channels[1])]), data_type)

        moving_mean = inputs[3]
        if moving_mean is None:
            # moving_mean = _create_typed_const(np.zeros([int(channels[1])]), data_type)
            moving_mean = _op.mean(data, axis=[0, 2 , 3,], keepdims=False)

        moving_var = inputs[4]
        if moving_var is None:
            # moving_var = _create_typed_const(np.zeros([int(channels[1])]), data_type)
            moving_var = _op.variance(data, axis=[0, 2, 3], keepdims=False)

        epsilon = float(inputs[7])

        return _op.nn.batch_norm(
            data,
            gamma,
            beta,
            moving_mean,
            moving_var,
            axis=1,
            epsilon=epsilon,
            center=center,
            scale=scale,
        )[0]

    def instance_norm(self, inputs, input_types):
        data = inputs[0]
        data_type = input_types[0]
        channels = self.infer_shape(data)
        running_mean = inputs[3]
        running_var = inputs[4]
        use_input_stats = inputs[5]

        if isinstance(inputs[1], _expr.Expr) and isinstance(inputs[2], _expr.Expr):
            scale = center = True
            weight = inputs[1]
            beta = inputs[2]
            gamma = weight
        else:
            scale = center = False

        if not scale:
            gamma = _create_typed_const(np.ones([int(channels[1])]), data_type)

        if not center:
            beta = _create_typed_const(np.zeros([int(channels[1])]), data_type)

        epsilon = float(inputs[7])

        if not use_input_stats:
            return _op.nn.batch_norm(
                data,
                gamma,
                beta,
                running_mean,
                running_var,
                axis=1,
                epsilon=epsilon,
                center=center,
                scale=scale,
            )[0]

        return _op.nn.instance_norm(
            data, gamma, beta, axis=1, epsilon=epsilon, center=center, scale=scale
        )

    def get_dims(self, data):
        import torch

        if isinstance(data, _expr.Expr):
            dims = self.infer_shape(data)
        elif isinstance(data, list):
            dims = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            dims = data.shape
        else:
            msg = f"Data type {type(data)} could not be parsed"
            raise AssertionError(msg)
        return dims

    def layer_norm(self, inputs, input_types):
        data = inputs[0]
        ndims = len(self.get_dims(inputs[1]))
        input_shape = self.infer_type(data).shape
        axis = len(input_shape) - 1
        assert ndims == 1, "Support only normalization over last one dimension."
        
        # When elementwise_affine=False in PyTorch LayerNorm, weight and bias are None.
        # However, TVM's layer_norm op expects gamma and beta to be valid Relay expressions.
        # So we explicitly create constant expressions: gamma=1.0, beta=0.0 to mimic no affine transform.
        
        if inputs[2] is None and inputs[3] is None:
            normalized_shape = input_shape[-1]
            dtype = self.infer_type(data).dtype
    
            # Create constant gamma = 1.0
            gamma = _expr.const(1.0, dtype)
            gamma = _op.full(gamma, [normalized_shape], dtype)
            
            # Create constant beta = 0.0
            beta = _expr.const(0.0, dtype)
            beta = _op.full(beta, [normalized_shape], dtype)
    
        else:
            gamma = inputs[2]
            beta = inputs[3]
    
        return _op.layernorm(
            data,
            gamma=gamma,
            beta=beta,
            axis=axis,
            eps=float(inputs[4]),
        )

    def group_norm(self, inputs, input_types):
        data = inputs[0]
        gamma = inputs[2]
        beta = inputs[3]
        num_groups = inputs[1]
        epsilon = float(inputs[4])

        return _op.nn.group_norm(
            data,
            gamma=gamma,
            beta=beta,
            num_groups=num_groups,
            axis=1,
            epsilon=epsilon,
            center=True,
            scale=True,
        )

    def transpose(self, inputs, input_types):
        data = inputs[0]

        import torch

        if isinstance(data, _expr.Expr):
            ndims = len(self.infer_shape_with_prelude(data))
        elif isinstance(data, list):
            ndims = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            ndims = data.shape
        else:
            msg = f"Data type {type(data)} could not be parsed in transpose op"
            raise AssertionError(msg)

        if isinstance(data, tvm.runtime.NDArray):
            ndims = len(data.shape)
        axes = list(range(ndims))

        num_inputs = len(inputs)

        if num_inputs == 1:
            if ndims >= 2:
                axes[-1] = ndims - 2
                axes[-2] = ndims - 1
            if not isinstance(data, _expr.Expr):
                data = _expr.const(data)

        elif num_inputs == 3:
            parse = lambda i: ndims * (i < 0) + i
            src, dst = [parse(int(inputs[i])) for i in [1, 2]]
            axes[src] = dst
            axes[dst] = src
        else:
            axes = inputs[1]
        return _op.transform.transpose(data, axes)

    def numpy_T(self, inputs, input_types):
        data = inputs[0]
        shape = self.infer_shape(data)
        if len(shape) != 2:
            logger.warning(
                "The use of Tensor.T on tensors of dimensions != 2 is deprecated"
                "and will be removed in a future release of PyTorch."
            )
        return _op.transform.transpose(data)

    def flatten(self, inputs, input_types):
        data = inputs[0]
        start = int(inputs[1])
        end = int(inputs[2])
        dshape = get_const_tuple(self.infer_shape_with_prelude(data))
        ndim = len(dshape)
        if start < 0:
            start += ndim
        if end < 0:
            end += ndim
        assert start <= end, "start dim cannot come after end dim"
        new_shape = [0] * start

        new_shape.append(-1)
        squeeze_axes = []
        for i in range(start + 1, end + 1):
            new_shape.append(1)
            squeeze_axes.append(i)
        for _ in range(end + 1, ndim):
            new_shape.append(0)
        out = _op.reshape(data, new_shape)
        if squeeze_axes:
            out = _op.squeeze(out, axis=squeeze_axes)
        return out

    def addmm(self, inputs, input_types):
        input_mat = inputs[0]
        mat1 = inputs[1]
        mat2 = inputs[2]
        beta = inputs[3]
        alpha = inputs[4]
        data_type = input_types[1]

        transposed_mat2 = _op.transform.transpose(mat2, axes=[1, 0])
        units = self.infer_shape(transposed_mat2)[0]
        dense_out = _op.nn.dense(mat1, transposed_mat2, units=units)

        if not isinstance(alpha, _expr.Expr) and alpha != 1:
            alpha = _create_typed_const(alpha, data_type)
            dense_out *= alpha

        if not isinstance(beta, _expr.Expr) and beta != 1:
            beta = _create_typed_const(beta, data_type)
            input_mat *= beta

        return dense_out + input_mat

    def size(self, inputs, input_types):
        shape = list(self.infer_shape_with_prelude(inputs[0]))
        axis = None
        if len(inputs) > 1:
            axis = int(inputs[1])

        if any(map(lambda s: isinstance(s, tvm.tir.expr.Any), shape)):
            if axis is None or isinstance(shape[axis], tvm.tir.expr.Any):
                shape_dynamic = _op.shape_of(inputs[0], dtype="int32")
                if axis is not None:
                    return _op.take(shape_dynamic, _expr.const(axis), 0)
                return shape_dynamic

        if axis is not None:
            return _expr.const(shape[axis])
        return _expr.const(shape)

    def numtotensor(self, inputs, input_types):
        val = inputs[0]
        dtype = input_types[0]

        if isinstance(val, _expr.Expr):
            return val

        if isinstance(val, tvm.tir.IntImm):
            val = val.__int__()
            dtype = int

        arr = val * np.ones([]).astype(dtype)
        return arr

    def tensortonum(self, inputs, input_types):
        return inputs[0]

    def view(self, inputs, input_types):
        data = inputs[0]
        if len(inputs) == 3:
            shape_inp = [inputs[1], self.infer_shape(inputs[2])[0]]
        else:
            if isinstance(inputs[1], list):
                shape_inp = inputs[1]
            elif isinstance(inputs[1], _expr.Tuple):
                shape_inp = list(inputs[1].fields)
                # return inputs[0]
            else:
                shape_inp = self.infer_shape(inputs[1])
        new_shape = shape_inp
        for i, shape in enumerate(shape_inp):
            if isinstance(shape, _expr.Expr):
                val = _infer_value_simulated(shape, {})
                new_shape[i] = val.numpy().item(0)

        if isinstance(data, _expr.Constant):
            old_shape = data.data.shape
            num_new_dims = len(new_shape) - len(old_shape)

            if num_new_dims > 1:
                data = _op.transform.expand_dims(data, -1, num_new_dims)
        return _op.transform.reshape(data, new_shape)

    def view_as(self, inputs, input_types):
        data = inputs[0]
        tensors = inputs[1]

        if not isinstance(tensors, (_expr.Call, _expr.Constant, _expr.Var)):
            msg = f"Data type {type(tensors)} could not be parsed in view_as op"
            raise AssertionError(msg)

        shape = self.infer_shape(tensors)

        return _op.transform.reshape(data, shape)

    def reshape(self, inputs, input_types):
        data = inputs[0]
        new_shape = inputs[1]

        tmp_shape = []
        is_dyn = False
        for s in new_shape:
            if isinstance(s, _expr.Constant):
                tmp_shape.append(int(s.data.numpy()))
            elif isinstance(s, _expr.Expr):
                dim, success = try_infer_value(s, lambda ret: int(ret))
                tmp_shape.append(dim)

                if not success:
                    is_dyn = True
            else:
                tmp_shape.append(s)

        if is_dyn:
            new_shape = []
            for i, s in enumerate(tmp_shape):
                if not isinstance(s, _expr.Expr):
                    s = _expr.const(s, "int64")
                else:
                    s = _op.cast(s, "int64")
                new_shape.append(_op.expand_dims(s, axis=0))
            new_shape = _op.concatenate(new_shape, axis=0)
        else:
            new_shape = tmp_shape
        return _op.transform.reshape(data, new_shape)

    def reshape_as(self, inputs, input_types):
        data = inputs[0]
        new_shape = self.infer_shape(inputs[1])
        tmp_shape = []
        is_dyn = False
        for s in new_shape:
            if isinstance(s, _expr.Constant):
                tmp_shape.append(int(s.data.numpy()))
            elif isinstance(s, _expr.Expr):
                dim, success = try_infer_value(s, lambda ret: int(ret))
                tmp_shape.append(dim)

                if not success:
                    is_dyn = True
            else:
                tmp_shape.append(s)

        if is_dyn:
            new_shape = []
            for i, s in enumerate(tmp_shape):
                if not isinstance(s, _expr.Expr):
                    s = _expr.const(s, "int64")
                else:
                    s = _op.cast(s, "int64")
                new_shape.append(_op.expand_dims(s, axis=0))
            new_shape = _op.concatenate(new_shape, axis=0)
        else:
            new_shape = tmp_shape
        return _op.transform.reshape(data, new_shape)

    def pixel_shuffle(self, inputs, input_types):
        data = inputs[0]
        upscale_factor = inputs[1]
        return _op.transform.pixel_shuffle(data, upscale_factor)
        upscale_squared = upscale_factor * upscale_factor
        b, c, h, w = self.infer_shape(data)
        assert (
            c % upscale_squared == 0
        ), "input channel should be divisible by square of upscale_factor"

        ndims = len(self.infer_shape_with_prelude(data))
        axes = list(range(ndims))
        num_inputs = len(inputs)
        oc = c // upscale_squared
        oh = h * upscale_factor
        ow = w * upscale_factor

        new_shape = [b, oc, upscale_factor, upscale_factor, h, w]
        out_shape = [b, oc, oh, ow]

        data = _op.transform.reshape(data, new_shape)
        # The data will be transposed to
        # [b, oc, h, upscale_factor, w, upscale_factor]
        # for further reshape
        axes = [0, 1, 4, 2, 5, 3]
        data = _op.transform.transpose(data, axes)
        return _op.transform.reshape(data, out_shape)

    def clone(self, inputs, input_types):
        data = inputs[0]
        return _op.tensor.copy(data)

    def log_softmax(self, inputs, input_types):
        data = inputs[0]
        axis = int(inputs[1])
        return _op.nn.log_softmax(data, axis)

    def sigmoid(self, inputs, input_types):
        data = inputs[0]

        def func(x):
            return _op.tensor.sigmoid(x)

        if self.is_quantized_tensor(data):
            assert len(inputs) == 5, "Input/Ouput quant param not found in op inputs"
            return qnn_torch.quantized_sigmoid(inputs)

        return func(data)

    def softplus(self, inputs, input_types):
        dtype = input_types[0]
        beta = _expr.const(float(inputs[1]), dtype=dtype)
        threshold = int(inputs[2]) if inputs[2] else 20
        threshold_ = _op.full_like(inputs[0], fill_value=_expr.const(threshold))
        softplus_value = _op.log(_op.exp(inputs[0] * beta) + _expr.const(1.0, dtype=dtype)) / beta
        return _op.where(_op.greater(inputs[0] * beta, threshold_), inputs[0], softplus_value)

    def make_avg_pool(self, dim):
        def avg_pool(inputs, input_types):
            data = inputs[0]

            pool_size = self.convert_const_list(inputs[1])
            strides = self.convert_const_list(inputs[2] if inputs[2] else pool_size)
            padding = inputs[3]
            ceil_mode = int(inputs[4])
            count_include_pad = int(inputs[5])

            def func(x):
                if dim == 1:
                    return _op.nn.avg_pool1d(
                        x,
                        pool_size=pool_size,
                        strides=strides,
                        padding=padding,
                        dilation=(1,),
                        ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad,
                    )
                elif dim == 2:
                    return _op.nn.avg_pool2d(
                        x,
                        pool_size=pool_size,
                        strides=strides,
                        padding=padding,
                        dilation=(1, 1),
                        ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad,
                    )
                elif dim == 3:
                    return _op.nn.avg_pool3d(
                        x,
                        pool_size=pool_size,
                        strides=strides,
                        padding=padding,
                        dilation=(1, 1, 1),
                        ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad,
                    )
                else:
                    msg = "Average Pooling dimension should be between 1 and 3"
                    raise RuntimeError(msg)

            if self.is_quantized_tensor(data):
                return qnn_torch.apply_with_upcast(data, func)

            return func(data)

        return avg_pool

    def linear(self, inputs, input_types):
        # https://pytorch.org/docs/stable/nn.functional.html#linear
        # 0 - input
        # 1 - weight
        bias = inputs[2]
        a_shape = self.infer_shape_with_prelude(inputs[0])
        b_shape = self.infer_shape_with_prelude(inputs[1])
        if len(a_shape) == 2 and len(b_shape) == 2:
            mm_out = _op.nn.dense(inputs[0], inputs[1])
        elif len(b_shape) == 1:
            mm_out = self.matmul([inputs[0], inputs[1]], input_types[:2])
        else:
            mm_out = self.matmul(
                [inputs[0], _op.transpose(inputs[1], axes=(1, 0))], input_types[:2]
            )
        if isinstance(bias, _expr.Expr):
            bias_ndims = len(self.infer_shape_with_prelude(bias))
            if bias_ndims == 1:
                return _op.nn.bias_add(mm_out, bias, axis=-1)
            mm_dtype = self.infer_type_with_prelude(mm_out).dtype
            return self.add([mm_out, bias], [mm_dtype, input_types[2]])
        return mm_out

    def dropout(self, inputs, input_types):
        data = inputs[0]
        rate = float(inputs[1])

        return _op.nn.dropout(data, rate)

    def make_reduce(self, name):
        def reduce(inputs, input_types):
            data = inputs[0]
            axis = None
            keepdims = False

            if len(inputs) > 2:  # default, torch have only data, axis=None, keepdims=False
                if inputs[1] is not None:
                    if isinstance(inputs[1], int):
                        axis = int(inputs[1])
                    elif _is_int_seq(inputs[1]):
                        axis = inputs[1]
                    elif inputs[1]:
                        axis = list(self.infer_shape(inputs[1]))
                keepdims = bool(inputs[2])

            return get_relay_op(name)(data, axis=axis, keepdims=keepdims)

        return reduce

    def norm(self, inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        axis = None
        keepdims = False
        if len(inputs) > 3:
            axis = inputs[2]
            keepdims = bool(inputs[3])

        order = inputs[1]
        if order == np.inf:
            return _op.reduce.max(_op.abs(data), axis=axis, keepdims=keepdims)
        elif order == np.NINF:
            return _op.reduce.min(_op.abs(data), axis=axis, keepdims=keepdims)
        else:
            reci_order = _expr.const(1.0 / order, dtype=dtype)
            order = _op.cast(_expr.const(order), dtype)
            return _op.power(
                _op.reduce.sum(_op.power(_op.abs(data), order), axis=axis, keepdims=keepdims),
                reci_order,
            )

    def frobenius_norm(self, inputs, input_types):
        data = inputs[0]
        axis = None
        keepdims = False
        if len(inputs) > 2:
            axis = inputs[1] if len(inputs[1]) > 0 else None
            keepdims = bool(inputs[2])

        return _op.sqrt(_op.reduce.sum((data * data), axis=axis, keepdims=keepdims))

    def std(self, inputs, input_types):
        data = inputs[0]
        if len(inputs) == 2:
            axis = None
            keepdims = False
            unbiased = bool(inputs[1])
        else:
            axis = inputs[1]
            keepdims = bool(inputs[3])
            unbiased = bool(inputs[2])

        return _op.reduce.std(data, axis=axis, keepdims=keepdims, unbiased=unbiased)

    def variance(self, inputs, input_types):
        data = inputs[0]
        if len(inputs) == 2:
            axis = None
            keepdims = False
            unbiased = bool(inputs[1])
        else:
            axis = inputs[1]
            keepdims = bool(inputs[3])
            unbiased = bool(inputs[2])

        return _op.reduce.variance(data, axis=axis, keepdims=keepdims, unbiased=unbiased)

    def mean(self, inputs, input_types):
        data = inputs[0]

        if inputs[1]:
            axis = inputs[1]
        else:
            axis = None

        if len(inputs) > 2 and inputs[2]:
            keepdims = int(inputs[2])
        else:
            keepdims = False
        if len(inputs) > 3 and inputs[3]:
            exclude = int(inputs[3])
        else:
            exclude = False

        def func(x):
            return _op.mean(x, axis, keepdims, exclude)

        if self.is_quantized_tensor(data):
            assert len(inputs) == 6, "Input quant param not found in op inputs"
            input_scale = _expr.const(inputs[4])
            input_zero_point = _expr.const(inputs[5])
            # refer to aten/src/ATen/native/quantized/cpu/qreduction.cpp
            return qnn_torch.apply_with_fp32_fallback(data, input_scale, input_zero_point, func)

        return func(data)

    def var_mean(self, inputs, input_types):
        data = inputs[0]
        if len(inputs) == 2:
            axis = None
            keepdims = False
            unbiased = bool(inputs[1])
        else:
            axis = inputs[1]
            keepdims = bool(inputs[3])
            unbiased = bool(inputs[2])

        m, v = _op.reduce.mean_variance(data, axis, keepdims, False, unbiased)
        return v, m

    def chunk(self, inputs, input_types):
        data = inputs[0]

        if isinstance(inputs[1],tvm.relay.expr.Constant) and inputs[1].data.shape == ():
            num_chunks = int(inputs[1].data.numpy())
        else:
            num_chunks = int(inputs[1])

        axis = int(inputs[2])

        if isinstance(data, _expr.Expr):
            inferred_shape = self.infer_shape_with_prelude(data)

        shape = []
        for infer in inferred_shape:
            shape.append(infer)

        dim = int(shape[axis])

        if dim % num_chunks:
            unif_size = int(dim / (num_chunks - 1))
        else:
            unif_size = int(dim / num_chunks)

        indeces = []
        for i in range(unif_size, dim, unif_size):
            indeces.append(i)

        return _op.split(data, indeces, axis)

    def baddbmm(self, inputs, _):
        input = inputs[0]
        batch1, batch2 = inputs[1:3]
        beta = _expr.const(float(inputs[3]))
        alpha = _expr.const(float(inputs[4]))
        return beta * input + alpha * _op.nn.batch_matmul(batch1, batch2, transpose_b=False)

    def matmul(self, inputs, input_types):

        inputs_0 = inputs[0]
        inputs_1 = inputs[1]

        # Need to check input shape as batch matmul must be supported.
        a_shape = self.infer_shape_with_prelude(inputs_0)
        b_shape = self.infer_shape_with_prelude(inputs_1)

        # When performing a batch matmul, we need to properly handle N-dim shapes.
        if len(a_shape) > 2 and len(b_shape) > 2:
            # Convert a into a 3 dimensional tensors.
            need_reshape_output = False
            if len(a_shape) != 3:
                a = _op.reshape(inputs_0, [-1, a_shape[-2], a_shape[-1]])
                need_reshape_output = True
            else:
                a = inputs_0

            # Transpose matrix dimensions of b.
            trans_axes = list(range(len(b_shape)))
            trans_axes[-2], trans_axes[-1] = trans_axes[-1], trans_axes[-2]
            b = _op.transpose(inputs_1, trans_axes)

            # Convert b into a 3 dimensional tensor. Note that the last two dimensions
            # are transposed.
            if len(b_shape) != 3:
                b = _op.reshape(b, [-1, b_shape[-1], b_shape[-2]])

            # Perform a batch matmul.
            output = _op.nn.batch_matmul(a, b)
            
            # Compute batch shape for final reshape operation.
            batch_shape = tuple(max(a, b) for a, b in zip(a_shape[:-2], b_shape[:-2]))  

            # Reshape output to original dimensions.
            if need_reshape_output:
                return _op.reshape(output, [*batch_shape, a_shape[-2], b_shape[-1]])
            return output

        elif len(a_shape) > 2 and len(b_shape) == 1:
            inputs_0 = _op.reshape(inputs_0, [-1, a_shape[-2], a_shape[-1]])
            inputs_1 = _op.reshape(inputs_1, [1, b_shape[0], 1])
            broadcast_shape = [
                (int)(np.prod(a_shape) // (a_shape[-2] * a_shape[-1])), b_shape[0], 1
            ]
            inputs_1 = _op.broadcast_to(inputs_1, broadcast_shape)
            output = _op.nn.batch_matmul(inputs_0, inputs_1, transpose_a=False, transpose_b=False)
            return _op.squeeze(output, axis=[-1,])

        elif len(a_shape) == 1 and len(b_shape) > 2:
            inputs_0 = _op.reshape(inputs_0, [1, 1, a_shape[0]])
            inputs_1 = _op.reshape(inputs_1, [-1, b_shape[-2], b_shape[-1]])
            broadcast_shape = [
                (int)(np.prod(b_shape) // (b_shape[-2] * b_shape[-1])), 1, a_shape[0],
            ]
            inputs_0 = _op.broadcast_to(inputs_0, broadcast_shape)
            output = _op.nn.batch_matmul(inputs_0, inputs_1, transpose_a=False, transpose_b=False)
            return _op.squeeze(output, axis=[-2,])

        elif len(a_shape) > 2:
            inputs_0 = _op.reshape(inputs_0, [-1, a_shape[-1]])
        elif len(a_shape) == 1:
            return _op.squeeze(_op.nn.matmul(_op.expand_dims(inputs_0, axis=0), inputs_1), axis=[0])

        if len(b_shape) > 2:
            trans_axes = list(range(len(b_shape)))
            trans_axes[-2], trans_axes[-1] = trans_axes[-1], trans_axes[-2]
            input_1 = _op.reshape(_op.transpose(inputs_1, trans_axes), [-1, b_shape[-2]])
        elif len(b_shape) == 2:
            input_1 = _op.transpose(inputs_1, axes=(1, 0))
        elif len(b_shape) == 1:
            input_1 = _op.expand_dims(inputs_1, 0, 1)

        out = _op.nn.dense(inputs_0, input_1)

        if len(b_shape) == 1:
            out = _op.squeeze(out, axis=[-1])

        # Reshape output into a N dimensional tensor when a or b dim > 2
        if len(a_shape) > 2:
            out = _op.reshape(out, [*a_shape[:-1], b_shape[-1]])
        elif len(b_shape) > 2:
            out = _op.reshape(out, [a_shape[-2], -1, b_shape[-1]])
            out = _op.reshape(
                _op.transpose(out, [1, 0, 2]), [*b_shape[:-2], a_shape[-2], b_shape[-1]]
            )

        return out

    def expand(self, inputs, input_types):
        data_in = inputs[0]
        shape = list(self.infer_shape(data_in))

        ndims = len(shape)
        sizes = inputs[1]
        out = data_in

        out_dims = len(sizes)
        if ndims < out_dims:
            num_newaxis = out_dims - ndims
            out = _op.expand_dims(out, axis=0, num_newaxis=num_newaxis)
            shape = [1] * num_newaxis + shape

        for i in range(out_dims):
            if sizes[i] != -1 and shape[i] == 1:
                if not isinstance(sizes[i], int):
                    sizes[i] = int(_infer_value(sizes[i], {}).numpy())
                out = _op.repeat(out, sizes[i], axis=i)

        return out

    def int(self, inputs, input_types):
        if isinstance(inputs[0], _expr.Expr):
            return inputs[0]
        return int(inputs[0])

    def identity(self, inputs, input_types):
        return inputs[0]

    def none(self, inputs, input_types):
        return None

    def pad_common(self, mode, pad_value, inputs, input_types):
        data = inputs[0]
        if isinstance(inputs[1], list):
            pad_list = inputs[1]
        else:
            pad_list = list(self.infer_shape(inputs[1]))

        # initialize paddings based on input len
        pad_len = len(self.infer_shape(data)) * 2
        paddings = [0] * pad_len

        if len(pad_list) >= 2:
            paddings[-1] = pad_list[1]
            paddings[-2] = pad_list[0]
        if len(pad_list) >= 4:
            paddings[-3] = pad_list[3]
            paddings[-4] = pad_list[2]
        if len(pad_list) >= 6:
            paddings[-5] = pad_list[5]
            paddings[-6] = pad_list[4]

        # group into tuple of 2 ints
        paddings = [paddings[i : i + 2] for i in range(0, len(paddings), 2)]

        const_paddings = []
        non_zero_found = False
        for pad in paddings:
            const_paddings.append([])
            for p in pad:
                if isinstance(p, _expr.Expr):
                    p = int(_infer_value(p, {}).numpy())
                elif not isinstance(p, int):
                    raise NotImplementedError("pad width should be int/expr")
                const_paddings[-1].append(p)
                if p != 0:
                    non_zero_found = True

        if not non_zero_found:
            return data
        elif mode == "constant":
            return _op.nn.pad(data, const_paddings, pad_value=pad_value, pad_mode=mode)
        else:
            return _op.nn.pad(data, const_paddings, pad_mode=mode)

    def pad(self, inputs, input_types):
        # mode: Optional default "constant"
        if len(inputs) > 2 and inputs[2] is not None:
            mode = inputs[2]
        else:
            mode = "constant"

        # pad_value: Optional default 0
        if len(inputs) == 4 and inputs[3] is not None:
            pad_value = inputs[3]
        else:
            pad_value = 0

        # replicate is edge in TVM's padding mode
        if mode == "replicate":
            mode = "edge"
        elif mode == "circular":
            raise ValueError("circular mode for torch.nn.functional.pad are not supported in TVM")
        return self.pad_common(mode, pad_value, inputs, input_types)

    def constant_pad_nd(self, inputs, input_types):
        return self.pad_common("constant", _expr.const(inputs[2]), inputs, input_types)

    def reflection_pad1d(self, inputs, input_types):
        return self.pad_common("reflect", 0, inputs, input_types)

    def reflection_pad2d(self, inputs, input_types):
        return self.pad_common("reflect", 0, inputs, input_types)

    def replication_pad1d(self, inputs, input_types):
        return self.pad_common("edge", 0, inputs, input_types)

    def replication_pad2d(self, inputs, input_types):
        return self.pad_common("edge", 0, inputs, input_types)

    def replication_pad3d(self, inputs, input_types):
        return self.pad_common("edge", 0, inputs, input_types)

    def clamp_common(self, data, min=None, max=None):
        def get_v(v, default_v):
            if isinstance(v, _expr.Constant):
                return float(v.data.numpy())
            if isinstance(v, _expr.Expr):
                infer_v, success = try_infer_value(v, lambda ret: float(ret))
                if success:
                    return infer_v
            if v is not None:
                return v
            return default_v

        dtype = self.infer_type(data).dtype

        type_info = np.finfo(dtype) if "float" in dtype else np.iinfo(dtype)

        # TODO(masahi): Properly handle inf in a one-way clamp case.
        if min is not None and max is not None:
            amin = get_v(min, type_info.min)
            amax = get_v(max, type_info.max)
        elif min is not None:
            amin = get_v(min, type_info.min)
            amax = type_info.max
        else:
            amin = type_info.min
            amax = get_v(max, type_info.max)

        return _op.clip(data, amin, amax)

    def clamp(self, inputs, _):
        return self.clamp_common(inputs[0], min=inputs[1], max=inputs[2])

    def clamp_min(self, inputs, input_types):
        return self.clamp_common(inputs[0], min=inputs[1])

    def clamp_max(self, inputs, input_types):
        return self.clamp_common(inputs[0], max=inputs[1])

    def to(self, inputs, input_types):
        data = inputs[0]
        dtype = inputs[1] if inputs[1] is not None and not isinstance(inputs[1], str) else inputs[2]
        # special handling for aten::to(data, 6, _, _, _) case
        # 6 means dtype = float
        # this happens when converting upsampling with scale factor
        cast_map = {5: "float16", 6: "float32", 7: "float64", 3: "int32", 4: "int64"}

        cast_func = {5: float, 6: float, 7: float, 3: int, 4: int}

        ret = data
        if isinstance(data, _expr.Expr):
            actual_dtype = str(self.infer_type(data).dtype)
            if dtype in cast_map and cast_map[dtype] != actual_dtype:
                ret = _op.cast(data, cast_map[dtype])
        elif dtype in cast_map:
            ret = cast_func[dtype](data)

        return ret

    def get_upsample_out_size(self, inputs, method):
        # This assumes a static shape
        out_size = []
        if inputs[1] is not None:
            for size in inputs[1]:
                if not isinstance(size, int):
                    out_size.append(int(_infer_value(size, {}).numpy()))
                else:
                    out_size.append(size)
        else:
            scale_index = 3 if method != "nearest_neighbor" else 2
            scales = inputs[scale_index]
            assert scales is not None, "neither out size nor scale provided"
            assert isinstance(scales, list)
            ishape = self.infer_shape(inputs[0])
            for i, scale in enumerate(scales):
                out_size.append(int(math.floor(float(ishape[2 + i]) * scale)))

        return out_size

    def make_upsample(self, method):
        def upsample(inputs, input_types):
            data = inputs[0]
            out_size = self.get_upsample_out_size(inputs, method)

            if len(inputs) > 2 and method != "nearest_neighbor":
                align_corners = inputs[2]
            else:
                align_corners = False

            if method == "nearest_neighbor":
                coord_trans = "asymmetric"
            elif align_corners:
                coord_trans = "align_corners"
            else:
                coord_trans = "half_pixel"

            def func(x):
                return _op.image.resize2d(
                    x, out_size, None, "NCHW", method, coord_trans, cubic_alpha=-0.75
                )

            if self.is_quantized_tensor(data):
                # input qparams are manually appended by us
                assert isinstance(inputs[-2], float)
                assert isinstance(inputs[-1], int)
                input_scale = _expr.const(inputs[-2])
                input_zero_point = _expr.const(inputs[-1])
                # currently piggy backs to fp32, it gets identical output as torch
                return qnn_torch.apply_with_fp32_fallback(data, input_scale, input_zero_point, func)

            return func(data)

        return upsample

    def make_upsample3d(self, method):
        def upsample3d(inputs, input_types):
            data = inputs[0]
            out_size = self.get_upsample_out_size(inputs, method)

            if len(inputs) > 2 and method == "linear":
                align_corners = inputs[2]
            else:
                align_corners = False

            if method == "nearest_neighbor":
                coord_trans = "asymmetric"
            elif align_corners:
                coord_trans = "align_corners"
            else:
                coord_trans = "half_pixel"

            return _op.image.resize3d(data, out_size, None, "NCDHW", method, coord_trans)

        return upsample3d

    def expand_as(self, inputs, input_types):
        target = inputs[1]
        t0 = self.infer_type(inputs[0]).dtype
        t1 = self.infer_type(inputs[1]).dtype
        if str(t0) != str(t1):
            target = _op.cast(target, t0)
        return _op.broadcast_to_like(inputs[0], target)

    def broadcast_tensors(self, inputs, input_types):
        tensor_list = inputs[0]
        import torch

        infer_shape_value = [self.infer_shape(t) for t in tensor_list]
        # "torch.broadcast_shapes" is available after PyTorch 1.8.0
        if hasattr(torch, "broadcast_shapes"):
            res_shape = list(torch.broadcast_shapes(*infer_shape_value))
        else:
            res_shape = list(torch.broadcast_tensors(*map(torch.empty, infer_shape_value))[0].shape)
        return [_op.broadcast_to(tensor, res_shape) for tensor in tensor_list]

    def broadcast_to(self, inputs, input_types):
        tensor_list = inputs[1]

        if type(tensor_list) is list:
            res_shape = tensor_list
        else:
            res_shape = [x.data.numpy().item() for x in tensor_list]

        return _op.broadcast_to(inputs[0], res_shape)

    def Bool(self, inputs, input_types):
        assert len(inputs) == 1
        return inputs[0]

    def Float(self, inputs, input_types):
        assert len(inputs) == 1
        return _op.cast(inputs[0], "float32")

    def bitwise_not(self, inputs, input_types):
        data = inputs[0]
        # The input tensor must be of integral or Boolean types.
        # For bool tensors, it computes the logical NOT
        if input_types[0] == "bool":
            out = _op.logical_not(_op.cast(data, "bool"))
        else:
            out = _op.bitwise_not(_op.cast(data, "int"))

        return out

    def bitwise_xor(self, inputs, input_types):
        lhs = inputs[0]
        rhs = inputs[1]
        lhs = _op.cast(lhs, "bool") if input_types[0] == "bool" else _op.cast(lhs, "int")
        rhs = _op.cast(rhs, "bool") if input_types[1] == "bool" else _op.cast(rhs, "int")

        return _op.bitwise_xor(lhs, rhs)

    def logical_not(self, inputs, input_types):
        data = _wrap_const(inputs[0])
        return _op.logical_not(_op.cast(data, "bool"))

    def logical_xor(self, inputs, input_types):
        lhs = _op.cast(inputs[0], "bool")
        rhs = _op.cast(inputs[1], "bool")

        return _op.logical_xor(lhs, rhs)

    def getitem(self, inputs, input_types):
            return self.prelude.nth(inputs[0], _wrap_const(inputs[1]))

    def list_len(self, inputs, input_types):
        if isinstance(inputs[0], _expr.Constant):
            if len(inputs[0].data.shape) == 1:
                return inputs[0].data.shape[0]
        return self.prelude.length(inputs[0])

    def type_as(self, inputs, input_types):
        assert len(inputs) == 2
        assert len(input_types) == 2
        return _op.cast(inputs[0], input_types[1])

    def gather(self, inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        indices = inputs[2]

        return _op.gather(data, axis, indices)

    def add(self, inputs, input_types):
        # add_ is overloaded for tensor add and list concat
        if input_types[0] == "ListType":
            return self.prelude.concat(inputs[0], inputs[1])
        return self.make_elemwise("add")(inputs, input_types)

    def tensor_array_stack(self, inputs, input_types):
        dim = inputs[1]
        assert dim == 0, "stacking on a dynamic tensor list only supported on a first axis"
        tensor_array, shape = self.convert_to_tensor_array(inputs[0])

        stacked_shape = (Any(),) + shape
        stack = self.prelude.get_global_var_static("tensor_array_stack", "float32", shape)
        stacked = stack(tensor_array)

        static_tensor_array_ops = StaticTensorArrayOps(self.prelude, "float32", stacked_shape)
        static_tensor_array_ops.register()
        get_tensor = self.prelude.get_global_var_static("tensor_get_data", "float32", stacked_shape)
        return get_tensor(stacked)

    def stack(self, inputs, input_types):
        if isinstance(inputs[0], list):
            # a static python list of tensors
            dim = inputs[1]
            return _op.stack(inputs[0], dim)
        else:
            # List ADT case
            assert isinstance(inputs[0], _expr.Expr)
            ty = self.infer_type_with_prelude(inputs[0])
            list_ty = self.prelude.mod.get_global_type_var("List")
            msg = "The input list is expected to be List ADT"
            assert isinstance(ty, tvm.ir.TypeCall) and ty.func == list_ty, msg
            return self.tensor_array_stack(inputs, input_types)

    def sub(self, inputs, input_types):
        if len(inputs) == 3:
            data0, data1, alpha = self.pytorch_promote_types(inputs, input_types)
            return get_relay_op("subtract")(data0, alpha * data1)
        else:
            data0, data1 = self.pytorch_promote_types(inputs, input_types)
            return get_relay_op("subtract")(data0, data1)

    def rsub(self, inputs, input_types):
        data0, data1, alpha = self.pytorch_promote_types(inputs, input_types)

        # note: rsub means data0 and data1 swap places
        return get_relay_op("subtract")(data1, alpha * data0)

    def embedding(self, inputs, input_types):
        weight = inputs[0]
        indices = inputs[1]
        # Check the type of indices
        indicies_dtype = _infer_type(indices).checked_type.dtype
        if indicies_dtype != "int32" and indicies_dtype != "int64":
            # we want to cast indices to int32 if they are not already.
            # However, If indices are int64, there is no need to cast them to int32
            # because forge doesn't support int64 and indices will be cast to int32
            # during lowering from torch to forge. Therefore adding cast of int64 to int32
            #  will result in cast op that casts int32 to int32. This is not only redundant, but also
            #  exposes a few bugs in tt-mlir https://github.com/tenstorrent/tt-mlir/issues/1215
            logger.warning("Casting input indices of embedding op from {} to int32", indicies_dtype)
            indices = tvm.relay.cast(indices, "int32")
        return _op.embedding(weight, indices, axis=0)

    def embedding_bag(self, inputs, input_types):

        weight = inputs[0]
        indices = inputs[1]

        modes = ["sum", "mean", "max"]
        mode = modes[inputs[4]]

        offsets = inputs[2]
        #TODO: Support multiple bags
        num_bags = self.infer_shape(offsets)[0] if isinstance(offsets, tvm.relay.expr.Call) else offsets.type_annotation.concrete_shape[0]
        assert num_bags == 1, "embedding_bag currently only supports a single bag"

        take = []
        take.append(_op.embedding(weight, indices.astype("int32"), axis=0))

        if mode == "sum":
            out = _op.sum(take[0], axis=0, keepdims=True)
        elif mode == "mean":
            out = _op.mean(take[0], axis=0, keepdims=True)
        else:
            out = _op.max(take[0], axis=0, keepdims=True)
        return (out,)

    def one_hot(self, inputs, input_types):
        indices = inputs[0].astype("int32")
        num_classes = inputs[1]
        if num_classes == -1:
            msg = "Inferring the number of classes is not yet supported."
            raise NotImplementedError(msg)

        dtype = "int32"
        on_value = tvm.relay.const(1.0, dtype)
        off_value = tvm.relay.const(0.0, dtype)

        return _op.one_hot(indices, on_value, off_value, num_classes, -1, dtype)

    def index(self, inputs, input_types):
        data = inputs[0]
        data_shape = self.infer_type(data).shape

        indices = inputs[1]
        # Cover case when first row is selected as whole in Python style using ':' (e.g. x[:, mask])
        # while second is selected using a boolean mask
        if len(_infer_shape(data)) == 2 and len(indices) == 1 and indices[0] == None:
            # Remove first None argument (represents ':')
            indices.pop(0)

            indices = indices[0]
            if len(_analysis.free_vars(indices)) == 0:
                if isinstance(indices, _expr.Constant):
                    # Extract direct data as numpy array
                    indices = indices.data.numpy()
                else:
                    # Infer values to get constant data (discards irelevant math which
                    # will eventually be optimized out)
                    indices = _infer_value(indices, {}).numpy()
                indices = np.transpose(np.argwhere(indices))
                indices = indices[0] if len(indices.shape) == 2 and indices.shape[0] == 1 else indices
                indices = _expr.const(indices)

            # Iterate over each row and extract the selected columns
            res = []
            for dim in range(_infer_shape(data)[0]):
                partial_res = _op.take(data, _expr.const(dim), 0)
                partial_res = _op.adv_index([partial_res, indices])
                partial_res = _op.expand_dims(partial_res, 0)
                res.append(partial_res)
            res = _op.concatenate(res, 0)

            return res

        # Cover cases when first dim (row) is selected based on boolean mask as indices (e.g. x[mask]),
        # and indices aren't dependant on input activation
        elif len(indices) == 1 and self.infer_type(indices[0]).dtype == "bool" and len(_analysis.free_vars(indices[0])) == 0:
            indices = indices[0]

            # Reduce data to 1D vector if needed
            data_shape = _infer_shape(data)
            data = _op.squeeze(data, axis=0) if len(data_shape) == 2 and data_shape[0] == 1 else data

            assert len(_infer_shape(data)) == 1 and len(_infer_shape(indices)) == 1, "Currently supportes only 1D tensors"

            if isinstance(indices, _expr.Constant):
                # Extract direct data as numpy array
                indices = indices.data.numpy()
            else:
                # Infer values to get constant data (discards irelevant math which
                # will eventually be optimized out)
                indices = _infer_value(indices, {}).numpy()

            indices = np.transpose(np.argwhere(indices))
            indices = indices[0] if len(indices.shape) == 2 and indices.shape[0] == 1 else indices
            indices = _expr.const(indices)
            res = _op.adv_index([data, indices])

            return res

        # Cover cases where boolean mask is used as indices (e.g. x[mask]), while indices have
        # dependecies on input activations (or parameters)
        elif len(indices) == 1 and self.infer_type(indices[0]).dtype == "bool" and len(_analysis.free_vars(indices[0])) > 0:
            indices = indices[0]

            # Reduce data to 1D vector if possible
            data_shape = _infer_shape(data)
            data = _op.squeeze(data, axis=_op.const([0])) if len(data_shape) == 2 and data_shape[0] == 1 else data

            # Reduce indices to 1D vector if possible
            indices_shape = _infer_shape(indices)
            indices = _op.squeeze(indices, axis=_op.const([0])) if len(indices_shape) == 2 and indices_shape[0] == 1 else indices

            assert len(_infer_shape(data)) == 1 and len(_infer_shape(indices)) == 1, "Currently supportes only 1D tensors"

            # Make sure to have valid indices shape
            # indices = _op.reshape(indices, newshape=[int(i) for i in _infer_shape(indices)])
            indices = _op.reshape(indices, newshape=(-1,))

            # Extract indices from boolean mask
            indices = _op.transform.argwhere(indices)
            # Doing this reshape to remove dynamic shapes caused by argwhere op (e.g. '?' shapes). This
            # reshape will ensure that the output of argwhere (and following ops) is "predictable" in a
            # manner to suport further TVM compilation. However, this is only valid if this op is fallback
            # on CPU. Otherwise, this reshape will cause incorrect results.
            # indices = _op.reshape(indices, newshape=_infer_shape(data)[0])
            indices = _op.transpose(indices, (1, 0))
            indices = _op.squeeze(indices, _expr.const([0])) if len(_infer_shape(indices)) == 2 and _infer_shape(indices)[0] == 1 else indices
            res = _op.adv_index([data, indices])

            return res
        
        elif len(_infer_shape(data)) > 2 and sum(idx is not None for idx in indices) == 1:
            axis = None
            index_expr = None
            for i, idx in enumerate(indices):
                if idx is not None:  
                    axis = i
                    index_expr = idx
                    break
                
            res = _op.take(data, index_expr, axis=axis)
            return res

        return _op.adv_index([data] + indices)

    def meshgrid(self, inputs, input_types):
        data = inputs[0]
        indexing = inputs[1] if len(inputs) > 1 else "ij"
        return _op.meshgrid(data,indexing)

    def nms(self, inputs, input_types):
        boxes = inputs[0]
        scores = inputs[1]
        iou_threshold = inputs[2]

        # TVM NMS assumes score > 0
        # - since there exists multi-comsumers for "scores", "num_boxes"
        # - invoke set_span here to prevent expr-rewritten occurrs in span-filling stage
        source_name = self.source_map[self.current_op[-1]]
        scores = set_span(scores - _op.min(scores) + _op.const(1.0), source_name)

        num_boxes = set_span(_op.shape_of(scores), source_name)
        # PyTorch NMS doesn't have score_threshold, so no need to run get_valid_count
        # - since "arange" op will fill expr into its attribute
        # - invoke set_span here to prevent expr-rewritten occurrs in span-filling stage
        indices = _op.transform.arange(set_span(_op.squeeze(num_boxes), source_name), dtype="int32")
        indices = _op.expand_dims(indices, 0, 1)

        # Generate data with shape (1, num_anchors, 5)
        scores = AttrCvt(op_name="expand_dims", extras={"axis": -1, "num_newaxis": 1})([scores], {})
        data = _op.concatenate([scores, boxes], -1)
        data = _op.expand_dims(data, 0, 1)

        # Perform Non-Maximum Suppression,
        # PyTorch NMS doesn't have parameter top_k and max_output_size
        score_index = 0
        top_k = max_out_size = -1
        nms_ret = get_relay_op("non_max_suppression")(
            data=data,
            valid_count=num_boxes,
            indices=indices,
            max_output_size=max_out_size,
            iou_threshold=iou_threshold,
            force_suppress=True,
            top_k=top_k,
            coord_start=1,
            score_index=score_index,
            id_index=-1,
            return_indices=True,
            invalid_to_bottom=False,
        )

        # squeeze the two outputs of nms for strided_slice
        size = get_relay_op("squeeze")(nms_ret[1], axis=[1])
        data_slice = get_relay_op("squeeze")(nms_ret[0], axis=[0])

        # strided slice to get the dynamic result
        ret = get_relay_op("strided_slice")(
            data_slice, begin=_expr.const([0]), end=size, slice_mode="size"
        )
        # in torchvision, indices from nms are int64
        return _op.cast(ret, "int64")

    def logsumexp(self, inputs, input_types):
        data = self.pytorch_promote_types(inputs[:1], input_types[:1])
        dim_list = inputs[1]
        keepdim = inputs[2] if len(inputs) > 2 else False
        # dim is output of prim::ListConstruct, even if it is int in python code
        assert isinstance(dim_list, list), "dim is expected to be a list"
        return _op.logsumexp(data[0], axis=dim_list, keepdims=keepdim)

    def roi_align(self, inputs, input_types):
        data = inputs[0]
        boxes = inputs[1]

        output_size = (inputs[3], inputs[4])
        spatial_scale = inputs[2]
        sample_ratio = inputs[5]
        aligned = False if len(inputs) < 7 else inputs[6]

        if aligned:
            boxes -= _expr.const(0.5 / spatial_scale)

        return _op.vision.roi_align(data, boxes, output_size, spatial_scale, sample_ratio)

    def deform_conv2d(self, inputs, input_types):
        data = inputs[0]
        weight = inputs[1]
        offset = inputs[2]

        if len(inputs) > 12:
            strides_offset = 5
            bias = inputs[4]
            logger.warning("mask argument in deformable conv2d is not supported and ignored")
        else:
            strides_offset = 4
            bias = inputs[3]

        strides = (inputs[strides_offset], inputs[strides_offset + 1])
        padding = (inputs[strides_offset + 2], inputs[strides_offset + 3])
        dilation = (inputs[strides_offset + 4], inputs[strides_offset + 5])
        groups = inputs[strides_offset + 6]
        deformable_groups = inputs[strides_offset + 7]
        weight_shape = self.infer_shape(weight)
        output_channels = weight_shape[0]
        kernel_size = (weight_shape[2], weight_shape[3])

        conv_out = _op.nn.deformable_conv2d(
            data,
            offset,
            weight,
            strides,
            padding,
            dilation,
            deformable_groups,
            groups,
            output_channels,
            kernel_size,
        )

        return _op.nn.bias_add(conv_out, bias)

    def stft(self, inputs, input_types):
        data = inputs[0]
        n_fft = inputs[1]
        hop_length = inputs[2]
        win_length = inputs[3]
        window = inputs[4]
        normalized = inputs[5]
        onesided = inputs[6]

        return _op.stft(data, n_fft, hop_length, win_length, window, normalized, onesided)

    def unbind(self, inputs, input_types):
        data = inputs[0]
        axis = int(inputs[1])
        return unbind(data, axis)

    def shape_as_tensor(self, inputs, input_types):
        is_symbolic_shape = False
        input_shape = self.infer_shape(inputs[0], self.prelude.mod)
        for axis in input_shape:
            if not isinstance(axis, (int, tvm.tir.IntImm)):
                is_symbolic_shape = True
                break

        if is_symbolic_shape:
            ret = _op.shape_of(inputs[0], dtype="int64")
        else:
            ret = _expr.const(np.array(input_shape), dtype="int64")

        return ret

    def logical_and(self, inputs, input_types):
        lhs = _op.cast(inputs[0], "bool")
        rhs = _op.cast(inputs[1], "bool")

        return _op.logical_and(lhs, rhs)

    def nonzero(self, inputs, input_types, is_numpy_style=False):
        data = inputs[0]
        ret = _op.transform.argwhere(data)
        if is_numpy_style or (len(inputs) > 1 and inputs[1]):
            return unbind(ret, 1)
        return ret

    def nonzero_numpy(self, inputs, input_types):
        return self.nonzero(inputs, input_types, is_numpy_style=True)

    def scatter(self, inputs, input_types):
        assert len(inputs) == 4 or len(inputs) == 5, (
            f"scatter takes 4 or 5 inputs: data, dim, index, src, reduce (optional), "
            f"but {len(inputs)} given"
        )
        data = inputs[0]
        axis = int(inputs[1])
        index = inputs[2]
        src = inputs[3]
        if len(inputs) == 5:
            reduce = inputs[4]
        else:
            reduce = "update"

        data_shape = self.infer_shape(data)
        data_rank = len(data_shape)
        index_shape = self.infer_shape(index)
        index_rank = len(index_shape)
        # When index is empty, the operation returns data unchanged
        if self.is_empty_shape(index_shape):
            return data

        if np.isscalar(src):
            assert self.infer_type(src).dtype == "float", "Scalar source can be float only"
            src = _op.broadcast_to_like(src, data_shape)
            src_shape = data_shape
        else:
            src_shape = self.infer_shape(src)
        src_rank = len(src_shape)
        assert data_rank == index_rank, "Index rank is not the same as data rank"
        assert data_rank == src_rank, "Src rank is not the same as data rank"

        assert 0 <= axis < data_rank, "Dim is out of bounds"

        for i in range(data_rank):
            index_dim = index_shape[i]
            src_dim = src_shape[i]
            data_dim = data_shape[i]
            # Skip check for dynamic dimensions
            if not any([isinstance(index_dim, tvm.tir.Any), isinstance(src_dim, tvm.tir.Any)]):
                assert index_dim <= src_dim, "Index dim size should be less than src one"
            if i != axis and not any(
                [isinstance(index_dim, tvm.tir.Any), isinstance(data_dim, tvm.tir.Any)]
            ):
                assert index_dim <= data_dim, "Index dim size should be less than data one"

        if reduce is None:
            reduce = "update"
        elif reduce == "multiply":
            reduce = "mul"
        assert reduce in [
            "update",
            "add",
            "mul",
        ], 'reduce arg is expected from "add", "multiply" or None'

        return _op.scatter_elements(data, index, src, axis, reduce)

    def index_put(self, inputs, input_types):
        in_tensor = inputs[0]
        indices = inputs[1]
        values = inputs[2]
        accumulate = inputs[3]
        if not accumulate:
            mode = "update"
        else:
            mode = "add"
        # Combine array of index tensors into one index tensor with shape (N,_)
        index_tensor = _op.stack(indices, axis=0)

        # Narrow index tensor to match input tensor
        shape_diff = len(_infer_shape(index_tensor)) - len(_infer_shape(in_tensor))
        if shape_diff > 0:
            for shape in range(shape_diff):
                index_tensor = _op.squeeze(index_tensor, axis=[0])

        # If indexes are in form of boolean mask instead of indices, use where op
        # instead of scatter_nd
        if _infer_type(index_tensor).checked_type.dtype == "bool":
            if isinstance(values, float):
                values = _expr.const(values, dtype=_infer_type(in_tensor).checked_type.dtype)

            # Make sure that dynamic output will be 1D vector
            # index_tensor = _op.reshape(index_tensor, newshape=(-1,))
            # Make sure that dynamic output will be 1D vector
            # index_tensor_shape = _infer_shape(index_tensor)
            # index_tensor = _op.squeeze(index_tensor, axis=_op.const([0])) if len(index_tensor_shape) == 2 and index_tensor_shape[0] == 1 else index_tensor
            # Make sure that dynamic output will be 1D vector
            # index_tensor_shape = _infer_shape(index_tensor)
            # index_tensor = _op.take(index_tensor, _op.const(0), axis=0) if len(index_tensor_shape) == 2 and index_tensor_shape[0] == 1 else index_tensor

            indices = _op.transform.argwhere(index_tensor)
            indices = _op.transpose(indices, (1, 0))
            indices = _op.squeeze(indices, _expr.const([0])) if len(_infer_shape(indices)) == 2 and _infer_shape(indices)[0] == 1 else indices

            # Make sure that dynamic output will be 1D vector
            values = _op.reshape(values, newshape=(-1,))

            # Reduce data to 1D vector if possible
            in_tensor = _op.reshape(in_tensor, newshape=(-1,))

            res = _op.scatter_elements(in_tensor, indices, values, 0, "add")

            return res

        return _op.transform.scatter_nd(in_tensor, index_tensor, values, mode)

    def index_copy(self, inputs, input_types):
        in_tensor = inputs[0]
        axis = inputs[1]
        indices = inputs[2]
        values = inputs[3]

        mode = "update"
        return _op.transform.scatter_elements(in_tensor, indices, values, axis)

    def scalar_tensor(self, inputs, input_types):
        data = inputs[0]
        cast_map = {6: "float32", 7: "float64", 3: "int32", 4: "int64"}
        type_key = inputs[1]
        if isinstance(data, _expr.Constant):
            data = data.data.numpy().tolist()
        return _expr.const(data, cast_map[type_key])

    def interpolate(self, inputs, input_types):
        if isinstance(inputs[1], _expr.Expr):
            out_size = inputs[1]
        elif isinstance(inputs[1], list):
            out_size = []
            for i in [0, 1]:
                size, _ = try_infer_value(
                    inputs[1][i],
                    lambda ret: ret.astype(int),
                    lambda: _op.expand_dims(inputs[1][i], axis=0),
                )
                out_size.append(size)
            out_size = _op.concatenate(out_size, axis=0)

        data = inputs[0]
        align_corners = inputs[4]
        method = inputs[3]
        if method.startswith("nearest"):
            method = "nearest_neighbor"
        elif method[0:2] == "bi":
            method = method[2:]

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        return _op.image.resize2d(
            data, out_size, None, "NCHW", method, coord_trans, cubic_alpha=-0.75
        )

    def numel(self, inputs, input_types):
        return _op.ndarray_size(inputs[0])

    def empty(self, inputs, input_types):
        if isinstance(inputs[0], list):
            shape = []
            for item in inputs[0]:
                if isinstance(item, _expr.Constant):
                    shape.append(int(item.data.numpy()))
                elif isinstance(item, _expr.Call):
                    array = _infer_value(inputs[0][1], {})
                    shape.append(array.numpy().item())
                else:
                    shape.append(item)
        else:
            shape = [int(dim.data.numpy()) if isinstance(dim, _expr.Constant) else dim for dim in inputs[0]]

        data = np.zeros(shape)
        return _create_typed_const(data, _convert_dtype_value(inputs[1]))

    def empty_like(self, inputs, input_types):
        shape = self.infer_shape(inputs[0])
        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = input_types[0]
        return _op.zeros(shape, dtype)

    def new_empty(self, inputs, input_types):
        size = inputs[1]

        import torch

        if not isinstance(size, (_expr.Expr, list, tuple, torch.Size, np.ndarray)):
            msg = f"Data type {type(size)} could not be parsed in empty op"
            raise AssertionError(msg)

        if inputs[2] is not None:
            dtype = _convert_dtype_value(inputs[2])
        else:
            dtype = input_types[0]
        return _op.zeros(size, dtype)

    def randn(self, inputs, input_types):
        import time  # use current time as seed

        shape = inputs[0]
        output = _op.random.normal(_op.random.threefry_key(int(time.time())), shape)
        _, values = _expr.TupleWrapper(output, 2)
        return values

    def bincount(self, inputs, input_types):
        data = inputs[0]
        weights = inputs[1]
        input_type = self.infer_type(data).dtype
        if input_type == "int64":
            logger.warning(
                "Casting an int64 input to int32, since we do not have int64 atomic add"
                "needed for bincount yet."
            )
            data = _op.cast(data, "int32")
        maximum = _op.max(data)
        dim = maximum + _expr.const(1, dtype="int32")
        if weights:
            weight_type = self.infer_type(weights)
            out_dtype = weight_type.dtype
            updates = weights
        else:
            out_dtype = "int32"
            updates = _op.ones_like(data)

        counts = _op.zeros(_op.reshape(dim, [1]), out_dtype)
        out = _op.scatter_elements(counts, data, updates, axis=0, reduction="add")
        if input_type == "int32":
            # Torch always outputs int64 results for bincount
            return _op.cast(out, "int64")
        return out

    def scatter_add(self, inputs, input_types):
        assert (
            len(inputs) == 4
        ), f"scatter_add takes 4 inputs (data, dim, index, src), but {len(inputs)} given"
        data = inputs[0]
        axis = inputs[1]
        index = _op.cast(inputs[2], "int64")
        src = inputs[3]

        data_shape = self.infer_shape(inputs[0])
        data_rank = len(data_shape)
        index_shape = self.infer_shape(inputs[2])
        index_rank = len(index_shape)
        # When index is empty, the operation returns data unchanged
        if self.is_empty_shape(index_shape):
            return data
        src_shape = self.infer_shape(inputs[3])
        src_rank = len(src_shape)
        assert data_rank == index_rank, "Index rank is not the same as data rank"
        assert data_rank == src_rank, "Src rank is not the same as data rank"

        assert 0 <= axis < data_rank, "Dim is out of bounds"

        for i in range(data_rank):
            assert index_shape[i] <= src_shape[i], "Index dim size should be less than src one"
            if i != axis:
                assert (
                    index_shape[i] <= data_shape[i]
                ), "Index dim size should be less than data one"

        return _op.scatter_elements(data, index, src, axis=axis, reduction="add")

    def scatter_reduce(self, inputs, input_types):
        assert len(inputs) == 5 or len(inputs) == 6, (
            f"scatter_reduce takes 5 or 6 inputs (data, dim, index, src, reduce, include_self), "
            f"but {len(inputs)} given"
        )
        data = inputs[0]
        dim = inputs[1]
        index = inputs[2]
        src = inputs[3]
        reduce = inputs[4]
        if len(inputs) == 6:
            include_self = inputs[5]
            # TODO(vvchernov): support include_self == False
            assert include_self, "include_self=False has not been suppoted for scatter_reduce yet"

        data_shape = self.infer_shape(inputs[0])
        data_rank = len(data_shape)
        index_shape = self.infer_shape(inputs[2])
        index_rank = len(index_shape)
        src_shape = self.infer_shape(inputs[3])
        src_rank = len(src_shape)
        assert data_rank == index_rank, "Index rank is not the same as data rank"
        assert data_rank == src_rank, "Src rank is not the same as data rank"

        assert 0 <= dim < data_rank, "Dim is out of bounds"

        for i in range(data_rank):
            assert index_shape[i] <= src_shape[i], "Index dim size should be less than src one"
            if i != dim:
                assert (
                    index_shape[i] <= data_shape[i]
                ), "Index dim size should be less than data one"

        red_valids = ["sum", "prod", "mean", "amax", "amin"]
        assert (
            reduce in red_valids
        ), f"Only {red_valids} modes are supported, but {reduce} is gotten"
        if reduce == "sum":
            reduce = "add"
        elif reduce == "prod":
            reduce = "mul"
        elif reduce == "amin":
            reduce = "min"
        elif reduce == "amax":
            reduce = "max"

        return _op.scatter_elements(data, index, src, axis=dim, reduction=reduce)


    def cumsum(self, inputs, input_types):
        data = inputs[0]
        dim = inputs[1]
        dtype = inputs[2]

        if inputs[2] is not None:
            dtype = _convert_dtype_value(inputs[2])

        return _op.cumsum(data, axis=dim, dtype=dtype, exclusive=False)

    def masked_fill(self, inputs, input_types):
        mask = inputs[1]

        def replace_inf(inp, replacement_val=1e4):
            if isinstance(inp, _expr.Call) and list(_analysis.free_vars(inp)) == []:
                inp = _infer_value(inp, {}).asnumpy()
                inp[np.isneginf(inp)] = -replacement_val
                inp[np.isposinf(inp)] = replacement_val
                inp = _expr.const(tvm.nd.array(inp))
            elif isinstance(inp, float) and np.isinf(inp):
                inp = np.sign(inp)*replacement_val if np.isinf(inp) else inp
            return inp

        # Handle constant inputs where we can determine -inf/inf values from it
        inputs[0] = replace_inf(inputs[0])
        inputs[2] = replace_inf(inputs[2])

        if isinstance(mask, _expr.Call) and list(_analysis.free_vars(mask)) == []:
            mask = _expr.const(_infer_value(mask, {}))
        mask = _op.cast(mask, "float32")

        disable_masked_fill_v2 = bool(int(os.environ.get("PYBUDA_DISABLE_MASKED_FILL_V2", 0)))
        if disable_masked_fill_v2:
            value = _op.cast(_wrap_const(inputs[2]), input_types[0])
            value = _op.broadcast_to_like(value, mask)

            one_const = _expr.const(1, dtype="float32")
            return _op.add(_op.multiply(inputs[0], _op.subtract(one_const, mask)), _op.multiply(value, mask))

        # Pybuda will convert all cast ops to Identity. This can be an issue.
        # In pytorch, when the mask has float values, it treats 0 as False and 
        # everything else as True. We cannot assume that mask will contain only
        # zeroes and ones, as multiplication with the mask will yeild incorrect
        # results.
        mask = _op.abs(mask)
        mask = _op.clip(mask, 0, 1.0)

        # In the event some values actually lie between 0 and 1 we use greater
        # to ceil those all to 1.
        # NOTE: The cast here is just to stop TVM from complaining. It will be
        # replaced with the identity function further down the line
        mask = _op.cast(_op.greater(mask, _expr.const(0, "float32")), "float32")

        value = _op.cast(_wrap_const(inputs[2]), input_types[0])
        value = _op.broadcast_to_like(value, mask)

        one_const = _expr.const(1, dtype="float32")
        inverse_mask = _op.subtract(one_const, mask)

        # Original implementation
        # return _op.where(mask, value, inputs[0])

        # Implementaiton without using where operator in order to avoide numerical instability
        # for certain models caused by the future matmul (once where is decomposed)
        
        # (!mask * x) + (mask*y)
        return _op.add(_op.multiply(inputs[0], inverse_mask), _op.multiply(value, mask))


    def baddbmm(self, inputs, input_types):
        bmm = _op.nn.batch_matmul(inputs[1], inputs[2], out_dtype=input_types[0], transpose_a=False, transpose_b=False)
        bmm_mul = _op.multiply(tvm.relay.const(tvm.nd.array([np.float32(inputs[4]),]), dtype=input_types[0]), bmm)
        beta_input = _op.multiply(tvm.relay.const(tvm.nd.array([np.float32(inputs[3]),]), dtype=input_types[0]), inputs[0])
        final_add = _op.add(bmm_mul, beta_input)
        return final_add

    def __or__(self, inputs, input_types):
        return _op.bitwise_or(inputs[0], inputs[1])

    def masked_select(self, inputs, input_types):
        mask = inputs[1]
        indices = self.nonzero([mask], input_types, is_numpy_style=True)
        return _op.adv_index([inputs[0]] + [indices[i] for i in range(indices.size)])

    def masked_scatter(self, inputs, input_types):
        """
        Performs a masked scatter operation on the input tensor `data`, replacing values where 
        the `mask` is True with corresponding values from the `source` tensor.

        This implementation follows the decomposition of the PyTorch `aten::masked_scatter` operation:
        "aten::masked_scatter": https://github.com/pytorch/pytorch/blob/f95bdf5e6c8ea482ba6f64d655513b6a191ac142/torch/_inductor/decomposition.py#L830

        The masked scatter requires `aten::unsafe_masked_index`, which is implemented in our TVM 
        decomposition based on the following PyTorch function:
        "aten::unsafe_masked_index": https://github.com/pytorch/pytorch/blob/f95bdf5e6c8ea482ba6f64d655513b6a191ac142/aten/src/ATen/native/TensorAdvancedIndexing.cpp#L773

        The unsafe masked index is adapted from PyTorch's TensorAdvancedIndexing.

        Args:
            inputs (list): A list containing the input tensors:
                - `data`: The tensor to be updated.
                - `mask`: A boolean tensor, where True values indicate positions to scatter the `source` tensor.
                - `source`: The tensor with values to scatter into `data` where `mask` is True.
            input_types (list): A list of the types of input tensors (not used in this implementation).

        Returns:
            Tensor: The updated tensor, with values from `source` scattered into `data` at positions where `mask` is True.
        """
        data = inputs[0]  
        mask = inputs[1]  
        source = inputs[2]

        # Count the number of True values in the mask
        mask_true_count = _op.sum(_op.cast(mask, "int32"))
        source_size = _op.prod(_op.shape_of(source))
        assert _op.less_equal(mask_true_count, source_size), "Source tensor must have at least as many elements as ones in mask (source size: %s, mask true count: %s)" % (source_size, mask_true_count)
        
        mask = _op.cast(mask, dtype="float32")

        def broadcast_tensors(mask,data):
            mask_shape = _infer_shape(mask)
            data_shape = _infer_shape(data)
            shape1 = [1] * (len(data_shape) - len(mask_shape)) + list(mask_shape)
            shape2 = [1] * (len(mask_shape) - len(data_shape)) + list(data_shape)
            common_shape = []
            for dim1, dim2 in zip(shape1, shape2):
                if dim1 == dim2:
                    common_shape.append(dim1)
                elif dim1 == 1:
                    common_shape.append(dim2)
                elif dim2 == 1:
                    common_shape.append(dim1)
                else:
                    raise ValueError(f"Cannot broadcast shapes: {shape1} and {shape2}")
            common_shape = tuple(common_shape)
            mask = _op.broadcast_to(mask, common_shape)
            data = _op.broadcast_to(data, common_shape)
            return mask,data

        mask,data = broadcast_tensors(mask,data)


        data_shape = _infer_shape(data)

        # Flatten the mask and compute the cumulative sum for indexing
        flattened_mask = _op.reshape(mask, newshape=(-1))
        cumsum_result = _op.cumsum(flattened_mask, axis=0, exclusive=False)

        # Calculate the indices for the source tensor, shifted by 1
        source_idx = _op.subtract(cumsum_result, _expr.const(1, dtype="float32"))
        source_idx = _op.cast(source_idx, dtype="int32")

        # Flatten data and source for easier manipulation
        data_flat = _op.reshape(data, newshape=(-1))
        source_flat = _op.reshape(source, newshape=(-1))
        source_flat_shape = _infer_shape(source_flat)

        # Clamp the indices to ensure they are within bounds
        clamped_indices = _op.minimum(
            _op.maximum(source_idx, _expr.const(0, dtype="int32")),
            _op.subtract(_op.const([source_flat_shape], dtype="int32"), _expr.const(1, dtype="int32"))
        )

        # Perform the scatter operation
        result = _op.transform.take(source_flat, _op.reshape(clamped_indices,(-1)), axis=0, mode='wrap')
        fills = _op.full_like(result, _expr.const(0, dtype='int64'))
        result = _op.where(flattened_mask, _op.reshape(result,(-1)), _op.reshape(fills,(-1)))
        result = _op.where(flattened_mask, result, data_flat)

        # Reshape the result back to the original data shape
        result = _op.reshape(result, newshape=data_shape)
        return result


    def sort(self, inputs, input_types):
        data = inputs[0]
        dim = inputs[1]
        is_descending = inputs[2]
        # pytorch sort returns both sorted indices and values
        indices = _op.argsort(data, dim, not is_descending)
        return _op.gather(data, dim, indices), indices

    def argsort(self, inputs, input_types):
        data = inputs[0]
        dim = inputs[1]
        is_descending = inputs[2]
        return _op.argsort(data, dim, not is_descending)

    def is_floating_point(self, inputs, input_types):
        assert len(inputs) == 1

        if isinstance(inputs[0], _expr.Expr):
            input_type = self.infer_type(inputs[0]).dtype
        else:
            input_type = input_types[0]

        is_float = input_type in ["float32", "float64", "float16", "bfloat16"]
        return _expr.const(is_float)

    def unique(self, inputs, input_types):
        assert len(inputs) == 4
        [data, is_sorted, return_inverse, return_counts] = inputs
        if not is_sorted:
            logger.warning("TVM always assumes sorted=True for torch.unique")
            is_sorted = True
        if return_counts:
            [unique, indices, inverse_indices, num_uniq, counts] = _op.unique(
                data, is_sorted=is_sorted, return_counts=True
            )
            unique_sliced = _op.strided_slice(unique, begin=[0], end=num_uniq, slice_mode="size")
            counts_sliced = _op.strided_slice(counts, begin=[0], end=num_uniq, slice_mode="size")
            return (unique_sliced, inverse_indices, counts_sliced)
        else:
            [unique, indices, inverse_indices, num_uniq] = _op.unique(
                data, is_sorted=is_sorted, return_counts=False
            )
            unique_sliced = _op.strided_slice(unique, begin=[0], end=num_uniq, slice_mode="size")
            return (unique_sliced, inverse_indices)

    def l1_loss(self, inputs, input_types):
        assert len(inputs) == 3
        [input, targets, reduction] = inputs
        if reduction == 0:
            reduction = "none"
        elif reduction == 1:
            reduction = "mean"
        else:
            reduction = "sum"

        from loguru import logger
        logger.warning("Abs not supported yet, running L1Loss without it. It won't match 'real' values.")
        diff = _op.subtract(input, targets)
        # diff = _op.abs(_op.subtract(input, targets))
        if reduction == "none":
            return diff
        elif reduction == "mean":
            return _op.mean(diff, keepdims=True)
        else:
            return _op.sum(diff, keepdims=True)

    def mse_loss(self, inputs, input_types):
        assert len(inputs) == 3
        [input, targets, reduction] = inputs
        if reduction == 0:
            reduction = "none"
        elif reduction == 1:
            reduction = "mean"
        else:
            reduction = "sum"

        diff = _op.power(_op.subtract(input, targets), _expr.const(2, input_types[0]))
        if reduction == "none":
            return diff
        elif reduction == "mean":
            return _op.mean(diff, keepdims=True)
        else:
            return _op.sum(diff, keepdims=True)

    def nll_loss(self, inputs, input_types):
        assert len(inputs) == 5
        [predictions, targets, weights, reduction, ignore_index] = inputs
        num_class = self.infer_shape(predictions)[1]
        if reduction == 0:
            reduction = "none"
        elif reduction == 1:
            reduction = "mean"
        else:
            reduction = "sum"
        if weights is None:
            weights = _op.full(_expr.const(1), (num_class,), dtype=input_types[0])
        return _op.nn.nll_loss(predictions, targets, weights, reduction, ignore_index)

    def flip(self, inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        out = data
        for ax in axis:
            out = _op.reverse(out, ax)
        return out

    def bidir_rnn_cell(self, input_seqs, weights_dicts, act=_op.tanh):
        """
        Bidirectional RNN cell
        """
        seq_len = len(input_seqs)
        forward_outputs, fw_H_t = rnn_cell(input_seqs, **weights_dicts[0], backwards=False, act=act)

        reverse_outputs, rev_H_t = rnn_cell(input_seqs, **weights_dicts[1], backwards=True, act=act)

        final_outputs = []
        for i in range(seq_len):
            final_outputs.append(
                _op.concatenate([forward_outputs[i], reverse_outputs[seq_len - 1 - i]], axis=-1)
            )

        return final_outputs, _op.stack([fw_H_t, rev_H_t], axis=0)

    def rnn_layers(self, input_data, layer_weights_dicts, bidirectional, act, dropout_p=0.0):
        """
        Methods iterates layers for Stacked RNN
        """
        layers_num = len(layer_weights_dicts)
        # split input sequence to samples set
        input_seqs = unbind(input_data, 0)  # [seq_num, (batch, feature_size)]
        output_hiddens = []
        for i in range(layers_num):
            weights_dicts = layer_weights_dicts[i]
            # input_seqs shape = [seq_num, (batch, feature_size)] or
            # [seq_num, (batch, 2*feature_size)] for bidirectional
            if bidirectional:
                input_seqs, H_t = self.bidir_rnn_cell(input_seqs, weights_dicts, act=act)
            else:
                input_seqs, H_t = rnn_cell(input_seqs, **weights_dicts[0], act=act)

            output_hiddens.append(H_t)

            # TODO (yuanfz98): in pytorch implementation train is also checked
            # see https://github.com/pytorch/pytorch/blob/70c8daf43946b53af6493d058899ef952d27d339
            # /aten/src/ATen/native/RNN.cpp#L1054
            if dropout_p != 0 and i < layers_num - 1:
                # for input in input_seqs:
                #     input = _op.dropout(input, dropout_p)
                raise NotImplementedError("Dropout for GRU has not been supported yet!")
        output_hiddens = (
            _op.concatenate(output_hiddens, 0) if bidirectional else _op.stack(output_hiddens, 0)
        )
        return _op.stack(input_seqs, 0), output_hiddens

    def rnn(self, inputs, input_types, nonlinearity):
        """
        Description of RNN in pytorch:
        https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
        Description of inputs:
        https://github.com/pytorch/pytorch/blob/736fb7d22cc948b739db2c35aeb5ad4d19aea4f4/torch/overrides.py#L937
        """
        # TODO (yuanfz98): support dropout
        assert len(inputs) == 9, "Input of size 9 is expected"
        # Unpack inputs, note that if optional and not provided then value will be None.
        _X = inputs[0]
        # _X shape (seq_num, batch, feature_size) or (batch, seq_num, feature_size)

        hidden_state = inputs[1]
        # Hidden state shape (hidden_layers_num, batch, hidden_size)

        _weights = inputs[2]
        # Wi layer[0] shape (hidden_size, feature_size)
        # Wh layer[0] shape (hidden_size, hidden_size)
        # Bi layer[0] shape (hidden_size)
        # Bh layer[0] shape (hidden_size)

        # Wi layer[>0] shape (hidden_size, hidden_size * num_directions)
        # Wh layer[>0] shape (hidden_size, hidden_size)
        # Bi layer[>0] shape (hidden_size)
        # Bh layer[>0] shape (hidden_size)

        # Scalar inputs
        has_biases = inputs[3]
        num_layers = inputs[4]
        dropout_p = inputs[5]  # dropout probability, if 0.0 it means there is no dropout
        # train = inputs[6]
        bidirectional = inputs[7]
        batch_first = inputs[8]

        num_directions = 1
        if bidirectional:
            num_directions = 2

        rsd = len(_weights) % num_layers
        assert rsd == 0, "The number of weights must be a multiple of the number of layers!"
        rsd = (len(_weights) / num_layers) % num_directions
        assert (
            rsd == 0
        ), "The number of weights in layer must be a multiple of the number of directions!"

        weights_num = int(len(_weights) / num_layers / num_directions)
        if has_biases:
            assert weights_num == 4, "The weights number in layer is expected equal to 4"
        else:
            assert weights_num == 2, "The weights number in layer is expected equal to 2"
        if nonlinearity == "tanh":
            act = _op.tanh
        elif nonlinearity == "relu":
            act = _op.nn.relu
        assert act, "The nonlinearity is unknown"
        X = (
            _op.transpose(_X, (1, 0, 2)) if batch_first else _X
        )  # always (seq_num, batch, feature_size)
        # TODO (yuanfz98): Which data type should be used? from input or weights?
        # Instead of it _infer_type(X).checked_type.dtype can be used
        X_dtype = input_types[0]
        X_shape = _infer_shape(X)  # (seq_num, batch, feature_size)

        hidden_size = int(_infer_shape(_weights[0])[0])
        batch_size = X_shape[1]

        # Initialize hidden states if not provided.
        layers_h = []
        hidden_layers_num = num_directions * num_layers
        if hidden_state is None:
            h_0 = _op.zeros((batch_size, hidden_size), X_dtype)
            for i in range(hidden_layers_num):
                layers_h.append(h_0)
        else:
            layers_h = unbind(hidden_state, 0)

        layer_weights_dicts = []
        k = 0  # layer counter
        if has_biases:
            names = ["hidden_state", "w_inp", "w_hid", "b_inp", "b_hid"]
            if bidirectional:
                rsd = len(_weights) % (2 * weights_num)
                assert rsd == 0, "got an incorrect number of RNN weights"
                for i in range(0, len(_weights), 2 * weights_num):
                    fw_tensors = [layers_h[2 * k], *_weights[i : i + 4]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    j = i + weights_num
                    rev_tensors = [layers_h[2 * k + 1], *_weights[j : j + 4]]
                    rev_weights_dict = dict(zip(names, rev_tensors))
                    layer_weights_dicts.append([fw_weights_dict, rev_weights_dict])
                    k += 1
            else:
                assert len(_weights) % weights_num == 0, "got an incorrect number of GRU weights"
                for i in range(0, len(_weights), weights_num):
                    fw_tensors = [layers_h[k], *_weights[i : i + 4]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    layer_weights_dicts.append([fw_weights_dict])
                    k += 1
        else:
            names = ["hidden_state", "w_inp", "w_hid"]
            if bidirectional:
                rsd = len(_weights) % (2 * weights_num)
                assert rsd == 0, "got an incorrect number of RNN weights"
                for i in range(0, len(_weights), 2 * weights_num):
                    fw_tensors = [layers_h[2 * k], *_weights[i : i + 2]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    j = i + weights_num
                    rev_tensors = [layers_h[2 * k + 1], *_weights[j : j + 2]]
                    rev_weights_dict = dict(zip(names, rev_tensors))
                    layer_weights_dicts.append([fw_weights_dict, rev_weights_dict])
                    k += 1
            else:
                assert len(_weights) % weights_num == 0, "got an incorrect number of RNN weights"
                for i in range(0, len(_weights), weights_num):
                    fw_tensors = [layers_h[k], *_weights[i : i + 2]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    layer_weights_dicts.append([fw_weights_dict])
                    k += 1
        assert (
            len(layer_weights_dicts) == num_layers and k == num_layers
        ), "For stacked RNN number of weights sets should be the same as number of layers!"
        output, out_hidden_state = self.rnn_layers(
            X, layer_weights_dicts, bidirectional, act, dropout_p=dropout_p
        )

        # output shape = (seq_num, batch, hidden_size) or
        # (seq_num, batch, 2*feature_size) for bidirectional
        if batch_first:
            output = _op.transpose(output, (1, 0, 2))

        return (output, out_hidden_state)

    def bidir_gru_cell(self, input_seqs, weights_dicts):
        """
        Bidirectional GRU cell
        """
        seq_len = len(input_seqs)
        forward_outputs, fw_H_t = gru_cell(input_seqs, **weights_dicts[0])

        reverse_outputs, rev_H_t = gru_cell(input_seqs, **weights_dicts[1], backwards=True)

        final_outputs = []
        for i in range(seq_len):
            final_outputs.append(
                _op.concatenate([forward_outputs[i], reverse_outputs[seq_len - 1 - i]], axis=-1)
            )

        return final_outputs, _op.stack([fw_H_t, rev_H_t], axis=0)

    def gru_layers(self, input_data, layer_weights_dicts, bidirectional, dropout_p=0.0):
        """
        Methods iterates layers for Stacked GRU
        """
        layers_num = len(layer_weights_dicts)
        # split input sequence to samples set
        input_seqs = unbind(input_data, 0)  # [seq_num, (batch, feature_size)]
        output_hiddens = []
        for i in range(layers_num):
            weights_dicts = layer_weights_dicts[i]
            # input_seqs shape = [seq_num, (batch, feature_size)] or
            # [seq_num, (batch, 2*feature_size)] for bidirectional
            if bidirectional:
                input_seqs, H_t = self.bidir_gru_cell(input_seqs, weights_dicts)
            else:
                input_seqs, H_t = gru_cell(input_seqs, **weights_dicts[0])

            output_hiddens.append(H_t)

            # TODO (vvchernov): in pytorch implementation train is also checked
            # see https://github.com/pytorch/pytorch/blob/70c8daf43946b53af6493d058899ef952d27d339
            # /aten/src/ATen/native/RNN.cpp#L1054
            if dropout_p != 0 and i < layers_num - 1:
                # for input in input_seqs:
                #     input = _op.dropout(input, dropout_p)
                raise NotImplementedError("Dropout for GRU has not been supported yet!")

        return _op.stack(input_seqs, 0), _op.stack(output_hiddens, 0)

    def gru(self, inputs, input_types):
        """
        Description of GRU in pytorch:
        https://pytorch.org/docs/stable/generated/torch.nn.GRU.html?highlight=gru#torch.nn.GRU
        """
        # TODO (vvchernov): support dropout
        assert len(inputs) == 9, "Input of size 9 is expected"
        # Unpack inputs, note that if optional and not provided then value will be None.
        _X = inputs[0]
        # _X shape (seq_num, batch, feature_size) or (batch, seq_num, feature_size)

        hidden_state = inputs[1]
        # Hidden state shape (hidden_layers_num, batch, hidden_size)

        _weights = inputs[2]
        # Wi layer[0] shape (3 * hidden_size, feature_size)
        # Wh layer[0] shape (3 * hidden_size, hidden_size)
        # Bi layer[0] shape (3 * hidden_size)
        # Bh layer[0] shape (3 * hidden_size)

        # Wi layer[>0] shape (3 * hidden_size, hidden_size * num_directions)
        # Wh layer[>0] shape (3 * hidden_size, hidden_size)
        # Bi layer[>0] shape (3 * hidden_size)
        # Bh layer[>0] shape (3 * hidden_size)

        # Scalar inputs
        has_biases = inputs[3]
        num_layers = inputs[4]
        dropout_p = inputs[5]  # dropout probability, if 0.0 it means there is no dropout
        # train = inputs[6]
        bidirectional = inputs[7]
        batch_first = inputs[8]

        num_directions = 1
        if bidirectional:
            num_directions = 2

        rsd = len(_weights) % num_layers
        assert rsd == 0, "The number of weights must be a multiple of the number of layers!"
        rsd = (len(_weights) / num_layers) % num_directions
        assert (
            rsd == 0
        ), "The number of weights in layer must be a multiple of the number of directions!"

        weights_num = int(len(_weights) / num_layers / num_directions)
        if has_biases:
            assert weights_num == 4, "The weights number in layer is expected equal to 4"
        else:
            assert weights_num == 2, "The weights number in layer is expected equal to 2"

        X = _op.transpose(_X, (1, 0, 2)) if batch_first else _X
        # TODO (vvchernov): Which data type should be used? from input or weights?
        # Instead of it _infer_type(X).checked_type.dtype can be used
        X_dtype = input_types[0]
        X_shape = _infer_shape(X)  # (seq_num, batch, feature_size)

        hidden_size = int(_infer_shape(_weights[0])[0] / 3)
        batch_size = X_shape[1]

        # Initialize hidden states if not provided.
        layers_h = []
        hidden_layers_num = num_directions * num_layers
        if hidden_state is None:
            h_0 = _op.zeros((batch_size, hidden_size), X_dtype)
            for i in range(hidden_layers_num):
                layers_h.append(h_0)
        else:
            layers_h = unbind(hidden_state, 0)

        layer_weights_dicts = []
        k = 0  # layer counter
        if has_biases:
            names = ["hidden_state", "w_inp", "w_hid", "b_inp", "b_hid"]
            if bidirectional:
                rsd = len(_weights) % (2 * weights_num)
                assert rsd == 0, "got an incorrect number of GRU weights"
                for i in range(0, len(_weights), 2 * weights_num):
                    fw_tensors = [layers_h[2 * k], *_weights[i : i + 4]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    j = i + weights_num
                    rev_tensors = [layers_h[2 * k + 1], *_weights[j : j + 4]]
                    rev_weights_dict = dict(zip(names, rev_tensors))
                    layer_weights_dicts.append([fw_weights_dict, rev_weights_dict])
                    k += 1
            else:
                assert len(_weights) % weights_num == 0, "got an incorrect number of GRU weights"
                for i in range(0, len(_weights), weights_num):
                    fw_tensors = [layers_h[k], *_weights[i : i + 4]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    layer_weights_dicts.append([fw_weights_dict])
                    k += 1
        else:
            names = ["hidden_state", "w_inp", "w_hid"]
            if bidirectional:
                rsd = len(_weights) % (2 * weights_num)
                assert rsd == 0, "got an incorrect number of GRU weights"
                for i in range(0, len(_weights), 2 * weights_num):
                    fw_tensors = [layers_h[2 * k], *_weights[i : i + 2]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    j = i + weights_num
                    rev_tensors = [layers_h[2 * k + 1], *_weights[j : j + 2]]
                    rev_weights_dict = dict(zip(names, rev_tensors))
                    layer_weights_dicts.append([fw_weights_dict, rev_weights_dict])
                    k += 1
            else:
                assert len(_weights) % weights_num == 0, "got an incorrect number of GRU weights"
                for i in range(0, len(_weights), weights_num):
                    fw_tensors = [layers_h[k], *_weights[i : i + 2]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    layer_weights_dicts.append([fw_weights_dict])
                    k += 1
        assert (
            len(layer_weights_dicts) == num_layers and k == num_layers
        ), "For stacked GRU number of weights sets should be the same as number of layers!"

        output, out_hidden_state = self.gru_layers(
            X, layer_weights_dicts, bidirectional, dropout_p=dropout_p
        )

        # output shape = (seq_num, batch, hidden_size) or
        # (seq_num, batch, 2*feature_size) for bidirectional
        if batch_first:
            output = _op.transpose(output, (1, 0, 2))

        return (output, out_hidden_state)

    def bidir_lstm_cell(self, input_seqs, weights_dicts):
        """
        Bidirectional LSTM cell
        """
        seq_len = len(input_seqs)
        forward_outputs, fw_H_t, fw_C_t = lstm_cell(input_seqs, **weights_dicts[0])

        reverse_outputs, rev_H_t, rev_C_t = lstm_cell(
            input_seqs, **weights_dicts[1], backwards=True
        )

        final_outputs = []
        for i in range(seq_len):
            final_outputs.append(
                _op.concatenate([forward_outputs[i], reverse_outputs[seq_len - 1 - i]], axis=-1)
            )

        return final_outputs, (fw_H_t, fw_C_t), (rev_H_t, rev_C_t)

    def lstm_layers(self, input_data, layer_weights_dicts, bidirectional, dtype, dropout_p=0.0):
        """
        Methods iterates layers for Stacked LSTM
        """
        layers_num = len(layer_weights_dicts)
        # split input sequence to samples set
        input_seqs = unbind(input_data, 0)  # [seq_num, (batch, feature_size)]
        output_hiddens = []
        for i in range(layers_num):
            weights_dicts = layer_weights_dicts[i]
            # input_seqs shape = [seq_num, (batch, feature_size)] or
            # [seq_num, (batch, 2*feature_size)] for bidirectional
            if bidirectional:
                input_seqs, H_t, C_t = self.bidir_lstm_cell(input_seqs, weights_dicts)
            else:
                input_seqs, H_t, C_t = lstm_cell(input_seqs, **weights_dicts[0])

            output_hiddens.append((H_t, C_t))

            # TODO (vvchernov): in pytorch implementation train is also checked
            # see https://github.com/pytorch/pytorch/blob/70c8daf43946b53af6493d058899ef952d27d339
            # /aten/src/ATen/native/RNN.cpp#L1054
            if dropout_p != 0 and i < layers_num - 1:
                # for input in input_seqs:
                #     input = _op.dropout(input, dropout_p)
                raise NotImplementedError("Dropout for LSTM has not been supported yet!")
        final_hiddens = []
        if bidirectional:
            for output_hidden in output_hiddens:
                final_hiddens.append(output_hidden[0])
                final_hiddens.append(output_hidden[1])
        else:
            final_hiddens = output_hiddens

        return _op.stack(input_seqs, 0), final_hiddens

    def lstm(self, inputs, input_types):
        """
        Description of LSTM in pytorch:https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        Native implementation for torch version less than 1.8.0 (projection is unsupported):
        https://github.com/pytorch/pytorch/blob/70c8daf43946b53af6493d058899ef952d27d339/aten/ \
        src/ATen/native/RNN.cpp#L1396
        Native implementation for torch version from 1.8.0 and higher (projection is supported):
        https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/RNN.cpp#L1483
        """
        # TODO (vvchernov): support dropout
        assert len(inputs) == 9, "Input of size 9 is expected"
        # Unpack inputs, note that if optional and not provided then value will be None.
        _X = inputs[0]
        # _X shape (seq_num, batch, feature_size) or (batch, seq_num, feature_size)

        hidden_states = inputs[1]
        assert len(hidden_states) == 2, "lstm expects two hidden states"
        h_0 = hidden_states[0]
        c_0 = hidden_states[1]
        # H0 shape (hidden_layers_num, batch, proj_size) if projection
        # else (hidden_layers_num, batch, hidden_size)
        # C0 shape (hidden_layers_num, batch, hidden_size)

        _weights = inputs[2]
        # If no projection
        # Wi layer[0] shape (4 * hidden_size, feature_size)
        # Wh layer[0] shape (4 * hidden_size, hidden_size)
        # Bi layer[0] shape (4 * hidden_size)
        # Bh layer[0] shape (4 * hidden_size)

        # Wi layer[>0] shape (4 * hidden_size, hidden_size * num_directions)
        # Wh layer[>0] shape (4 * hidden_size, hidden_size)
        # Bi layer[>0] shape (4 * hidden_size)
        # Bh layer[>0] shape (4 * hidden_size)

        # If projection
        # Wi layer[0] shape (4 * hidden_size, feature_size)
        # Wh layer[0] shape (4 * hidden_size, proj_size)
        # Bi layer[0] shape (4 * hidden_size)
        # Bh layer[0] shape (4 * hidden_size)
        # P  layer[0] shape (proj_size, hidden_size)

        # Wi layer[>0] shape (4 * hidden_size, proj_size * num_directions)
        # Wh layer[>0] shape (4 * hidden_size, proj_size)
        # Bi layer[>0] shape (4 * hidden_size)
        # Bh layer[>0] shape (4 * hidden_size)
        # P  layer[>0] shape (proj_size, hidden_size)

        # Scalar inputs
        has_biases = inputs[3]
        num_layers = inputs[4]
        dropout_p = inputs[5]  # dropout probability, if 0.0 it means there is no dropout
        # train = inputs[6]
        bidirectional = inputs[7]
        batch_first = inputs[8]

        num_directions = 1
        if bidirectional:
            num_directions = 2

        rsd = len(_weights) % num_layers
        assert rsd == 0, "The number of weights must be a multiple of the number of layers!"
        rsd = (len(_weights) / num_layers) % num_directions
        assert (
            rsd == 0
        ), "The number of weights in layer must be a multiple of the number of directions!"
        has_proj = False
        proj_size = 0
        weights_num = int(len(_weights) / num_layers / num_directions)
        if has_biases:
            if weights_num == 5:
                has_proj = True
                proj_size = _infer_shape(_weights[4])[0]
            else:
                assert weights_num == 4, "The weights number in layer is expected equal to 4"
        else:
            if weights_num == 3:
                has_proj = True
                proj_size = _infer_shape(_weights[2])[0]
            else:
                assert weights_num == 2, "The weights number in layer is expected equal to 2"

        X = _op.transpose(_X, (1, 0, 2)) if batch_first else _X
        # TODO (vvchernov): Which data type should be used? from input or weights?
        # Instead of it _infer_type(X).checked_type.dtype can be used
        X_dtype = input_types[0]
        X_shape = _infer_shape(X)  # (seq_num, batch, feature_size)

        hidden_size = _infer_shape(_weights[0])[0] / 4
        batch_size = X_shape[1]

        # Initialize hidden states if not provided.
        layers_h = []
        layers_c = []
        hidden_layers_num = num_directions * num_layers
        if h_0 is None:
            if has_proj:
                h_0 = _op.zeros((batch_size, proj_size), X_dtype)
            else:
                h_0 = _op.zeros((batch_size, hidden_size), X_dtype)
            for i in range(hidden_layers_num):
                layers_h.append(h_0)
        else:
            layers_h = unbind(h_0, 0)
        if c_0 is None:
            c_0 = _op.zeros((batch_size, hidden_size), X_dtype)
            for i in range(hidden_layers_num):
                layers_c.append(c_0)
        else:
            layers_c = unbind(c_0, 0)

        layer_weights_dicts = []
        k = 0  # layer counter
        if has_biases:
            names = ["hidden_state", "cell_state", "w_inp", "w_hid", "b_inp", "b_hid"]
            if bidirectional:
                rsd = len(_weights) % (2 * weights_num)
                assert rsd == 0, "got an incorrect number of LSTM weights"
                for i in range(0, len(_weights), 2 * weights_num):
                    fw_tensors = [layers_h[2 * k], layers_c[2 * k], *_weights[i : i + 4]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    if has_proj:
                        fw_weights_dict["proj"] = _weights[i + 4]
                    j = i + weights_num
                    rev_tensors = [layers_h[2 * k + 1], layers_c[2 * k + 1], *_weights[j : j + 4]]
                    rev_weights_dict = dict(zip(names, rev_tensors))
                    if has_proj:
                        rev_weights_dict["proj"] = _weights[j + 4]
                    layer_weights_dicts.append([fw_weights_dict, rev_weights_dict])
                    k += 1
            else:
                assert len(_weights) % weights_num == 0, "got an incorrect number of LSTM weights"
                for i in range(0, len(_weights), weights_num):
                    fw_tensors = [layers_h[k], layers_c[k], *_weights[i : i + 4]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    if has_proj:
                        fw_weights_dict["proj"] = _weights[i + 4]
                    layer_weights_dicts.append([fw_weights_dict])
                    k += 1
        else:
            names = ["hidden_state", "cell_state", "w_inp", "w_hid"]
            if bidirectional:
                rsd = len(_weights) % (2 * weights_num)
                assert rsd == 0, "got an incorrect number of LSTM weights"
                for i in range(0, len(_weights), 2 * weights_num):
                    fw_tensors = [layers_h[2 * k], layers_c[2 * k], *_weights[i : i + 2]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    if has_proj:
                        fw_weights_dict["proj"] = _weights[i + 2]
                    j = i + weights_num
                    rev_tensors = [layers_h[2 * k + 1], layers_c[2 * k + 1], *_weights[j : j + 2]]
                    rev_weights_dict = dict(zip(names, rev_tensors))
                    if has_proj:
                        rev_weights_dict["proj"] = _weights[j + 2]
                    layer_weights_dicts.append([fw_weights_dict, rev_weights_dict])
                    k += 1
            else:
                assert len(_weights) % weights_num == 0, "got an incorrect number of LSTM weights"
                for i in range(0, len(_weights), weights_num):
                    fw_tensors = [layers_h[k], layers_c[k], *_weights[i : i + 2]]
                    fw_weights_dict = dict(zip(names, fw_tensors))
                    if has_proj:
                        fw_weights_dict["proj"] = _weights[i + 2]
                    layer_weights_dicts.append([fw_weights_dict])
                    k += 1
        assert (
            len(layer_weights_dicts) == num_layers and k == num_layers
        ), "For stacked LSTM number of weights sets should be the same as number of layers!"

        outputs = self.lstm_layers(
            X, layer_weights_dicts, bidirectional, dtype=X_dtype, dropout_p=dropout_p
        )

        # output shape = (seq_num, batch, hidden_size) or
        # (seq_num, batch, 2*feature_size) for bidirectional
        output = outputs[0]

        hy = []
        cy = []
        for hidden in outputs[1]:
            hy.append(hidden[0])
            cy.append(hidden[1])

        if batch_first:
            output = _op.transpose(output, (1, 0, 2))

        return (output, _op.stack(hy, 0), _op.stack(cy, 0))

    def all_any_common(self, op, inputs, input_types):
        # return True
        if len(inputs) > 1:
            dim = inputs[1]
        else:
            dim = 0

        if len(inputs) > 2:
            keepdim = inputs[2]
        else:
            keepdim = False

        if self.infer_type(inputs[0]).dtype != "bool":
            # The input dtype can be uint8.
            inp = _op.cast(inputs[0], "bool")
        else:
            inp = inputs[0]
        return op(inp, axis=dim, keepdims=keepdim)

    def searchsorted_common(
        self, sorted_sequence, values, out_int32, right, side=None, out=None, sorter=None
    ):
        assert side is None and out is None and sorter is None, "unsupported parameters"
        dtype = "int32" if out_int32 else "int64"
        values_shape = _infer_shape(values)

        if len(values_shape) == 0:
            values = _op.expand_dims(values, 0)

        out = _op.searchsorted(sorted_sequence, values, right=right, dtype=dtype)

        if len(values_shape) == 0:
            return _op.squeeze(out)

        return out

    def searchsorted(self, inputs, input_types):
        return self.searchsorted_common(*inputs)

    def bucketize(self, inputs, input_types):
        return self.searchsorted_common(inputs[1], inputs[0], inputs[2], inputs[3])

    def roll(self, inputs, input_types):
        def slide_axes(inp, shape, ax):
            axes = list(range(len(shape)))
            axes = axes[:ax] + [-1] + axes[ax:-1]
            return _op.transpose(inp, axes)

        x = inputs[0]
        shifts = inputs[1]
        dims = inputs[2]
        shape = self.infer_shape(x)
        start = _expr.const(0, "int64")
        step = _expr.const(1, "int64")

        out = x
        for i, dim in enumerate(dims):
            roll_dim = _expr.const(shape[dim], "int64")
            indices_1d = _op.mod(
                _op.transform.arange(start, roll_dim, step, "int64")
                - _expr.const(shifts[i], "int64")
                + roll_dim,
                roll_dim,
            )
            # First fill in the last axis with roll indices, and then do transpose to
            # bring the roll indices into the desired axis.
            indices = slide_axes(
                _op.tile(indices_1d, shape[:dim] + shape[dim + 1 :] + (1,)), shape, dim
            )
            out = _op.gather(out, dim, indices)

        return out

    def einsum(self, inputs, input_types):
        equation = inputs[0]
        data = inputs[1]
        return _op.einsum(data, equation)

    def dot(self, inputs, _):
        lhs, rhs = inputs
        return _op.sum(_op.multiply(lhs, rhs))

    def mv(self, inputs, _):
        lhs, rhs = inputs

        # Convert the 1D matrix (vector) into a 2D matrix with the extra
        # dimension=1
        rhs_matrix = _op.transform.expand_dims(rhs, 0)

        # Run multiplication
        dense_result = _op.nn.dense(lhs, rhs_matrix, units=None)

        # Chop off the extra result dimension
        return _op.transform.squeeze(dense_result)

    def grid_sampler(self, inputs, input_types):
        interpolate_mode = inputs[2]
        padding_mode = inputs[3]
        align_corners = inputs[4]
        data_shape = self.infer_shape_with_prelude(inputs[0])

        if len(data_shape) == 4:
            layout = "NCHW"
            axes = [0, 3, 1, 2]
            grid = _op.transform.transpose(inputs[1], axes)
        elif len(data_shape) == 5:
            layout = "NCDHW"
            axes = [0, 4, 1, 2, 3]
            grid = _op.transform.transpose(inputs[1], axes)
        else:
            msg = "only 4D and 5D are supported."
            raise ValueError(msg)

        if interpolate_mode == 0:
            interpolate_str = "bilinear"
        elif interpolate_mode == 1:
            interpolate_str = "nearest"
        elif interpolate_mode == 2:
            interpolate_str = "bicubic"
        else:
            msg = f"interpolation method {interpolate_mode} is not supported"
            raise ValueError(msg)

        if padding_mode == 0:
            padding_mode_str = "zeros"
        elif padding_mode == 1:
            padding_mode_str = "border"
        elif padding_mode == 2:
            padding_mode_str = "reflection"
        else:
            msg = f"padding_mode {padding_mode} is not supported"
            raise ValueError(msg)

        return _op.image.grid_sample(
            inputs[0], grid, interpolate_str, layout, padding_mode_str, align_corners
        )

    def trilu(self, inputs, input_types, mode):
        data = inputs[0]
        k = inputs[1] if inputs[1] else 0
        upper = True if mode == "triu" else False
        return _op.trilu(data, k, upper)

    def multinomial(self, inputs, input_types):
        probs = inputs[0]
        num_samples = inputs[1]
        replacement = inputs[2] if inputs[2] else True
        assert not (
            replacement is False and num_samples > 1
        ), "Multinomial without replacement is not yet supported."
        # Ideally this seed would be generated by a previous threefry operation.
        # Eventually we might want to add a global store for random keys.
        seed = np.random.randint(1e6)
        key = _op.random.threefry_key(seed)
        output = _op.random.multinomial(key, probs, num_samples)
        _, indices = _expr.TupleWrapper(output, 2)
        return indices
    
    def bernoulli(self, inputs, input_types):
        prob_tensor = inputs[0]
        seed = np.random.randint(1e6)
        key = _op.random.threefry_key(seed)  
        shape = self.infer_shape(prob_tensor)
        dtype = self.infer_type(prob_tensor).dtype

        rand_normal = tvm.relay.random.normal(key=key, shape=shape, dtype=dtype)

        _, sampled_tensor = tvm.relay.TupleWrapper(rand_normal, 2)
        
        output = _op.less(sampled_tensor, prob_tensor)
        return output
    
    def weight_norm(self, inputs, input_types):
        weight_v, weight_g = inputs[0], inputs[1]
        dim = inputs[2]
        dtype = input_types[0]
        order = 2.0
        reci_order = _expr.const(1.0 / order, dtype=dtype)
        order = _expr.const(order)

        norm_v = _op.power(
            _op.reduce.sum(_op.power(_op.abs(weight_v), order), axis=dim, exclude=2, keepdims=True),
            reci_order,
        )
        return weight_g * (weight_v / norm_v)
    def new_empty(self, inputs, input_types):
        shape = inputs[1]

        for i, val in enumerate(shape):
            if isinstance(val, _expr.Call):
                shape[i] = int(_infer_value(inputs[1][i], {}).numpy())

        return _op.zeros(shape, _convert_dtype_value(inputs[2]))

    def new_zeros(self, inputs, input_types):
        data = inputs[1]

        import torch

        if not isinstance(data, (_expr.Expr, list, tuple, torch.Size, torch.Tensor, np.ndarray)):
            msg = "Data type %s could not be parsed in zeros op" % (type(data))
            raise AssertionError(msg)

        if inputs[2] is not None:
            dtype = _convert_dtype_value(inputs[2])
        else:
            dtype = self.default_dtype

        for i, dim in enumerate(data):
            if isinstance(dim, int):
                data[i] = _expr.const(dim, dtype=dtype)

        return self.full_impl(data, 0, dtype)


    def new_ones(self, inputs, input_types):
        data = inputs[1]

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = "Data type %s could not be parsed in zeros op" % (type(data))
            raise AssertionError(msg)

        if inputs[2] is not None:
            dtype = _convert_dtype_value(inputs[2])
        else:
            dtype = self.default_dtype

        return self.full_impl(data, 1, dtype)


    def tril(self, inputs, input_types):
        x = inputs[0]
        x_shape = _infer_shape(x)
        diagonal = inputs[1]
        
        if isinstance(diagonal, tvm.relay.expr.Call):
            diagonal = _infer_value(diagonal,{})
            diagonal = diagonal.asnumpy().item()
        
        y = np.tril(np.ones(x_shape), k = diagonal).astype(_convert_tvm_to_np_dtype(input_types[0]))
        y = tvm.nd.array(y)
        y = tvm.relay.Constant(y)

        return _op.multiply(x, y)


    def triu(self, inputs, input_types):
        x = inputs[0]
        x_shape = _infer_shape(x)

        mask = np.triu(np.ones(x_shape), inputs[1]).astype(np.bool_)
        mask = tvm.nd.array(mask)
        mask = tvm.relay.Constant(mask)

        zeros = np.zeros(x_shape).astype(_convert_tvm_to_np_dtype(input_types[0]))
        zeros = tvm.nd.array(zeros)
        zeros = tvm.relay.Constant(zeros)

        return _op.where(mask, x, zeros)


    def as_strided(self, inputs, input_types):
        input_tensor = inputs[0]
        output_size = inputs[1]
        strides = inputs[2]

        input_dtype = str(self.infer_type(input_tensor).dtype)

        for i, out in enumerate(output_size):
            if not isinstance(out, int):
                output_size[i] = int(_infer_value(output_size[i], {}).numpy())
        n_out_batch = output_size[-4]
        n_out_time = output_size[-3]
        n_out_row = output_size[-2]
        n_out_col = output_size[-1]

        stride_batch = strides[-4]
        stride_time = strides[-3]
        stride_col = strides[-1]

        input_tensor = _op.transform.reshape(input_tensor, [-1])

        batch_rows = np.array([])
        for batch_dim in range(0, n_out_batch):

            time_rows = np.array([])
            for time_dim in range(0, n_out_time):

                rc_rows = np.array([])
                rc_begin = 0 + (time_dim * stride_time) + (batch_dim * stride_batch)
                rc_end = rc_begin + n_out_col * stride_col
                for _ in range(0, n_out_row):
                    strided_row = _op.transform.strided_slice(input_tensor, begin=[rc_begin], end=[rc_end], strides=[stride_col], slice_mode="end")
                    strided_row = _op.expand_dims(strided_row, axis=0)
                    strided_row = _op.cast(strided_row, input_dtype)
                    rc_rows = np.append(rc_rows, strided_row)

                    rc_begin += stride_col
                    rc_end = rc_begin + (n_out_col * stride_col)

                rc_rows = _op.concatenate(rc_rows, axis=0)
                rc_rows = _op.expand_dims(rc_rows, axis=0)
                time_rows = np.append(time_rows, rc_rows)

            time_rows = _op.concatenate(time_rows, axis=0)
            time_rows = _op.expand_dims(time_rows, axis=0)
            batch_rows = np.append(batch_rows, time_rows)

        return _op.concatenate(batch_rows, axis=0)


    def weight_norm(self, inputs, input_types):
        weight_v = inputs[0]
        weight_g = inputs[1]
        dim = inputs[2]

        # Axis on which to reduce
        axes = list(range(len(self.infer_shape(weight_v))))
        axes.remove(dim)

        # Euclidean normalization
        v_euc_norm = _op.sqrt(_op.reduce.sum((weight_v * weight_v), axis=axes, keepdims=True))

        # Weight normalization
        weight_norm = weight_v * (weight_g / v_euc_norm)

        return weight_norm

    def dim(self, inputs, input_types):
        if isinstance(inputs[0], _expr.Constant):
            return len(inputs[0].data.shape)
        else:
            return len(inputs[0].type_annotation.shape)

    def is_impl(self, inputs, input_types):
        return _expr.const(inputs[0] is inputs[1], dtype='bool')

    def as_tensor(self, inputs, input_types):
        vals = []
        for val in inputs[0]:
            vals.append(val.data.numpy().item())
        return _expr.const(vals, dtype=inputs[1])

    def format(self, inputs, input_types):
        return inputs[0]

    def warn(self, inputs, inputs_types):
        return inputs[0]

    def alias(self, inputs, inputs_types):
        shape = inputs[0]

        if not isinstance(shape, _expr.Constant):
            raise AssertionError("Shape must be a constant")

        # Get constant dtype
        dtype = _convert_data_type(shape.data.dtype, default_dtype="float32")

        # Convert to numpy array
        shape = shape.data.numpy()
        if len(shape.shape) == 0:
            shape = np.array([shape.item()])

        return self.full_impl(shape, 0, dtype)

    def linalg_vector_norm(self, inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        ord = inputs[1]
        dim = inputs[2]
        keepdim = inputs[3]

        assert dtype == "float32" or dtype == "float64"

        if ord == 0:
            return _op.reduce.sum(
                _op.cast(_op.not_equal(data, _expr.const(0, dtype=dtype)), dtype=dtype),
                axis=dim,
                keepdims=keepdim,
            )
        elif ord == np.inf:
            return _op.reduce.max(_op.abs(data), axis=dim, keepdims=keepdim)
        elif ord == np.NINF:
            return _op.reduce.min(_op.abs(data), axis=dim, keepdims=keepdim)
        reci_ord = _expr.const(1.0 / ord, dtype=dtype)
        ord = _expr.const(ord, dtype=dtype)
        return _op.power(
            _op.reduce.sum(_op.power(_op.abs(data), ord), axis=dim, keepdims=keepdim),
            reci_ord,
        )

    def scaled_dot_product_attention(self, inputs, input_types):
        query = inputs[0]
        key = inputs[1]
        value = inputs[2]
        attn_mask = inputs[3] if len(inputs) > 3 else None
        dropout_p = inputs[4] if len(inputs) > 4 else 0.0
        is_causal = inputs[5] if len(inputs) > 5 else False
        scale = inputs[6] if len(inputs) > 6 else None
        enable_gqa = inputs[7] if len(inputs) > 7 else False

        assert (
            input_types[0] == input_types[1] == input_types[2]
        ), "Expected query, key, and value to have the same dtype"

        dtype = input_types[0]
        assert dtype == "float32" or dtype == "float64", "Data type can be float32 or float64"

        query_shape = self.infer_shape_with_prelude(query)
        key_shape = self.infer_shape_with_prelude(key)
        value_shape = self.infer_shape_with_prelude(value)
        assert 3 <= len(query_shape) <= 4, "Only 3D or 4D query supported"
        assert 3 <= len(key_shape) <= 4, "Only 3D or 4D key supported"
        assert 3 <= len(value_shape) <= 4, "Only 3D or 4D value supported"

        assert dropout_p == 0.0, "Only dropout_p==0.0 supported"

        L, S = query_shape[-2], key_shape[-2]

        enable_flash_attention = bool(int(os.environ.get("FORGE_ENABLE_FLASH_ATTENTION", 0)))

        if enable_flash_attention:
            # Return a scaled_dot_product_attention op to Forge
            assert scale is None, "Must not provide a scale factor"
            assert attn_mask is not None, "Explicit attn_mask should be set when is_causal=False"
            assert not is_causal, "Not supporting is_causal=True yet since we would need to generate causal mask."

            scale_factor = _expr.const(1 / math.sqrt(query_shape[-1]), dtype=dtype)
            scale_factor = _op.broadcast_to(scale_factor, shape=tuple(1 for _ in range(len(query_shape))))

            # Early out if not decomposing
            return _op.nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask,
            scale_factor,
            is_causal=is_causal
            )


        if scale is None:
            scale_factor = _expr.const(1 / math.sqrt(query_shape[-1]), dtype=dtype)
        else:
            scale_factor = _expr.const(scale, dtype=dtype)

        attn_bias = _op.full(_expr.const(0.0, dtype=dtype), (L, S))

        if is_causal:
            assert attn_mask is None, "Explicit attn_mask shouldn't be set when is_causal=True"
            temp_mask = _op.full(_expr.const(True), [L, S], dtype="bool")
            temp_mask = _op.trilu(temp_mask, 0, upper=False)
            temp_mask = _op.cast(temp_mask, dtype="bool")
            temp_mask = _op.logical_not(temp_mask)
            fill_value = _op.cast(_expr.const(float("-inf")), dtype=dtype)
            attn_bias = _op.where(temp_mask, fill_value, attn_bias)
            attn_bias = _op.cast(attn_bias, dtype)

        if attn_mask is not None:
            if input_types[3] == "bool":
                attn_mask = _op.logical_not(attn_mask)
                fill_value = _op.cast(_expr.const(float("-inf")), dtype=dtype)
                attn_bias = _op.where(attn_mask, fill_value, attn_bias)
            else:
                attn_bias = _op.add(attn_bias, attn_mask)

        if len(query_shape) < len(key_shape):
            batch_size = key_shape[0]
        else:
            batch_size = query_shape[0]

        if len(query_shape) == 4 and len(key_shape) == 4:
            query = _op.reshape(query, newshape=[-3, -2])
            key = _op.reshape(key, newshape=[-3, -2])
        if len(query_shape) == 3 and len(key_shape) == 4:
            query = _op.broadcast_to(query, shape=(batch_size,) + query_shape)
            query = _op.reshape(query, newshape=[-3, -2])
            key = _op.reshape(key, newshape=[-3, -2])
        if len(query_shape) == 4 and len(key_shape) == 3:
            query = _op.reshape(query, newshape=[-3, -2])
            key = _op.broadcast_to(key, shape=(batch_size,) + key_shape)
            key = _op.reshape(key, newshape=[-3, -2])

        if enable_gqa:
            key = _op.transform.repeat(key, repeats=query_shape[-3] // key_shape[-3], axis=-3)
            value = _op.transform.repeat(value, repeats=query_shape[-3] // value_shape[-3], axis=-3)

        attn_weight = _op.nn.batch_matmul(query, key)
        if len(query_shape) == 4 or len(key_shape) == 4:
            attn_weight = _op.reshape(attn_weight, newshape=[-4, batch_size, -1, -2])
        attn_weight = _op.squeeze(attn_weight, axis=[])

        attn_weight = _op.multiply(attn_weight, scale_factor)
        attn_weight = _op.add(attn_weight, attn_bias)
        attn_weight = _op.nn.softmax(attn_weight)
        attn_weight = _op.nn.dropout(attn_weight, rate=dropout_p)

        aw_shape = self.infer_shape_with_prelude(attn_weight)
        if len(aw_shape) < len(value_shape):
            batch_size = value_shape[0]
        else:
            batch_size = aw_shape[0]
        if len(aw_shape) == 4 and len(value_shape) == 4:
            attn_weight = _op.reshape(attn_weight, newshape=[-3, -2])
            value = _op.reshape(value, newshape=[-3, -2])
        if len(aw_shape) == 3 and len(value_shape) == 4:
            attn_weight = _op.broadcast_to(attn_weight, shape=(batch_size,) + aw_shape)
            attn_weight = _op.reshape(attn_weight, newshape=[-3, -2])
            value = _op.reshape(value, newshape=[-3, -2])
        if len(aw_shape) == 4 and len(value_shape) == 3:
            attn_weight = _op.reshape(attn_weight, newshape=[-3, -2])
            value = _op.broadcast_to(value, shape=(batch_size,) + value_shape)
            value = _op.reshape(value, newshape=[-3, -2])
        attn_weight = _op.nn.batch_matmul(attn_weight, value, transpose_b=False)
        if len(aw_shape) == 4 or len(value_shape) == 4:
            attn_weight = _op.reshape(attn_weight, newshape=[-4, batch_size, -1, -2])

        return attn_weight
    
     
    def nan_to_num(self, inputs, input_types):
        
        """
        Mimics the behavior of torch.nan_to_num (https://pytorch.org/docs/stable/generated/torch.nan_to_num.html).
        """
        
        # Extract input tensor and replacement values 
        data = inputs[0]
        nan_value = inputs[1]
        posinf = inputs[2]
        neginf = inputs[3]

        # Ensure the data type is one of the supported floating-point types
        dtype = input_types[0]
        assert dtype in ["float16", "float32"], f"Unsupported dtype: {dtype}. Supported types are ['float16', 'float32']."

        # Define the maximum and minimum representable values for the data type
        dtype_max = np.finfo(dtype).max
        dtype_min = np.finfo(dtype).min

        # Create constants for NaN, positive infinity, and negative infinity replacements
        nan_tensor = tvm.relay.const(nan_value if nan_value is not None else 0.0, dtype)
        posinf_tensor = tvm.relay.const(posinf if posinf is not None else dtype_max, dtype)
        neginf_tensor = tvm.relay.const(neginf if neginf is not None else dtype_min, dtype)
        
        # Replace NaN values with the specified or default value
        data= tvm.relay.where(_op.isnan(data)  , nan_tensor, data)
        
        # Replace positive infinity with the specified or greatest finite value representable by input’s dtype
        data = tvm.relay.where(tvm.relay.greater(data, posinf_tensor),posinf_tensor, data)
        
        # Replace negative infinity with the specified or least finite value representable by input’s dtype
        result = tvm.relay.where(tvm.relay.less(data, neginf_tensor),  neginf_tensor, data)
       
        return result

        
    def atan2(self, inputs, input_types):
        
        """
        Mimics the behavior of torch.atan2 (https://pytorch.org/docs/stable/generated/torch.atan2.html).
        """
        
        data_1 = inputs[1]  # x (denominator)
        data_2 = inputs[0]  # y (numerator)

        # Compute the ratio y/x. This is the tangent of the angle.
        ratio = tvm.relay.divide(data_2, data_1)

        # Compute the arctangent of the ratio, which gives the angle in the range [-π/2, π/2].
        atan_res = tvm.relay.atan(ratio)

        # Define constants for π and 0 for use in correction logic.
        pi = tvm.relay.const(np.pi, "float32")  # π constant
        zero = tvm.relay.const(0.0, "float32")  # Zero constant

        # Compute the correction term to adjust the angle to the correct quadrant.
        # If x < 0:
        #   - If y >= 0, add π to the angle (to move from 1st to 2nd quadrant).
        #   - If y < 0, subtract π from the angle (to move from 4th to 3rd quadrant).
        # If x >= 0, no correction is needed.
        correction = tvm.relay.where(
            tvm.relay.less(data_1, zero),  # Check if x < 0
            tvm.relay.where(
                tvm.relay.greater_equal(data_2, zero),  # Check if y >= 0
                pi,  # Add π if x < 0 and y >= 0
                -pi  # Subtract π if x < 0 and y < 0
            ),
            zero  # No correction if x >= 0
        )

        # Add the correction term to the arctangent result.
        result = tvm.relay.add(atan_res, correction)
        
        return result
    
    def resolve_conj(self, inputs, input_types):
        # TVM doesn't track conjugate/negate flags; input is already materialized.
        # Returning the input directly is appropriate for resolve_conj.
        return inputs
    
    def resolve_neg(self, inputs, input_types):
        # TVM doesn't track conjugate/negate flags; input is already materialized.
        # Returning the input directly is appropriate for resolve_neg.
        return inputs
        
    
    def unflatten(self, inputs, input_types):
        
        """
        Mimics the behavior of torch.unflatten (https://pytorch.org/docs/stable/generated/torch.unflatten.html).
        """
        data = inputs[0]
        dim = int(inputs[1])
        unflattened_size = tuple(inputs[2])
        dshape = get_const_tuple(self.infer_shape_with_prelude(data))

        # Handle negative dimension
        dim = dim if dim >= 0 else len(dshape) + dim
        assert len(dshape) > dim >= 0

        # Ensure at most one inferred dimension (-1)
        assert unflattened_size.count(-1) <= 1
        
        # Helper to convert each element of unflattened_size to a valid int
        def extract_int(val):
            if isinstance(val, _expr.Constant):
                # Extract integer from relay Constant
                return int(val.data.numpy().item())
            elif isinstance(val, int):
                # Already a plain Python int
                return val
            else:
                raise TypeError(f"Unsupported shape type in unflatten: {type(val)} with value {val}")

        # Convert the unflattened size tuple into a pure tuple of Python ints
        unflattened_shape = tuple(extract_int(v) for v in unflattened_size)
        
        # Construct the new shape by replacing the target dim with unflattened_shape
        new_shape = dshape[:dim] + unflattened_shape + dshape[dim + 1 :]
        
        out = _op.reshape(data, new_shape)
        return out
    
    def block_diag(self, inputs, input_types):
        
        """
        Mimics the behavior of torch.block_diag (https://docs.pytorch.org/docs/stable/generated/torch.block_diag.html).
        """
        
        # Pre-check: ensure all input tensors have rank 1 or 2
        for inp in inputs[0]:
            shape = _infer_shape(inp)
            if len(shape) > 2:
                assert False, f"Input tensor rank should be <= 2, but got shape {shape}"
        
        shapes = []
        new_inputs = []

        # Prepare all input tensors and record their shapes
        for inp in inputs[0]:
            shape = _infer_shape(inp)
            
            if len(shape) == 1:
                # Reshape 1D tensor to 2D (1, N) to align for block-diagonal layout
                flat = tvm.relay.expand_dims(inp, axis=0)
                new_inputs.append(flat)
                shapes.append((1, shape[0]))
            elif len(shape) == 2:
                # Keep 2D tensors as-is
                new_inputs.append(inp)
                shapes.append(shape)

        # Calculate the final output shape after block-diagonal stacking
        total_rows = sum(s[0] for s in shapes)
        total_cols = sum(s[1] for s in shapes)

        blocks = []
        row_offset = 0
        col_offset = 0

        # Pad each tensor to position it correctly in the final block-diagonal matrix
        for inp, (rows, cols) in zip(new_inputs, shapes):
            pad = (
                (row_offset, total_rows - row_offset - rows),  # pad before/after rows
                (col_offset, total_cols - col_offset - cols),  # pad before/after columns
            )
            padded = tvm.relay.nn.pad(inp, pad_width=pad, pad_value=tvm.relay.const(0))
            blocks.append(padded)

            # Update position for next tensor
            row_offset += rows
            col_offset += cols

        # Add all padded tensors to get the final result
        result = blocks[0]
        for block in blocks[1:]:
            result = tvm.relay.op.add(result, block)

        return result


    # Operator mappings
    def create_convert_map(self):
        self.convert_map = {
            "aten::is_floating_point": self.is_floating_point,
            "aten::pixel_shuffle": self.pixel_shuffle,
            "aten::device": self.none,
            "prim::device": self.none,
            "aten::sub": self.sub,
            "aten::max": self.max,
            "aten::min": self.min,
            "aten::maximum": self.maximum,
            "aten::minimum": self.minimum,
            "aten::amax": self.max,
            "aten::amin": self.min,
            "aten::stft": self.stft,
            "aten::mul": self.make_elemwise("multiply"),
            "aten::pow": self.make_elemwise("power"),
            "aten::lerp": self.lerp,
            "aten::arange": self.arange,
            "aten::meshgrid": self.meshgrid,
            "aten::multinomial": self.multinomial,
            "aten::bernoulli": self.bernoulli,
            "aten::div": self.make_elemwise("divide"),
            "aten::floor_divide": self.make_elemwise("floor_divide"),
            "aten::floordiv": self.make_elemwise("floor_divide"),
            "aten::true_divide": self.make_elemwise("divide"),
            "aten::fmod": self.make_elemwise("trunc_mod"),
            "aten::remainder": self.make_elemwise("floor_mod"),
            "aten::addcdiv": self.addcdiv,
            "aten::addcmul": self.addcmul,
            "aten::ones": self.ones,
            "aten::ones_like": self.ones_like,
            "aten::zeros": self.zeros,
            "aten::zero_": self.zero_,
            "aten::zeros_like": self.zeros_like,
            "aten::new_ones": self.new_ones,
            "aten::full": self.full,
            "aten::fill": self.fill,
            "aten::full_like": self.full_like,
            "aten::new_full": self.new_full,
            "aten::fill_": self.fill_,
            "aten::linspace": self.linspace,
            "aten::reciprocal": self.reciprocal,
            "aten::repeat": self.repeat,
            "aten::repeat_interleave": self.repeat_interleave,
            "aten::tensordot": self.tensordot,
            "aten::to": self.to,
            "aten::squeeze": self.squeeze,
            "aten::unsqueeze": self.unsqueeze,
            "aten::cat": self.concatenate,
            "aten::concat": self.concatenate,
            "aten::slice": self.slice,
            "aten::narrow": self.narrow,
            "aten::split": self.split,
            "aten::tensor_split": self.tensor_split,
            "aten::split_with_sizes": self.split_with_sizes,
            "aten::select": self.select,
            "aten::take": self.take,
            "aten::where": self.where,
            "aten::topk": self.topk,
            "aten::relu": self.relu,
            "aten::relu6": self.relu6,
            "aten::prelu": self.prelu,
            "aten::leaky_relu": self.leaky_relu,
            "aten::elu": self.elu,
            "aten::celu": self.celu,
            "aten::gelu": self.gelu,
            "aten::selu": self.selu,
            "aten::silu": self.silu,
            "aten::glu": self.glu,
            "aten::log_sigmoid": self.log_sigmoid,
            "aten::adaptive_avg_pool1d": functools.partial(
                self.adaptive_avg_pool, _op.nn.adaptive_avg_pool1d
            ),
            "aten::adaptive_avg_pool2d": functools.partial(
                self.adaptive_avg_pool, _op.nn.adaptive_avg_pool2d
            ),
            "aten::adaptive_avg_pool3d": functools.partial(
                self.adaptive_avg_pool, _op.nn.adaptive_avg_pool3d
            ),
            "aten::adaptive_max_pool1d": functools.partial(
                self.adaptive_max_pool, _op.nn.adaptive_max_pool1d
            ),
            "aten::adaptive_max_pool2d": functools.partial(
                self.adaptive_max_pool, _op.nn.adaptive_max_pool2d
            ),
            "aten::adaptive_max_pool3d": functools.partial(
                self.adaptive_max_pool, _op.nn.adaptive_max_pool3d
            ),
            "aten::max_pool2d": self.maxpool_2d,
            "aten::max_pool2d_with_indices": self.maxpool_2d_with_indices,
            "aten::max_pool1d": self.maxpool_1d,
            "aten::max_pool3d": self.maxpool_3d,
            "aten::hardtanh": self.hardtanh,
            "aten::_convolution": self.convolution,
            "aten::softmax": self.softmax,
            "aten::threshold": self.threshold,
            "aten::contiguous": self.contiguous,
            "aten::batch_norm": self.batch_norm,
            "aten::instance_norm": self.instance_norm,
            "aten::layer_norm": self.layer_norm,
            "aten::group_norm": self.group_norm,
            "aten::transpose": self.transpose,
            "aten::t": self.transpose,
            "aten::numpy_T": self.transpose,
            "aten::flatten": self.flatten,
            "aten::addmm": self.addmm,
            "aten::size": self.size,
            "aten::view": self.view,
            "aten::view_as": self.view_as,
            "aten::reshape": self.reshape,
            "aten::reshape_as": self.reshape_as,
            "aten::clone": self.clone,
            "aten::log_softmax": self.log_softmax,
            "aten::sigmoid": self.sigmoid,
            "aten::softplus": self.softplus,
            "aten::avg_pool1d": self.make_avg_pool(1),
            "aten::avg_pool2d": self.make_avg_pool(2),
            "aten::avg_pool3d": self.make_avg_pool(3),
            "aten::linear": self.linear,
            "aten::dropout": self.dropout,
            "aten::feature_dropout": self.dropout,
            "aten::alpha_dropout": self.dropout,
            "aten::mean": self.mean,
            "aten::chunk": self.chunk,
            "aten::unsafe_chunk": self.chunk,
            "aten::matmul": self.matmul,
            "aten::bmm": self.matmul,
            "aten::baddbmm": self.baddbmm,
            "aten::expand": self.expand,
            "aten::Int": self.int,
            "prim::NumToTensor": self.numtotensor,
            "prim::ImplicitTensorToNum": self.tensortonum,
            "aten::ScalarImplicit": self.tensortonum,
            "aten::pad": self.pad,
            "aten::constant_pad_nd": self.constant_pad_nd,
            "aten::reflection_pad1d": self.reflection_pad1d,
            "aten::reflection_pad2d": self.reflection_pad2d,
            "aten::replication_pad1d": self.replication_pad1d,
            "aten::replication_pad2d": self.replication_pad2d,
            "aten::replication_pad3d": self.replication_pad3d,
            "aten::permute": self.transpose,
            "aten::sum": self.make_reduce("sum"),
            "aten::prod": self.make_reduce("prod"),
            "aten::argmin": self.make_reduce("argmin"),
            "aten::argmax": self.make_reduce("argmax"),
            "aten::norm": self.norm,
            "aten::frobenius_norm": self.frobenius_norm,
            "aten::std": self.std,
            "aten::var": self.variance,
            "aten::var_mean": self.var_mean,
            "aten::abs": self.make_unary("abs"),
            "aten::neg": self.make_unary("negative"),
            "aten::cos": self.make_unary("cos"),
            "aten::cosh": self.make_unary("cosh"),
            "aten::sin": self.make_unary("sin"),
            "aten::sinh": self.make_unary("sinh"),
            "aten::tan": self.make_unary("tan"),
            "aten::tanh": self.make_unary("tanh"),
            "aten::acos": self.make_unary("acos"),
            "aten::asin": self.make_unary("asin"),
            "aten::atan": self.make_unary("atan"),
            "aten::log": self.make_unary("log"),
            "aten::log2": self.make_unary("log2"),
            "aten::log10": self.make_unary("log10"),
            "aten::log1p": self.log1p,
            "aten::exp": self.make_unary("exp"),
            "aten::erf": self.make_unary("erf"),
            "aten::trunc": self.make_unary("trunc"),
            "aten::sign": self.make_unary("sign"),
            "aten::sqrt": self.make_unary("sqrt"),
            "aten::rsqrt": self.make_unary("rsqrt"),
            "aten::square": self.square,
            "aten::tril": functools.partial(self.trilu, mode="tril"),
            "aten::triu": functools.partial(self.trilu, mode="triu"),
            "aten::ceil": self.make_unary("ceil"),
            "aten::floor": self.make_unary("floor"),
            "aten::round": self.make_unary("round"),
            "aten::isfinite": self.make_unary("isfinite"),
            "aten::isinf": self.make_unary("isinf"),
            "aten::isnan": self.make_unary("isnan"),
            "aten::clamp": self.clamp,
            "aten::clamp_min": self.clamp_min,
            "aten::clamp_max": self.clamp_max,
            "aten::detach": self.identity,
            "aten::upsample_bilinear2d": self.make_upsample("linear"),
            "aten::upsample_bicubic2d": self.make_upsample("cubic"),
            "aten::upsample_nearest2d": self.make_upsample("nearest_neighbor"),
            "aten::upsample_trilinear3d": self.make_upsample3d("linear"),
            "aten::upsample_nearest3d": self.make_upsample3d("nearest_neighbor"),
            "aten::expand_as": self.expand_as,
            "aten::broadcast_tensors": self.broadcast_tensors,
            "aten::broadcast_to": self.broadcast_to,
            "aten::lt": self.make_elemwise("less"),
            "aten::gt": self.make_elemwise("greater"),
            "aten::le": self.make_elemwise("less_equal"),
            "aten::ge": self.make_elemwise("greater_equal"),
            "aten::ne": self.make_elemwise("not_equal"),
            "aten::eq": self.make_elemwise("equal"),
            "aten::logical_not": self.logical_not,
            "aten::logical_xor": self.logical_xor,
            "aten::bitwise_not": self.bitwise_not,
            "aten::bitwise_xor": self.bitwise_xor,
            "aten::Bool": self.Bool,
            "aten::Float": self.Float,
            "aten::rsub": self.rsub,
            "aten::embedding": self.embedding,
            "aten::embedding_bag": self.embedding_bag,
            "aten::one_hot": self.one_hot,
            "aten::mm": self.matmul,
            "aten::add": self.add,
            "aten::stack": self.stack,
            "aten::__getitem__": self.getitem,
            "aten::len": self.list_len,
            "aten::type_as": self.type_as,
            "aten::gather": self.gather,
            "aten::index_select": self.select,
            "aten::index": self.index,
            "torchvision::nms": self.nms,
            "aten::logsumexp": self.logsumexp,
            "torchvision::roi_align": self.roi_align,
            "torchvision::deform_conv2d": self.deform_conv2d,
            "aten::unbind": self.unbind,
            "aten::__and__": self.logical_and,
            "aten::logical_and": self.logical_and,
            "aten::_shape_as_tensor": self.shape_as_tensor,
            "aten::nonzero": self.nonzero,
            "aten::nonzero_numpy": self.nonzero_numpy,
            "aten::scatter": self.scatter,
            "aten::scatter_add": self.scatter_add,
            "aten::scatter_reduce": self.scatter_reduce,
            "aten::index_copy": self.index_copy,
            "aten::index_put": self.index_put,
            "aten::scalar_tensor": self.scalar_tensor,
            "aten::__interpolate": self.interpolate,
            "aten::IntImplicit": self.identity,
            "aten::tensor": self.identity,  # used for example in tensor(1.0)
            "aten::numel": self.numel,
            "aten::empty": self.empty,
            "aten::empty_like": self.empty_like,
            "aten::new_empty": self.new_empty,
            "aten::randn": self.randn,
            "aten::bincount": self.bincount,
            "aten::__not__": self.logical_not,
            "aten::hardswish": self.hard_swish,
            "aten::hardsigmoid": self.hard_sigmoid,
            "aten::cumsum": self.cumsum,
            "aten::masked_fill": self.masked_fill,
            "aten::masked_select": self.masked_select,
            "aten::masked_scatter": self.masked_scatter,
            "aten::argsort": self.argsort,
            "aten::sort": self.sort,
            "aten::_unique2": self.unique,
            "aten::l1_loss": self.l1_loss,
            "aten::mse_loss": self.mse_loss,
            "aten::nll_loss": self.nll_loss,
            "aten::nll_loss2d": self.nll_loss,
            "aten::nll_loss_nd": self.nll_loss,
            "aten::cross_entropy_loss": self.cross_entropy_loss_with_logits,
            "aten::l1_loss": self.l1_loss,
            "aten::mse_loss": self.mse_loss,
            "aten::flip": self.flip,
            "aten::rnn_tanh": functools.partial(self.rnn, nonlinearity="tanh"),
            "aten::rnn_relu": functools.partial(self.rnn, nonlinearity="relu"),
            "aten::gru": self.gru,
            "aten::lstm": self.lstm,
            "aten::all": functools.partial(self.all_any_common, _op.all),
            "aten::any": functools.partial(self.all_any_common, _op.any),
            "aten::searchsorted": self.searchsorted,
            "aten::bucketize": self.bucketize,
            "aten::roll": self.roll,
            "aten::einsum": self.einsum,
            "aten::dot": self.dot,
            "aten::mv": self.mv,
            "aten::grid_sampler": self.grid_sampler,
            "aten::__ior__": self.make_elemwise("bitwise_or"),
            "aten::bitwise_or_": self.make_elemwise("bitwise_or"),
            "aten::__iand__": self.make_elemwise("bitwise_and"),
            "aten::__ixor__": self.make_elemwise("bitwise_xor"),
            "aten::__lshift__": self.make_elemwise("left_shift"),
            "aten::__rshift__": self.make_elemwise("right_shift"),
            "aten::multinomial": self.multinomial,
            "aten::_weight_norm": self.weight_norm,
            "aten::copy_": self.identity,
            "aten::new_empty": self.new_empty,
            "aten::new_zeros": self.new_zeros,
            "aten::new_ones": self.new_ones,
            "aten::tril": self.tril,
            "aten::triu": self.triu,
            "aten::as_strided": self.as_strided,
            "aten::_weight_norm": self.weight_norm,
            "aten::new_full": self.new_full,
            "aten::dim": self.dim,
            "aten::__is__": self.is_impl,
            "aten::conv2d": self.conv2d,
            "aten::as_tensor": self.as_tensor,
            "aten::format": self.format,
            "aten::warn": self.warn,
            "aten::baddbmm": self.baddbmm,
            "aten::__or__": self.__or__,
            "aten::alias": self.alias,
            "aten::linalg_vector_norm": self.linalg_vector_norm,
            "aten::scaled_dot_product_attention": self.scaled_dot_product_attention,
            "aten::lift_fresh": self.identity,
            "aten::nan_to_num": self.nan_to_num,
            "aten::atan2": self.atan2,
            "aten::resolve_conj": self.resolve_conj,
            "aten::resolve_neg": self.resolve_neg,
            "aten::unflatten": self.unflatten,
            "aten::block_diag":self.block_diag,
        }

    def update_convert_map(self, custom_map):
        self.convert_map.update(custom_map)

    def report_missing_conversion(self, op_names):
        """Check if all ops in an input graph are supported by TVM"""
        known_ops = [
            "prim::Constant",
            "prim::GetAttr",
            "prim::ListConstruct",
            "prim::ListUnpack",
            "prim::TupleConstruct",
            "prim::TupleUnpack",
            "prim::RaiseException",
            "prim::If",
            "prim::Loop",
            "prim::DictConstruct",
            "prim::dtype",
            "prim::SetAttr",
            "prim::unchecked_cast"
        ]
        known_ops += list(self.convert_map.keys())
        known_ops += list(qnn_torch.convert_map.keys())

        missing = []

        for op_name in op_names:
            # Also take care of in-place variant ops like aten::relu_
            if op_name not in known_ops and not (
                op_name.endswith("_") and op_name[:-1] in known_ops
            ):
                missing.append(op_name)

        if missing:
            msg = f"The following operators are not implemented: {missing}"
            raise NotImplementedError(msg)

    def convert_block(self, block, outputs):
        """Translate Torch "Block", used for prim::If and prim::Loop"""
        ops = _get_operator_nodes(
            block.nodes(), self.source_map, self.op_type_dict, self.use_parser_friendly_name
        )
        ret_names = _get_input_names(block.returnNode(), self.input_remap)
        return self.convert_operators(ops, outputs, ret_names)

    def convert_if(self, if_node, outputs):
        """Translate Torch prim::If to Relay If"""
        cond = outputs[if_node.inputsAt(0).debugName()]
        blocks = list(if_node.blocks())
        true_branch = self.convert_block(blocks[0], outputs)
        false_branch = self.convert_block(blocks[1], outputs)
        assert len(true_branch) == 1 and len(false_branch) == 1

        ce = _CondEvaluator(self.types)
        new_cond = ce.visit(cond)
        if isinstance(new_cond, _expr.Constant):
            if np.all(new_cond.data.numpy()):
                return true_branch[0]
            else:
                return false_branch[0]
        return _expr.If(cond, true_branch[0], false_branch[0])

    def convert_loop(self, loop_node, outputs):
        """Translate Torch prim::Loop to Relay while_loop"""

        def get_input(index):
            ivalue = loop_node.inputsAt(index)
            inode = ivalue.node()
            if inode.kind() == "prim::Constant":
                return _expr.const(_get_constant(inode))
            var_name = ivalue.debugName()
            assert var_name in outputs
            return _wrap_const(outputs[var_name])

        # Refer to the spec for prim::Loop below
        # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#loops
        # The first input: %max_trip_count
        # The second input: %initial_condition
        # The rest of input: loop variables
        max_loop_count = get_input(0)
        init_cond = get_input(1)
        num_loop_var = len(list(loop_node.inputs())) - 2
        init_vals = [get_input(i + 2) for i in range(num_loop_var)]

        # while loop has always max_loop_count being int64 max
        # max_loop_count.data (tvm.runtime.NDArray) is -1, so _get_constant again
        is_while_loop = (
            isinstance(max_loop_count, _expr.Constant)
            and _get_constant(loop_node.inputsAt(0).node()) == sys.maxsize
        )

        if is_while_loop:
            loop_iter_dtype = "bool"
            # while loop with non input dependent condition such as while i < 10:
            # init_cond is int, need to cast to bool to type check
            if isinstance(init_cond, _expr.Constant):
                init_cond = _op.cast(init_cond, "bool")
            init_loop_iter_val = init_cond
        else:
            loop_iter_dtype = "int32"
            # always count from 0
            init_loop_iter_val = _expr.const(0, dtype="int32")

        body_block = list(loop_node.blocks())[0]
        block_input_names = _get_input_names(body_block, self.input_remap)
        num_block_inputs = len(block_input_names)
        name_val_pairs = list(zip(block_input_names, [init_loop_iter_val] + init_vals))
        outputs.update(name_val_pairs)

        def get_var(name, val):
            if val:
                checked_type = self.infer_type_with_prelude(val)
                if hasattr(checked_type, "shape"):
                    shape = get_const_tuple(checked_type.shape)
                    actual_shape = []
                    for dim in shape:
                        if isinstance(dim, int) and dim == 0:
                            actual_shape.append(Any())
                        else:
                            actual_shape.append(dim)
                    expr = _expr.var(name, shape=actual_shape, dtype=checked_type.dtype)
                else:
                    expr = _expr.var(name, type_annotation=checked_type)
                return set_span(expr, val.span) if val.span else expr
            return _expr.var(name)

        source_name = self.source_map[loop_node]
        loop_iter_var = set_span(
            _expr.var(block_input_names[0], shape=(), dtype=loop_iter_dtype), span=source_name
        )
        loop_vars = set_span(
            [get_var(name, val) for name, val in name_val_pairs[1:]], span=source_name
        )

        # Add non constant free variables to loop variables to prevent code blow up
        # Without this, if there are two for loops in a row, which often happens
        # if the outer loop is unrolled, the computation corresponding to the first for loop
        # is inlined inside loop body, turning O(N) + O(N) computation into O(N^2).
        # This issue was found when converting from Stacked LSTM test. Torch does not add the
        # outputof the eariler loop into loop variables of the next loop.
        # So the variable corresponding to the first loop output appears free in the second
        # loop body.
        free_vars = [
            var
            for var in _get_free_vars_from_block(body_block, self.input_remap)
            if var in outputs
            and not isinstance(outputs[var], (_expr.Constant, int, float, str))
            and outputs[var]
        ]

        prev_outputs = {}
        for name in free_vars:
            prev_output = outputs[name]
            new_loop_var = get_var(name, prev_output)
            prev_outputs[name] = prev_output
            outputs[name] = set_span(new_loop_var, source_name)
            loop_vars.append(new_loop_var)
            init_vals.append(prev_output)

        def cond(*current_vals):
            i = current_vals[0]

            if is_while_loop:
                return _op.equal(i, _expr.const(True, "bool"))

            return _op.less(i, max_loop_count)

        def body(*current_vals):
            # Update loop variables using the prev iteration outputs
            assert len(current_vals) == num_block_inputs + len(free_vars)

            for (i, val) in enumerate(current_vals):
                if i < num_block_inputs:
                    outputs[block_input_names[i]] = val
                else:
                    outputs[free_vars[i - num_block_inputs]] = val

            block_outputs = self.convert_block(body_block, outputs)
            block_outputs += [outputs[name] for name in free_vars]

            if not is_while_loop:
                # iter var increment implicit in torch, so do it manually
                # for while loop, block_outputs[0] is already a boolean,
                # the result of termination check
                incr = _expr.const(1, dtype="int32")
                block_outputs[0] = current_vals[0] + incr

            return block_outputs

        loop = while_loop(cond, [loop_iter_var] + loop_vars, body)
        loop_val = loop(init_loop_iter_val, *init_vals)

        # restore original output values for free vars
        outputs.update(prev_outputs)

        # The first element is a loop counter or boolean condition, ignore it
        return [_expr.TupleGetItem(loop_val, i + 1) for i in range(num_loop_var)]

    def convert_operators(self, operators, outputs, ret_names, input_remap={}):
        """Convert each Torch IR operators to Relay equivalent"""
        # an op node might not belong to any of scope in trace info natively
        # use a cunter to prevent from messing up its scope in span
        empty_counter = 0
        self.input_remap = input_remap
        for node_name, op_node in operators:

            logger.trace(f"Converting: {op_node.kind()} : {node_name}")
            operator = op_node.kind()
            inputs = _get_op_inputs(op_node, outputs, self.input_remap)
            # we need to record what current operator is to provide correct source name
            # for operators needed to be taken care with (e.g. nms / arange ...)
            self.current_op.append(op_node)

            if operator == "prim::Constant":
                outputs[node_name] = _get_constant(op_node)
            elif operator == "prim::ListConstruct" and _should_construct_dynamic_list(op_node):
                outputs[node_name] = set_span(
                    self.convert_to_list_adt(inputs), self.source_map[op_node]
                )
            elif operator == "prim::ListConstruct":
                # This assumes that no more elements will be appended to this list
                # In this case, we keep the Python list
                outputs[node_name] = inputs
            elif operator == "prim::TupleConstruct":

                def _handel_nested_input(inputs):
                    inputs_list = []
                    for i, _ in enumerate(inputs):
                        if isinstance(inputs[i], list):
                            inputs_list.append(_handel_nested_input(inputs[i]))
                        else:
                            assert isinstance(inputs[i], _expr.Expr)
                            inputs_list.append(inputs[i])
                    return _expr.Tuple(inputs_list)

                outputs[node_name] = set_span(
                    _handel_nested_input(inputs), self.source_map[op_node]
                )
            elif operator in ["prim::ListUnpack", "prim::TupleUnpack"]:
                # if isinstance(inputs[0], _expr.Call):
                #     inputs[0] = unbind(inputs[0], 1)

                assert len(inputs) == 1
                if isinstance(inputs[0], (list, _expr.TupleWrapper)):
                    unpacked = inputs[0]
                else:
                    unpacked = _unpack_tuple(inputs[0])
                outputs.update(
                    zip(_get_output_names(op_node), set_span(unpacked, self.source_map[op_node]))
                )
            elif operator == "prim::prim::RaiseException":
                logger.warning("raising exceptions is ignored")
                outputs[node_name] = None
            elif operator == "prim::If":
                if_out = self.convert_if(op_node, outputs)
                outputs[node_name] = set_span(if_out, self.source_map[op_node])
            elif operator == "prim::Loop":
                loop_out = self.convert_loop(op_node, outputs)
                unpacked_names = _get_output_names(op_node)
                assert len(loop_out) == len(unpacked_names)
                outputs.update(zip(unpacked_names, set_span(loop_out, self.source_map[op_node])))

            elif operator == "prim::DictConstruct":
                assert len(inputs) % 2 == 0, "Expected even number of inputs to unpack"
                # Just turn into tuple for now
                return _expr.Tuple(inputs[1::2])
            elif operator == "prim::unchecked_cast":
                outputs[node_name] = inputs[0]
            elif operator == "prim::dtype":
                tf = _TypeFinder(self.types)
                typ = tf.visit(inputs[0])
                outputs[node_name] = typ.type_annotation.dtype
            else:
                if operator not in self.convert_map:
                    # At this point, the only possible ops that are not in convert_map are
                    # in-place variant of ops like aten::relu_
                    assert operator.endswith("_"), f"Found operator \'{operator}\'"
                    op_base = operator[:-1]
                    logger.warning(
                        f"In-place operator '{operator}' not found in convert_map. "
                        f"Falling back to out-of-place equivalent '{op_base}'. "
                        f"This may cause incorrect behavior if the model relies on side-effects of in-place mutation."
                    )
                    relay_op = self.convert_map[operator[:-1]]
                else:
                    relay_op = self.convert_map[operator]

                self._set_parameter_source_name(op_node, outputs, self.input_remap)
                relay_out = relay_op(
                    # since the elements in "outputs" may change due to span-filling process
                    # we have to call "_get_op_inputs" again rather than use "inputs" directly
                    _get_op_inputs(op_node, outputs, self.input_remap),
                    _get_input_types(op_node, outputs, self.input_remap, default_dtype=self.default_dtype),
                )
                span_str, empty_counter = self._get_torch_span(op_node, empty_counter)
                relay_out = set_span(relay_out, span_str)

                self.record_output_type(relay_out)

                if isinstance(relay_out, tuple):
                    # This is for torch operators that return multiple outputs
                    # See _adaptive_max_2d above for example
                    out_names = _get_output_names(op_node)
                    outputs.update(zip(out_names, relay_out))
                else:
                    assert op_node.outputsSize() == 1
                    outputs[node_name] = relay_out

            self.current_op.pop()

        # TODO(@haoyang9804): outputs[ret_name] could be None and cause some issue
        # revealed by https://github.com/apache/tvm/issues/15004
        # Now only adaptive_max_pool1d is considered. Maybe other ops could also
        # trigger this problem.
        return [
            _wrap_const(outputs[ret_name])
            for ret_name in ret_names
            if ret_name != "aten::adaptive_max_pool1d_0_1"
        ]

    def _set_parameter_source_name(self, op_node, outputs, input_remap):
        """A helper function to rewrite source_name of parameter."""
        for name in _get_input_names(op_node, input_remap):
            expr = outputs[name]
            if isinstance(expr, (_expr.Var, _expr.Constant)):
                name_sep = "_" if self.use_parser_friendly_name else "."
                source_name = [self.source_map[op_node]]
                if isinstance(expr, _expr.Var):
                    # variable name should have contained node source name
                    # for op with attributes in convert_params stage
                    # e.g. "aten::batch_norm_5.running_mean"
                    if expr.name_hint.startswith(source_name[0]):
                        source_name[0] = expr.name_hint
                    else:
                        source_name.append(expr.name_hint)
                new_expr = set_span(expr, name_sep.join(source_name))
                outputs[name] = new_expr

    def _get_torch_span(self, node, empty_counter):
        # torch span looks like
        # %input.5 : Float(...) = aten::relu_(%input.3), scope: __module.relu # ${torch}/nn file
        # the scope part might not exist
        if node.scopeName():
            scope_name_str = "jit._trace.TopLevelTracedModule: " + node.scopeName()
        else:
            scope_name_str = "warning: no trace info " + str(empty_counter)
            empty_counter += 1
        span_str = "C.graph: {}, {}".format(node.kind(), scope_name_str)
        return span_str, empty_counter


def _pytorch_result_type(dtypes, non_tensor_inputs):
    """This promotes TVM dtypes like PyTorch would"""
    import torch

    dtype_map = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if len(dtypes) > 0:
        result_type = dtypes[0]
        for dt in dtypes[1:]:
            if dt != result_type:  # we don't want to work with same types as we
                # don't do quantized here (which cannot be promoted?)
                result_type = _convert_data_type(
                    str(
                        torch.result_type(
                            torch.zeros((), dtype=dtype_map[result_type]),
                            torch.zeros((), dtype=dtype_map[dt]),
                        )
                    )
                )
    else:
        result_type = "bool"  # this is the smallest type...
    for inp in non_tensor_inputs:
        result_type = _convert_data_type(
            str(torch.result_type(torch.zeros((), dtype=dtype_map[result_type]), inp))
        )
    return result_type

def _convert_tvm_to_np_dtype(dtype):
    if dtype == "float64":
        np_type = np.float64
    elif dtype == "float32":
        np_type = np.float32
    elif dtype == "float16":
        np_type = np.float16
    elif dtype == "int64":
        np_type = np.int64
    elif dtype == "int32":
        np_type = np.int32
    elif dtype == "int16":
        np_type = np.int16
    elif dtype == "int8":
        np_type = np.int8
    elif dtype == "uint8":
        np_type = np.uint8
    elif dtype == "bool":
        np_type = bool
    else:
        raise NotImplementedError("input_type {} is not handled yet".format(dtype))
    return np_type


# Helper functions for operator implementation
def _convert_dtype_value(val):
    """converts a PyTorch the PyTorch numeric type id to a torch scalar type."""
    convert_torch_dtype_map = {
        11: "torch.bool",
        7: "torch.float64",
        6: "torch.float32",
        5: "torch.float16",
        4: "torch.int64",
        3: "torch.int32",
        2: "torch.int16",
        1: "torch.int8",
        0: "torch.uint8",
        None: "torch.int64",
    }  # Default is torch.int64
    if val in convert_torch_dtype_map:
        return _convert_data_type(convert_torch_dtype_map[val])
    else:
        msg = f"Torch data type value {val} is not handled yet."
        raise NotImplementedError(msg)


def _convert_data_type(input_type, default_dtype=None):
    """converts the PyTorch scalar type input_type to a TVM dtype.
    optionally, default_dtype can be a TVM dtype that is used
    if input_type is None (but not when it is unknown)"""
    if input_type is None and default_dtype is not None:
        return default_dtype

    input_type = input_type.lower()
    if input_type in ["double", "float64", "torch.float64"]:
        return "float64"
    elif input_type in ["float", "float32", "torch.float32"]:
        return "float32"
    elif input_type in ["torch.bfloat16", "bfloat16"]:
        return "bfloat16"
    elif input_type in ["half", "float16", "torch.float16"]:
        return "float16"
    elif input_type in ["long", "int64", "torch.int64"]:
        return "int64"
    elif input_type in ["int", "int32", "torch.int32"]:
        return "int32"
    elif input_type in ["short", "int16", "torch.int16"]:
        return "int16"
    elif input_type in ["char", "int8", "torch.int8"]:
        return "int8"
    elif input_type in ["byte", "uint8", "torch.uint8"]:
        return "uint8"
    elif input_type in ["quint8", "torch.quint8"]:
        return "quint8"
    elif input_type in ["qint8", "torch.qint8"]:
        return "qint8"
    elif input_type in ["qint32", "torch.qint32"]:
        return "qint32"
    elif input_type in ["bool", "torch.bool"]:
        return "bool"
    elif input_type in ["str"]:
        return "str"
    else:
        raise NotImplementedError(f"input_type {input_type} is not handled yet")
    return "float32"  # Never reached


def _create_typed_const(data, dtype):
    """create a (scalar) constant of given value and dtype.
    dtype should be a TVM dtype"""

    if dtype == "float64":
        typed_data = _expr.const(np.asarray(data, dtype="float64"), dtype=dtype)
    elif dtype == "float32":
        typed_data = _expr.const(np.float32(data), dtype=dtype)
    elif dtype == "float16":
        typed_data = _expr.const(np.float16(data), dtype=dtype)
    elif dtype == "int64":
        typed_data = _expr.const(np.int64(data), dtype=dtype)
    elif dtype == "int32":
        typed_data = _expr.const(np.int32(data), dtype=dtype)
    elif dtype == "int16":
        typed_data = _expr.const(np.int16(data), dtype=dtype)
    elif dtype == "int8":
        typed_data = _expr.const(np.int8(data), dtype=dtype)
    elif dtype == "uint8":
        typed_data = _expr.const(np.uint8(data), dtype=dtype)
    elif dtype == "bool":
        typed_data = _expr.const(np.array(data, dtype=dtype), dtype=dtype)
    else:
        raise NotImplementedError(f"input_type {dtype} is not handled yet")
    return typed_data


def _wrap_const(c):
    if not isinstance(c, (_expr.Expr, list, tvm.tir.expr.Any)):
        return _expr.const(c)
    return c


def _run_jit_passes(graph, enable_lower_all_tuples=True):
    """The inline pass is necessary to unwrap prim::CallMethod"""
    # pylint: disable=c-extension-no-member
    import torch

    if is_version_greater_than("1.5.1"):
        # This is required for torchvision detection models from 1.6 above
        # It is the same as _jit_pass_inline, except that it has some special
        # case behaviors for some ops such as aten::__interpolate()
        torch._C._jit_pass_onnx_function_substitution(graph)
    else:
        torch._C._jit_pass_inline(graph)

    if enable_lower_all_tuples:
        torch._C._jit_pass_lower_all_tuples(graph)


def _get_tensor_and_var(torch_tensor, name, do_convert_params, id):
    if do_convert_params:
        orig_dtype = str(torch_tensor.dtype).replace("torch.", "")
        if orig_dtype == 'bfloat16':
            torch_tensor = torch_tensor.detach().float()
        tensor = tvm.nd.array(torch_tensor.cpu().numpy())
        var = _expr.var(name, shape=tensor.shape, dtype=tensor.dtype, framework_dtype=orig_dtype, id=id)
    else:
        orig_dtype = str(torch_tensor.dtype).replace("torch.", "")
        shape = tuple(torch_tensor.shape)
        dtype = _convert_data_type(str(torch_tensor.dtype))
        tensor = None
        var = _expr.var(name, shape=shape, dtype=dtype, framework_dtype=orig_dtype, id=id)
    return tensor, var


def _get_output_name(node):
    if node.outputsSize() != 1:
        return node.inputsAt(0).debugName()

    return node.output().debugName()

def _get_output_names(node):
    return [output.debugName() for output in node.outputs()]


def _get_input_names(node_or_graph, input_remap):
    input_names = [inp.debugName() for inp in node_or_graph.inputs()]
    input_names = [input_remap[name] if name in input_remap else name for name in input_names]
    return input_names


def _get_op_inputs(op_node, outputs, input_remap={}):
    return [outputs[name] for name in _get_input_names(op_node, input_remap)]


def _get_node_type(node):
    assert node.outputsSize() == 1
    return node.output().type().kind()


def _get_uses(node):
    uses = []
    for output in node.outputs():
        uses += output.uses()
    return uses


def _get_users(node):
    return [use.user for use in _get_uses(node)]


def _getattr_full_name(getattrs, sep="."):
    return sep.join([getattr_attr_name(node) for node in getattrs])


def _get_pytorch_value_type(typ, default_dtype="float32"):
    kind = typ.kind()
    if kind == "TensorType":
        if typ.scalarType() is None:
            # Tensor's type can be unknown if we use torch.jit.script(...)
            # Defaults can be passed in, if not it is float32
            logger.warning(f"Untyped Tensor found, assume it is {default_dtype}")
            return default_dtype
        else:
            return _convert_data_type(typ.scalarType())

    elif kind == "ListType":
        return "ListType"
    elif kind in ["IntType", "FloatType", "BoolType", "StringType", "OptionalType"]:
        pt_dtype = str(typ).lower()
        dtype = pt_dtype if kind == "OptionalType" else _convert_data_type(pt_dtype)
        return dtype
    else:
        return "UnsupportedType"


def _get_input_types(op_node, outputs, input_remap, default_dtype="float32"):
    """Returns a TVM dtype for each input nodes derived from the torch type"""
    in_types = []
    for inp in op_node.inputs():
        if inp.node().kind() == "prim::GetAttr":
            # GetAttr nodes always return None when we call scalarType() on it
            name = inp.debugName()
            name = input_remap[name] if name in input_remap else name

            assert name in outputs, f"\'{name}\' is not a key in outputs"
            if isinstance(outputs[name], _expr.Var):
                in_types.append(outputs[name].type_annotation.dtype)
            else:
                # For quantized modules with parameters, here we would get
                # "prim::GetAttr[name="_packed_params"]". Since the dtype corresponding to
                # _packed_params is not needed by quantized ops, we return an arbitrary type.
                in_types.append(default_dtype)
        else:
            in_types.append(_get_pytorch_value_type(inp.type(), default_dtype=default_dtype))

    return in_types


def _get_constant(node):
    """Retrieve a constant associated with this prim::Constant node"""
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)

    if num_attributes == 1:
        attr_name = attribute_names[0]
        ty = node.output().type().kind()

        if ty == "IntType":
            return node.i(attr_name)
        elif ty == "BoolType":
            return bool(node.i(attr_name))
        elif ty in ["FloatType", "LongType"]:
            return node.f(attr_name)
        elif ty in ["TensorType", "CompleteTensorType"]:
            tensor = node.t(attr_name)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            if len(tensor.shape) == 0:  # tensor(0.1)
                # TODO(t-vi): When is this needed?
                return tensor.item()
            # Convert bfloat16 to float32 before numpy conversion since numpy doesn't support bfloat16
            if hasattr(tensor, 'dtype') and str(tensor.dtype) == 'torch.bfloat16':
                tensor = tensor.float()
            return _wrap_const(tensor.numpy())
        elif ty in ["DeviceObjType", "StringType"]:
            return node.s(attr_name)
        elif ty == "FunctionType":
            return None
        elif ty == "ListType":
            lst = node.outputsAt(0).toIValue()
            for i in range(len(lst)):
                lst[i] = _wrap_const(lst[i])
            return _expr.Tuple(lst)
        else:
            raise NotImplementedError(f"Unsupported type: {ty}")
    else:
        assert num_attributes == 0
        return None


def _rename_outputs(node, source_map, op_type_dict, use_parser_friendly_name):
    """Rewrite debug name of node outputs with its operator type"""

    def _get_source_name(op_type):
        op_idx = 0
        if op_type in op_type_dict:
            op_idx = op_type_dict[op_type] + 1
        op_type_dict[op_type] = op_idx
        return "_".join([op_type, str(op_idx)])

    # get source name of operator and rename all of its outputs
    # e.g. node.kind(): aten::adaptive_max_pool2d
    # node_src_name -> aten::adaptive_max_pool2d_x
    # output_1 -> aten::adaptive_max_pool2d_x_0
    # output_2 -> aten::adaptive_max_pool2d_x_1
    if node.kind() != "prim::GetAttr":
        node_src_name = _get_source_name(node.kind())
        for index, output in enumerate(node.outputs()):
            output.setDebugName("_".join([node_src_name, str(index)]))
        # update source map
        # if use_parser_friendly_name is True: e.g. prim::Constant_0 -> prim__Constant_0
        if use_parser_friendly_name:
            node_src_name = re.sub(r":|\.", "_", node_src_name)
        source_map[node] = node_src_name


def _debug_rename(graph, use_parser_friendly_name):
    """Returns map between node and source name"""
    source_map, op_type_dict = {}, {}
    prim_with_blocks = ["prim::If", "prim::Loop"]

    def _traverse_graph(nodes):
        for node in nodes:
            if node.outputsSize() == 0:
                continue
            if node.kind() in prim_with_blocks:
                for block in node.blocks():
                    _traverse_graph(block.nodes())
            _rename_outputs(node, source_map, op_type_dict, use_parser_friendly_name)

    _traverse_graph(graph.nodes())
    return source_map


def _get_operator_nodes(nodes, source_map=None, op_type_dict=None, use_parser_friendly_name=False):
    """Returns torch IR nodes that need conversion to Relay"""
    ops, should_rename_graph = [], all([source_map, op_type_dict]) is not None

    # Traverse nodes and add to graph
    for node in nodes:
        if node.outputsSize() == 0:
            continue

        if should_rename_graph:
            _rename_outputs(node, source_map, op_type_dict, use_parser_friendly_name)

        if node.outputsSize() > 1:
            node_name = "_".join(_get_output_names(node))
        else:
            node_name = _get_output_name(node)

        if node.kind() not in ["prim::GetAttr", "prim::SetAttr"]:
            ops.append((node_name, node))

    return ops


def _get_relay_input_vars(graph, input_infos, prelude, is_module=True, default_dtype="float32"):
    """
    Return Relay vars from input shapes and create entries based on
    expected graph inputs - to allow translation
    """
    graph_inputs = list(graph.inputs())
    if is_module:
        # a module has "self" as first input, which we do not need/want
        graph_inputs = graph_inputs[1:]

    if not isinstance(input_infos, list):
        msg = "Graph inputs input_infos should be a list"
        raise RuntimeError(msg)

    if len(graph_inputs) != len(input_infos):
        msg = f"PyTorch has {len(graph_inputs)} inputs and input_infos lists {len(input_infos)}."
        raise RuntimeError(msg)

    def get_relay_ty(ishape, itype, pt_type):
        if pt_type.kind() == "TensorType":
            if not (_is_int_seq(ishape) or len(ishape) == 0):
                msg = "Shape for Tensors must be lists of ints"
                raise RuntimeError(msg)
            if (pt_type.dim() is not None and pt_type.dim() != len(ishape)) or (
                pt_type.sizes() is not None
                and any([s1 != s2 for s1, s2 in zip(pt_type.sizes(), ishape)])
            ):
                msg = "Shapes of input list and information in the graph do not match"
                raise RuntimeError(msg)
            if len(ishape) > 1 and any(dim <= 0 for dim in ishape[1:]):
                msg = (
                    "Expected input's non-batch dimensions to have positive length, "
                    f"but input has a shape of {pt_type.sizes()}"
                )
                raise RuntimeError(msg)
            pt_dtype = pt_type.scalarType()
            if not pt_dtype and itype:
                pt_dtype = itype
            dtype = _convert_data_type(pt_dtype, default_dtype=default_dtype)
            return TensorType(ishape, dtype)
        elif pt_type.kind() == "TupleType":
            if not isinstance(ishape, tuple):
                msg = "Shapes for tuples must be tuples"
                raise RuntimeError(msg)
            return TupleType(
                [get_relay_ty(elem, itype, pt_t) for elem, pt_t in zip(ishape, pt_type.elements())]
            )
        elif pt_type.kind() == "ListType":
            if not isinstance(ishape, list):
                msg = "Shapes for lists must be lists"
                raise RuntimeError(msg)
            pt_elemtype = pt_type.getElementType()
            elem_tys = [get_relay_ty(s, itype, pt_elemtype) for s in ishape]
            if len(elem_tys) > 0 and not all(map(lambda ty: ty == elem_tys[0], elem_tys)):
                msg = "List elements need have identical types"
                raise RuntimeError(msg)
            rlist, _, _ = prelude.mod.get_type("List")
            return rlist(elem_tys[0])
        elif pt_type.kind() == "OptionalType":
            # we do not support None yet, so we fill in the type
            return get_relay_ty(ishape, itype, pt_type.getElementType())
        # TODO: scalar inputs
        raise NotImplementedError("unsupported input type")

    input_vars = {}


    def get_new_input_infos(input_infos):
        new_input_infos = []
        for num, inp in enumerate(input_infos):

            if not isinstance(inp, tuple):
                msg = "Graph input {} is not a tuple".format(num)
                raise RuntimeError(msg)
            if len(inp) != 2 or not isinstance(inp[0], str):
                msg = (
                    "Graph input {} is not valid,"
                    " expected ('name', shape) or ('name', (shape, dtype))".format(inp)
                )

                raise RuntimeError(msg)
            if isinstance(inp[1], (list, tuple)) and isinstance(inp[1][0], (list, tuple)) and isinstance(inp[1][0][0], str):
                new_input_infos.append((inp[0], get_new_input_infos(inp[1])))
            elif not isinstance(inp[1], tuple) or len(inp[1]) == 0 or not isinstance(inp[1][-1], str):
                new_input_infos.append((inp[0], (inp[1], default_dtype)))
            else:
                new_input_infos.append(inp)
        return new_input_infos

    new_input_infos = get_new_input_infos(input_infos)

    def get_input_types(input_infos, graph_input_types):
        input_types = []
        for (name, info), gi_type in zip(input_infos, graph_input_types):
            if isinstance(info[0], (list, tuple)) and isinstance(info[0][0], str):
                new_graph_input_types = gi_type.elements()
                input_types.append((name, get_input_types(info, new_graph_input_types)))
            else:
                input_types.append((name, get_relay_ty(info[0], info[1], gi_type), info[1])) # info[1] is the framework datatype, which may differ after being converted to relay
        return input_types


    graph_input_types = [gi.type() for gi in graph_inputs]
    input_types = get_input_types(new_input_infos, graph_input_types)

    def get_input_vars(input_types, graph_input_names, use_tuple_type=False, tuple_name=""):
        input_vars = {} if not use_tuple_type else []
        for gi_name, gi_type in zip(graph_input_names, input_types):
            name, itype = gi_type[0], gi_type[1]
            framework_dtype = gi_type[2] if len(gi_type) == 3 else None
            if isinstance(itype, (tuple, list)):
                new_graph_input_names = [f"{gi_name}_{i}" for i in range(len(itype))]
                if use_tuple_type:
                    input_vars.append(get_input_vars(itype, new_graph_input_names, True, name))
                else:
                    input_vars[gi_name] = get_input_vars(itype, new_graph_input_names, True, name)
            else:
                assert framework_dtype is not None, "framework_dtype must be specified"
                if use_tuple_type:
                    input_vars.append(_expr.var(name, type_annotation=itype, framework_dtype=framework_dtype))
                else:
                    input_vars[gi_name] = _expr.var(name, type_annotation=itype, framework_dtype=framework_dtype)
        if use_tuple_type:
            input_vars = _expr.Tuple(input_vars)
        return input_vars

    graph_input_names = [i.debugName() for i in graph_inputs]
    input_vars = get_input_vars(input_types, graph_input_names)

    return input_vars


def _unpack_tuple(tup):
    def unpack(tup, num_fields):
        unpacked = []
        for i in range(num_fields):
            field = tup.fields[i]
            if isinstance(field, _expr.Tuple):
                unpacked.append(field)
            else:
                unpacked.append(_expr.TupleGetItem(tup, i))
        return unpacked

    if isinstance(tup, _expr.Tuple):
        return unpack(tup, len(tup.fields))
    elif isinstance(tup.type_annotation, TupleType):
        return unpack(tup, len(tup.type_annotation.fields))
    # shouldn't happen
    assert False


def _get_free_vars_from_block(block, input_remap):
    block_inp_names = _get_input_names(block, input_remap)
    bound_names = block_inp_names
    free_vars = set()

    for node in block.nodes():
        inp_names = _get_input_names(node, input_remap)
        list_diff = [name for name in inp_names if name not in bound_names]
        free_vars.update(list_diff)
        bound_names += _get_output_names(node)

    return free_vars


def get_use_chains(root_node, terminate=lambda _: False):
    """
    Track a chain of users of this node forward, returning a list of chains
    See get_attr_chains below for its usage
    """
    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = _get_users(current)

        if not users or terminate(users):
            return [accum]

        return concat_lists([inner(nxt, accum + [nxt]) for nxt in users])

    return inner(root_node, [root_node])


def get_attr_chains(root_getattr_node):
    """Returns chains of attribute access starting from root_getattr_node

    For example, given attribute "block", as in "self.block" when "self" points
    to the top level torch.nn.Module, it returns lists of attribute "chains",
    e.g. ['block', '2'], ['block', '1'], ['block', '0', '_packed_params']

    These sets of attributes form full attribute accessors. For example,
    "self.block.1", "self.block.2" will return the second and third submodule,
    and "self.block.0._packed_params" will return the parameters of the first
    submodule.
    """

    def terminate(users):
        next_attrs = [user for user in users if (user.kind() in ["prim::GetAttr", "prim::SetAttr"])]
        return len(next_attrs) == 0

    return get_use_chains(root_getattr_node, terminate)


def convert_params(graph, state_dict, source_map, use_parser_friendly_name=False, do_convert_params=True):
    """
    Return Relay vars and TVM NDArrays for input parameters
    A chain of prim::GetAttr nodes is processed one at a time
    """
    getattr_nodes = list(graph.findAllNodes("prim::GetAttr", recurse=True)) + list(graph.findAllNodes("prim::SetAttr", recurse=True))
    params = {}
    param_tensors = {}
    packed_param_map = {}
    param_debug_name_map = {}
    vars_by_name = {}
    seen = set()
    attr_name_sep = "_" if use_parser_friendly_name else "."
    param_to_id = {}
    next_param_id = 1
    input_remap = {}
    outputs_by_var_name = {}
    for node in getattr_nodes:
        if _get_output_name(node) in seen:
            continue

        for getattrs in get_attr_chains(node):
            seen.update(map(_get_output_name, getattrs))

            full_attr = _getattr_full_name(getattrs, attr_name_sep)
            full_attr_node_name = _get_output_name(getattrs[-1])
            # set variable name by concatenating first consumer's name with full attribute
            # e.g. "aten::batch_norm_5.running_mean"

            # We don't want to do this, we want to use the param name, since we don't want
            # duplicated vars for a param, so we use "full_attr" which is the framework name
            # var_name = attr_name_sep.join(
            #     [source_map[_get_users(getattrs[-1])[0]], full_attr.split(attr_name_sep)[-1]]
            # )
            var_name = full_attr
            if full_attr.endswith("_packed_params"):  # for quantized models
                packed_param_map[full_attr_node_name] = full_attr
            elif full_attr in state_dict:
                if var_name in vars_by_name:
                    var = vars_by_name[var_name]
                    # we need to remap inputs that pointed to the old
                    input_remap[full_attr_node_name] = outputs_by_var_name[var_name]
                else:
                    torch_tensor = state_dict[full_attr]
                    if torch_tensor in param_to_id:
                        id = param_to_id[torch_tensor]
                    else:
                        id = next_param_id
                        param_to_id[torch_tensor] = id
                        next_param_id += 1
                    tensor, var = _get_tensor_and_var(torch_tensor, var_name, do_convert_params, id)
                    param_tensors[var_name] = tensor
                    # for quantized parameters to be correctly located
                    param_debug_name_map[full_attr_node_name] = var_name
                    vars_by_name[var_name] = var
                    params[full_attr_node_name] = var
                    outputs_by_var_name[var_name] = full_attr_node_name
    return params, param_tensors, packed_param_map, param_debug_name_map, input_remap


def get_all_op_names(graph):
    """Return all operator names in the input graph"""
    nodes = list(graph.nodes())
    prim_with_blocks = ["prim::If", "prim::Loop"]
    for prim in prim_with_blocks:
        prim_nodes = graph.findAllNodes(prim, recurse=True)
        for prim_node in prim_nodes:
            for block in prim_node.blocks():
                nodes += block.nodes()
    return set(node.kind() for node in nodes)


def export_c_graph(location, graph):
    """Convert the graph to an onnx model and export it to the location."""
    import datetime
    import os

    if not os.path.exists(location):
        os.makedirs(location)
    time_stamp = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    fname = os.path.join(location, f"tvm_exported_c_graph_{time_stamp}.txt")
    with open(f"{fname}", "w") as f:
        f.write(str(graph))

def _binary_search(lst, condition):
    """Binary search for the first index that `condition` returns True"""
    l, r = 0, len(lst)
    while l < r:
        m = (l + r) // 2
        if condition(lst[m]):
            r = m
        else:
            l = m + 1
    return l


def outplace_inplace_ops(opnodes):

    replace_map = []

    # Find in-place ops and record their outputs
    for i, (node_name, op_node) in enumerate(opnodes):
        operator = op_node.kind()
        # Check if op is in-place    (avoid '__not__', etc.)
        if operator[-1] == '_' and operator[-2:] != "__":
            input_node = op_node.inputsAt(0)
            replace_map.append((i, input_node, op_node.outputsAt(0)))

    # Replace future uses of node with an in-place op applied to it with the output of the op
    node_inputs_map = {}
    for idx, (node_name, op_node) in enumerate(opnodes):
        for inp in op_node.inputs():
            if inp not in node_inputs_map:
                node_inputs_map[inp] = []
            node_inputs_map[inp].append((idx, op_node))

    for node_idx, orig_node, replacement_node in replace_map:
        if orig_node not in node_inputs_map:
            continue

        relevant_ops = node_inputs_map[orig_node]
        begin_idx = _binary_search(relevant_ops, lambda x: x[0] > node_idx)
        for _, node in relevant_ops[begin_idx:]:
            node.replaceInputWith(orig_node, replacement_node)

    del node_inputs_map

def from_pytorch(
    script_module,
    input_infos,
    custom_convert_map=None,
    default_dtype="float32",
    use_parser_friendly_name=False,
    keep_quantized_weight=False,
    export_renamed_c_graph_path=None,
    do_convert_params=True,
):
    """Load PyTorch model in the form of a scripted PyTorch model and convert into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    script_module : TopLevelTracedModule object
        TorchScripted PyTorch graph
        Note: We currently only support traces (ie: torch.jit.trace(model, input))

    input_infos : List of tuples
        Can be (input name, input shape) or (input name, (input shape, input types))
        Graph level input shape and type list
        The same input names need to be used for deployment, so choose easy to
        remember names (such as: input0, input1)
        e.g.
        [('input0', (1, 2)), ('input1', (3, 4))]
        or
        [('input0', ((1, 2), 'int')), ('input1', ((3, 4), 'float'))]

    custom_convert_map : Dictionary of str to Relay op
        A custom op conversion map in the same format as _convert_map above

    default_type : str
        The default dtype to use when type information is not provided by PyTorch.

    use_parser_friendly_name : bool
        When True, replace '.' with `_' in a original parameter name.
        The Relay text parser treats a variable name followed by a period as a tuple element access,
        so a variable name like "dense.weight" cannot be parsed correctly.
        Use this option when you want to run the AnnotateSpans pass on the imported module.

    keep_quantized_weight : bool
        Return quantized weights and bias, rather than float ones. PyTorch stores quantized weights
        in a custom format, so we cannot directly access 8 bit weights as Numpy arrays. We use
        a PyTorch function to unpack quantized weights into float32 arrays and quantization
        parameters. By default, we return float32 weights and rely on the QNN lowering and the
        Relay constant folding pass to quantize weights at compile time. In BYOC use cases, however,
        we cannot apply the constant folding pass on a QNN graph. If keep_quantized_weight is True,
        we quantize weights in the frontend using a function that is equivalent to
        qnn.op.quantize(...) operating on Numpy arrays.

    export_renamed_c_graph_path : str, optional
        Export the renamed torch._C.Graph to the path.
        During the conversion, variable names in torch._C.Graph will be assigned based on their op
        types. The exported text file can be the reference to spans.

    Returns
    -------
    mod : tvm.IRModule
        The module that optimizations will be performed on.

    params : dict of str to tvm.runtime.NDArray
        Dict of converted parameters stored in tvm.runtime.ndarray format
    """
    import torch

    mod = tvm.IRModule()
    prelude = Prelude(mod)
    enable_lower_all_tuples = True

    converter = PyTorchOpConverter(prelude, default_dtype, use_parser_friendly_name)

    graph = script_module.graph.copy()
    # Check if lower_all_tuples pass can be enabled
    graph_inputs = list(graph.inputs())
    for inp in graph_inputs:
        if inp.type().kind() == "TupleType" or inp.type().kind() == "ListType":
            enable_lower_all_tuples = False
            break

    _run_jit_passes(graph, enable_lower_all_tuples)


    if custom_convert_map:
        converter.update_convert_map(custom_convert_map)

    op_names = get_all_op_names(graph)
    converter.report_missing_conversion(op_names)

    is_module = isinstance(script_module, torch.jit.ScriptModule)
    params = script_module.state_dict() if is_module else {}
    outputs = _get_relay_input_vars(
        graph, input_infos, prelude, default_dtype=default_dtype, is_module=is_module
    )

    if use_parser_friendly_name:
        new_names = [key.replace(".", "_") for key in params.keys()]
        params = dict(zip(new_names, params.values()))

    # rename _C.Graph here for constructing meaningful source name of graph nodes
    # by doing so, we could Use source_map as the reference to rename model parameters
    source_map = _debug_rename(graph, use_parser_friendly_name)
    param_vars, tensors, packed_param_map, param_debug_name_map, input_remap = convert_params(
        graph, params, source_map, use_parser_friendly_name, do_convert_params
    )

    tvm_params = {k: tvm.nd.array(v) if v is not None else v for k, v in tensors.items()}

    outputs.update(param_vars)

    # For quantized models
    quantized_ops = set(["aten::quantize_per_tensor", "quantized::linear_dynamic"])
    if len(quantized_ops.intersection(set(op_names))) > 0:
        assert do_convert_params, "Parameters need to be converted for quantized models"
        weight_quant_params = qnn_torch.get_weight_quant_params(
            script_module, packed_param_map.values()
        )
        qnn_torch.inline_input_quant_params_for_fx(graph, tensors, param_debug_name_map)
        input_scales_for_bias = qnn_torch.add_input_quant_params_to_op_inputs(graph)
        qnn_torch.add_quant_params_to_outputs(
            outputs,
            packed_param_map,
            weight_quant_params,
            input_scales_for_bias,
            keep_quantized_weight,
        )
        qnn_torch.add_quant_params(tvm_params, weight_quant_params)
        converter.update_convert_map(qnn_torch.convert_map)

    operator_nodes = _get_operator_nodes(
        graph.nodes(), converter.source_map, converter.op_type_dict, use_parser_friendly_name
    )
    outplace_inplace_ops(operator_nodes)
    ret_name = _get_input_names(graph.return_node(), input_remap)
    outputs = converter.convert_operators(operator_nodes, outputs, ret_name, input_remap)

    # ListConstruct kept original python list. Convert to tuple.
    outputs = [_expr.Tuple(output) if isinstance(output, list) else output for output in outputs]

    if len(outputs) > 1:
        ret = _expr.Tuple(outputs)
    else:
        ret = outputs[0]

    # Separate data inputs and parameters to make sure data inputs come first.
    func_args = []
    data_inputs = []
    for arg in _analysis.free_vars(ret):
        if arg.name_hint not in tvm_params.keys():
            data_inputs.append(arg)
        else:
            func_args.append(arg)

    # Ensures the order of data_input is the same as the order of inputs specified in input_info.
    order_input_infos = {
        input_info[0]: len(input_infos) - idx for idx, input_info in enumerate(input_infos)
    }
    data_inputs = sorted(
        data_inputs,
        key=lambda data_input: order_input_infos[data_input.name_hint]
        if data_input.name_hint in order_input_infos
        else -1,
        reverse=True,
    )

    func_args = data_inputs + func_args

    mod["main"] = tvm.relay.Function(func_args, ret)
    if not do_convert_params:
        tvm_params = None

    if export_renamed_c_graph_path:
        export_c_graph(export_renamed_c_graph_path, graph)

    return transform.RemoveUnusedFunctions()(mod), tvm_params
