# SPDX-FileCopyrightText: © 2019-2023 The Apache Software Foundation
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
import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm.contrib import utils


def test_add():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = te.create_schedule(C.op)

    def check_c():
        mhost = tvm.build(s, [A, B, C], "c", name="test_fadd")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["test_fadd"]
        dev = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_c()


def test_add_pipeline():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    AA = te.compute((n,), lambda *i: A(*i), name="A")
    BB = te.compute((n,), lambda *i: B(*i), name="B")
    T = te.compute(A.shape, lambda *i: AA(*i) + BB(*i), name="T")
    C = te.compute(A.shape, lambda *i: T(*i), name="C")
    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    xo1, xo2 = s[C].split(xo, factor=13)
    s[C].parallel(xo2)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xo2, "parallel_stride_pattern")
    s[C].pragma(xo2, "parallel_barrier_when_finish")
    # FIXME(tvm-team): vector operators are not supported for codegen to C yet
    # s[C].vectorize(xi)

    def check_c():
        # Specifically allow offset to test codepath when offset is available
        Ab = tvm.tir.decl_buffer(
            A.shape, A.dtype, elem_offset=te.size_var("Aoffset"), offset_factor=8, name="A"
        )
        binds = {A: Ab}
        # BUILD and invoke the kernel.
        f1 = tvm.lower(s, [A, B, C], name="test_fadd_pipeline")
        mhost = tvm.build(f1, target="c")

        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["test_fadd_pipeline"]
        dev = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_c()


def test_reinterpret():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="int32")
    B = te.compute(
        A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.reinterpret", 2 + A(*i)), name="B"
    )
    s = te.create_schedule(B.op)

    def check_c():
        mhost = tvm.build(s, [A, B], "c", name="test_reinterpret")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["test_reinterpret"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.randint(-(2**30), 2**30, size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fadd(a, b)
        tvm.testing.assert_allclose(b.numpy(), (2 + a.numpy()).view("float32"))

    check_c()


def test_ceil():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.compute(A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.ceil", A(*i)), name="B")
    s = te.create_schedule(B.op)

    def check_c():
        mhost = tvm.build(s, [A, B], "c", name="test_ceil")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fceil = m["test_ceil"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.rand(n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fceil(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.ceil(a.numpy()).view("float32")))

    check_c()


def test_floor():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.compute(A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.floor", A(*i)), name="B")
    s = te.create_schedule(B.op)

    def check_c():
        mhost = tvm.build(s, [A, B], "c", name="test_floor")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        ffloor = m["test_floor"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.rand(n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        ffloor(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.floor(a.numpy()).view("float32")))

    check_c()


def test_round():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.compute(A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.round", A(*i)), name="B")
    s = te.create_schedule(B.op)

    def check_c():
        mhost = tvm.build(s, [A, B], "c", name="test_round")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fround = m["test_round"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.rand(n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fround(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.round(a.numpy()).view("float32")))

    check_c()


def test_call_packed():
    def fake_func(fname="fake.func"):
        ib = tvm.tir.ir_builder.create()
        A = ib.pointer("float32", name="A")
        fake_func1 = tvm.tir.call_packed(fname, A[0])

        ib.emit(fake_func1)
        body = ib.get()
        return A, body

    def check_global_packed_func():
        fname = "fake.func"
        A, body = fake_func(fname)
        func1 = tvm.tir.PrimFunc([A], body).with_attr("global_symbol", "func1")
        B, body = fake_func()
        func2 = tvm.tir.PrimFunc([B], body).with_attr("global_symbol", "func2")
        mod = tvm.IRModule({"fake_func1": func1, "fake_func2": func2})
        fcode = tvm.build(mod, None, "c")
        src = fcode.get_source()

        # there are two locations calling the packed func
        assert src.count(fname) == 2

        suffix = "_packed"
        packed_func_name = fname + suffix
        # func name will be standardized by GetUniqueName and not exists anymore
        assert src.find(packed_func_name) == -1

        packed_func_real_name = "_".join(fname.split(".")) + suffix
        func_declaration = "static void* %s = NULL;" % packed_func_real_name
        # src only has 1 valid declaration
        assert src.count(func_declaration) == 1

    check_global_packed_func()


if __name__ == "__main__":
    test_add()
    test_add_pipeline()
    test_reinterpret()
    test_ceil()
    test_floor()
    test_round()
    test_call_packed()
