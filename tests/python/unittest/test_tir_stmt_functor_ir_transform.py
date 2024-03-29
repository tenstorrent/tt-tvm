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
from tvm import te


def test_ir_transform():
    ib = tvm.tir.ir_builder.create()
    n = te.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            x = tvm.tir.call_extern("int32", "TestA", i * 3 + j * 1)
            ib.emit(tvm.tir.call_extern("int32", "TestB", x))
            ib.emit(tvm.tir.call_extern("int32", "TestC", x))
    body = ib.get()
    builtin_call_extern = tvm.ir.Op.get("tir.call_extern")

    def preorder(op):
        if op.op.same_as(builtin_call_extern) and op.args[0].value == "TestC":
            return tvm.tir.const(42, "int32")
        return None

    def postorder(op):
        assert isinstance(op, tvm.tir.Call)
        if op.op.same_as(builtin_call_extern) and op.args[0].value == "TestA":
            return tvm.tir.call_extern("int32", "TestB", op.args[1] + 1)
        return op

    body = tvm.tir.stmt_functor.ir_transform(body, preorder, postorder, ["tir.Call"])
    stmt_list = tvm.tir.stmt_list(body.body.body)
    assert stmt_list[0].value.args[1].args[0].value == "TestB"
    assert stmt_list[1].value.value == 42


if __name__ == "__main__":
    test_ir_transform()
