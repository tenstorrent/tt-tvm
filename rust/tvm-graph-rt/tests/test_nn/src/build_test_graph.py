# SPDX-FileCopyrightText: © 2019-2023 The Apache Software Foundation
#
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
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

"""Builds a simple graph for testing."""

from os import path as osp
import sys

from tvm import runtime as tvm_runtime
from tvm import relay
from tvm.relay import testing


def _get_model(dshape):
    data = relay.var("data", shape=dshape)
    fc = relay.nn.dense(data, relay.var("dense_weight"), units=dshape[-1] * 2)
    fc = relay.nn.bias_add(fc, relay.var("dense_bias"))
    left, right = relay.split(fc, indices_or_sections=2, axis=1)
    one = relay.const(1, dtype="float32")
    return relay.Tuple([(left + one), (right - one), fc])


def main():
    dshape = (4, 8)
    net = _get_model(dshape)
    mod, params = testing.create_workload(net)
    runtime = relay.backend.Runtime("cpp", {"system-lib": True})
    graph, lib, params = relay.build(mod, "llvm", runtime=runtime, params=params)

    out_dir = sys.argv[1]
    lib.save(osp.join(sys.argv[1], "graph.o"))
    with open(osp.join(out_dir, "graph.json"), "w") as f_resnet:
        f_resnet.write(graph)

    with open(osp.join(out_dir, "graph.params"), "wb") as f_params:
        f_params.write(tvm_runtime.save_param_dict(params))


if __name__ == "__main__":
    main()
