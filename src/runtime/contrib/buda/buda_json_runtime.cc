// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/buda/buda_json_runtime.cc
 * \brief A simple JSON runtime for Buda.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>
#include <assert.h>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"


#include "graph_lib/python_bindings.hpp"

// #include "torch/torch.h"
// #include "torch/csrc/autograd/utils/wrap_outputs.h"
// #include "torch/csrc/autograd/python_variable.h"
#include "pybind11/pytypes.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

using Graph = tt::graphlib::Graph;
using namespace tt::graphlib;

class BudaRuntime : public JSONRuntimeBase {

 public:
  BudaRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "buda_json"; }

  void Init(const Array<NDArray>& consts) override {
  }
  void Run() override {
  }

  // Build up the engine based on the input graph.
  void BuildEngine() {
  }

};

runtime::Module BudaRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<BudaRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.BudaRuntimeCreate").set_body_typed(BudaRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_buda_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<BudaRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
