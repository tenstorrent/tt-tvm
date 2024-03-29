// SPDX-FileCopyrightText: © 2019-2023 The Apache Software Foundation
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
 * \file src/runtime/contrib/libtorch/libtorch_runtime.cc
 * \brief runtime implementation for LibTorch/TorchScript.
 */

// we do not want clang to reorder our includes
// clang-format off
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/contrib/libtorch_runtime.h>

#include <ATen/dlpack.h>
#include <ATen/DLConvertor.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/torch.h>

// clang-format on

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {

static void monly_deleter(DLManagedTensor* self) { delete self; }

void run_torch_module(torch::jit::Module* module, TVMArgs args, TVMRetValue* rv) {
  std::vector<torch::jit::IValue> inputs;
  std::vector<torch::Tensor> outputs;
  auto m = module->get_method("forward");
  for (int i = 0; i < args.size(); i++) {
    const DLTensor* arg;
    if (args[i].IsObjectRef<NDArray>()) {
      NDArray arr = args[i];
      arg = arr.operator->();
    } else {
      arg = args[i].operator DLTensor*();
    }
    DLManagedTensor* inp = new DLManagedTensor{};
    inp->dl_tensor = *arg;
    inp->deleter = &monly_deleter;
    // m.num_inputs includes the self argument of forward(self, ...)
    // num_inputs - 1 is the number of (Tensor) inputs
    if (i < static_cast<int>(m.num_inputs()) - 1) {
      inputs.emplace_back(at::fromDLPack(inp));
    } else {
      outputs.emplace_back(at::fromDLPack(inp));
    }
  }
  ICHECK(outputs.size() == 1) << "wrong number of args, can handle only one output";
  torch::Tensor res = module->forward(inputs).toTensor();
  outputs[0].copy_(res);  // too bad
}

/*!
 * \brief A json runtime that executes the serialized JSON format. This runtime
 * can be extended by user defined runtime for execution.
 */
class TorchModuleNode : public ModuleNode {
 public:
  TorchModuleNode(const std::string& symbol_name, const torch::jit::Module& module)
      : symbol_name_(symbol_name), module_(module) {}

  const char* type_key() const { return "torch"; }
  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }

  /*!
   * \brief Get a packed function.
   * \param name The name/symbol of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The packed function.
   */
  virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_name_; });
    } else if (name == "get_const_vars") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = Array<String>{}; });
    } else if (this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        run_torch_module(&module_, args, rv);
      });
    } else if ("__init_" + this->symbol_name_ == name) {
      // The function to initialize constant tensors.
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = 0; });
    } else {
      return PackedFunc(nullptr);
    }
  }

  virtual void SaveToBinary(dmlc::Stream* stream) {
    // Save the symbol
    stream->Write(symbol_name_);
    // Save the module
    std::stringstream str;
    module_.save(str);
    stream->Write(str.str());
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string symbol;
    std::string module_str;
    // Load the symbol
    ICHECK(stream->Read(&symbol)) << "Loading symbol name failed";
    ICHECK(stream->Read(&module_str)) << "Loading module str failed";
    std::stringstream str(module_str);
    torch::jit::Module mod = torch::jit::load(str);
    auto n = make_object<TorchModuleNode>(symbol, mod);
    return Module(n);
  }

  /*!
   * \brief Get the source generated by codegen.
   *
   * \param format the format to return.
   * \return A string of JSON.
   */
  String GetSource(const String& format = "json") override {
    return module_.dump_to_str(true, true, true);
  }

 protected:
  /*! \brief The only subgraph name for this module. */
  std::string symbol_name_;
  /*! \brief Module. */
  torch::jit::Module module_;
};

runtime::Module TorchRuntimeCreate(const String& symbol_name,
                                   const std::string& serialized_function) {
  std::stringstream str(serialized_function);
  torch::jit::Module mod = torch::jit::load(str);
  auto n = make_object<TorchModuleNode>(symbol_name, mod);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_torch")
    .set_body_typed(TorchModuleNode::LoadFromBinary);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
