// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>
#include <fstream>

#include "../../utils.h"

#include "../../../../runtime/contrib/json/json_node.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;


class ForgeJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  ForgeJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    Expr expr = GetRef<Expr>(cn);
    std::string name;
    std::string span;

    const CallNode* call = cn;
    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      auto comp = fn->GetAttr<String>(attr::kComposite);
      auto body = fn->body.as<CallNode>();
      ICHECK(comp.defined()) << "Forge JSON runtime only supports composite functions.";
      name = comp.value();
      if (name == "forge.select") {
        call = GetRootCall(body, 0, "strided_slice");
      } else if (name == "forge.concatenate") {
        call = GetRootCall(fn->body.as<CallNode>(), 0, "concatenate");
      } else if (name == "forge.forge_conv2d_with_bias") {
        std::vector<std::string> names = {"nn.conv2d", "nn.bias_add"};
        call = GetRootCall(fn->body.as<CallNode>(), 1, names);
      } else if (name == "forge.forge_conv2d_transpose_with_bias") {
        std::vector<std::string> names = {"nn.conv2d_transpose", "nn.bias_add"};
        call = GetRootCall(fn->body.as<CallNode>(), 1, names);
      } else if (name == "forge.adv_index") {
        call = GetRootCall(fn->body.as<CallNode>(), 0, "adv_index");
      } else if (name == "forge.channel_last_conv") {
        call = GetRootCall(fn->body.as<CallNode>(), 6, "nn.conv2d");
      } else if (name == "forge.channel_last_maxpool") {
        call = GetRootCall(fn->body.as<CallNode>(), 6, "nn.max_pool2d");
      } else if (name == "forge.channel_last_resize2d") {
        call = GetRootCall(fn->body.as<CallNode>(), 4, "image.resize2d");
      } else if (name == "forge.channel_last_conv2d_transpose") {
        call = GetRootCall(fn->body.as<CallNode>(), 6, "nn.conv2d_transpose");
      }
    } else {
      LOG(FATAL) << "Forge JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }

    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    if (cn->span.defined() and not cn->span->source_name->name.empty()) {
      node->SetAttr("span", std::string(cn->span->source_name->name));
    }
    SetCallNodeAttribute(node, call);
    return AddNode(node, GetRef<Expr>(cn));
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module ForgeCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  ForgeJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.const_names();

  const auto* jgr = runtime::Registry::Get("retrieve_forge_json_graph");
  ICHECK(jgr != nullptr) << "Cannot find retrieve_forge_json_graph";
  (*jgr)(func_name, graph_json, params);

  const auto* pf = runtime::Registry::Get("runtime.ForgeRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.forge").set_body_typed(ForgeCompiler);

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module ForgeCPUCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  ForgeJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.const_names();

  const auto* jgr = runtime::Registry::Get("retrieve_forge_cpudevice_json_graph");
  ICHECK(jgr != nullptr) << "Cannot find retrieve_forge_cpudevice_json_graph";
  (*jgr)(func_name, graph_json, params);

  const auto* pf = runtime::Registry::Get("runtime.ForgeRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.forge_cpudevice").set_body_typed(ForgeCPUCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

