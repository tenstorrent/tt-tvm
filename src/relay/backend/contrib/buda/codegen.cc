
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


class BudaJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  BudaJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    Expr expr = GetRef<Expr>(cn);
    std::string name;
    const CallNode* call = cn;
    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      auto comp = fn->GetAttr<String>(attr::kComposite);
      auto body = fn->body.as<CallNode>();
      ICHECK(comp.defined()) << "Buda JSON runtime only supports composite functions.";
      name = comp.value();
      if (name == "buda.select") {
        call = GetRootCall(body, 0, "strided_slice");
      } else if (name == "pybuda.concatenate") {
        call = GetRootCall(fn->body.as<CallNode>(), 0, "concatenate");
      } else if (name == "pybuda.buda_conv2d_with_bias") {
        std::vector<std::string> names = {"nn.conv2d", "nn.bias_add"};
        call = GetRootCall(fn->body.as<CallNode>(), 1, names);
      } else if (name == "pybuda.adv_index") {
        call = GetRootCall(fn->body.as<CallNode>(), 0, "adv_index");
      } else if (name == "pybuda.channel_last_conv") {
        call = GetRootCall(fn->body.as<CallNode>(), 6, "nn.conv2d");
      } else if (name == "pybuda.channel_last_maxpool") {
        call = GetRootCall(fn->body.as<CallNode>(), 6, "nn.max_pool2d");
      } else if (name == "pybuda.channel_last_resize2d") {
        call = GetRootCall(fn->body.as<CallNode>(), 4, "image.resize2d");
      } else if (name == "pybuda.channel_last_conv2d_transpose") {
        call = GetRootCall(fn->body.as<CallNode>(), 6, "nn.conv2d_transpose");
      }
    } else {
      LOG(FATAL) << "Buda JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }

    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    SetCallNodeAttribute(node, call);
    return AddNode(node, GetRef<Expr>(cn));
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module BudaCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  BudaJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();

  const auto* jgr = runtime::Registry::Get("retrieve_pybuda_json_graph");
  ICHECK(jgr != nullptr) << "Cannot find retrieve_pybuda_json_graph";
  (*jgr)(func_name, graph_json, params);

  const auto* pf = runtime::Registry::Get("runtime.BudaRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.pybuda").set_body_typed(BudaCompiler);



/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module BudaCPUCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  BudaJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();

  const auto* jgr = runtime::Registry::Get("retrieve_pybuda_cpudevice_json_graph");
  ICHECK(jgr != nullptr) << "Cannot find retrieve_pybuda_cpudevice_json_graph";
  (*jgr)(func_name, graph_json, params);

  const auto* pf = runtime::Registry::Get("runtime.BudaRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.pybuda_cpudevice").set_body_typed(BudaCPUCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

