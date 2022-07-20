
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
      ICHECK(comp.defined()) << "Buda JSON runtime only supports composite functions.";
      name = comp.value();
      if (name == "buda.select") {
        call = GetRootCall(fn->body.as<CallNode>(), {"strided_slice"});
      } else if (name == "buda.concatenate") {
        call = GetRootCall(fn->body.as<CallNode>(), {"concatenate"});
      } else if (name == "buda.buda_conv2d_with_bias") {
        call = GetRootCall(fn->body.as<CallNode>(), {"nn.conv2d", "nn.bias_add"});
      }
      // if (name == "buda.matmul") {
      //   call = GetRootCall(fn->body.as<CallNode>(), 1, {"transpose", "nn.dense"});
      //   ICHECK(call->op.as<OpNode>()) << "Not op node";
      // } else {
      //   LOG(FATAL) << "Unrecognized Buda pattern: " << name;
      // }
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

  // std::cout << "Buda json: " << std::endl << graph_json << std::endl;
  const auto* jgr = runtime::Registry::Get("retrieve_json_graph");
  ICHECK(jgr != nullptr) << "Cannot find retrieve_json_graph";
  (*jgr)(func_name, graph_json, params);

  const auto* pf = runtime::Registry::Get("runtime.BudaRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.buda").set_body_typed(BudaCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

