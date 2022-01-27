
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>

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

      if (name == "buda.matmul") {
        call = GetRootCall(fn->body.as<CallNode>(), 1, {"transpose", "nn.dense"});
        ICHECK(call->op.as<OpNode>()) << "Not op node";
      }
      /*
      if (name == "dnnl.conv2d_bias_relu") {
        call = GetRootCall(fn->body.as<CallNode>(), 2, {"nn.conv2d", "add", "nn.relu"});
      } else if (name == "dnnl.conv2d_bias_tanh") {
        call = GetRootCall(fn->body.as<CallNode>(), 2, {"nn.conv2d", "add", "tanh"});
        ICHECK(call->op.as<OpNode>()) << "Not op node";
      } else if (name == "dnnl.conv2d_bias_sigmoid") {
        call = GetRootCall(fn->body.as<CallNode>(), 2, {"nn.conv2d", "add", "sigmoid"});
        ICHECK(call->op.as<OpNode>()) << "Not op node";
      } else if (name == "dnnl.conv2d_bias") {
        call = GetRootCall(fn->body.as<CallNode>(), 1, {"nn.conv2d", "add"});
        ICHECK(call->op.as<OpNode>()) << "Not op node";
      } else if (name == "dnnl.conv2d_relu") {
        call = GetRootCall(fn->body.as<CallNode>(), 1, {"nn.conv2d", "nn.relu"});
        ICHECK(call->op.as<OpNode>()) << "Not op node";
      } else if (name == "dnnl.conv2d_tanh") {
        call = GetRootCall(fn->body.as<CallNode>(), 1, {"nn.conv2d", "tanh"});
        ICHECK(call->op.as<OpNode>()) << "Not op node";
      } else if (name == "dnnl.conv2d_sigmoid") {
        call = GetRootCall(fn->body.as<CallNode>(), 1, {"nn.conv2d", "sigmoid"});
        ICHECK(call->op.as<OpNode>()) << "Not op node";
      } else if (name == "dnnl.dense_bias") {
        call = GetRootCall(fn->body.as<CallNode>(), 1, {"nn.dense", "add"});
        ICHECK(call->op.as<OpNode>()) << "Not op node";
      } */else {
        LOG(FATAL) << "Unrecognized Buda pattern: " << name;
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
  std::cout << "STAN: call serialize" << std::endl;
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();

  std::cout << "Buda json: " << std::endl << graph_json << std::endl;

  //const auto* pf = runtime::Registry::Get("runtime.BudaJSONRuntimeCreate");
  const auto* pf = runtime::Registry::Get("runtime.DNNLJSONRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.buda").set_body_typed(BudaCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
