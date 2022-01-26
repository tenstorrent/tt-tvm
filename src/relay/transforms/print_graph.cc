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
 * \file print_graphcc
 */
#include <tvm/ir/attrs.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {
namespace transform {

class PrintGraphVisitor : public MixedModeVisitor {
 private:
  using ExprVisitor::VisitExpr_;
  void VisitLeaf(const Expr& expr) override {
    MixedModeVisitor::VisitLeaf(expr);
    
    std::cout << "Leaf: " << expr->span << std::endl; // << expr << std::endl;
  }
  void VisitExpr_(const CallNode* n) final {
    VisitExpr(n->op);

    std::cout << "CallNode op is: " << n->op << std::endl;
  }

  void VisitExpr_(const OpNode* n) final {
    std::cout << "OpNode is: " << n->name << std::endl;
  }

  void VisitExpr_(const ConstantNode* n) final {
    std::cout << "ConstantNode: " << std::endl;
  }
};

Pass PrintGraph() {
  std::cout << "Entering PrintGraph()" << std::endl;
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        PrintGraphVisitor().VisitExpr(f);
        return f;
      };
  return CreateFunctionPass(pass_func, 0, "PrintGraph", {});
}

TVM_REGISTER_GLOBAL("relay._transform.PrintGraph").set_body_typed(PrintGraph);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
