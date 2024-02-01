// SPDX-FileCopyrightText: © 2019-2023 The Apache Software Foundation © 2024 Tenstorrent AI ULC
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
 * \file simplify_inference.cc
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>

#include "pattern_utils.h"

namespace tvm {
namespace relay {


Expr VarianceToInferUnpack(const Attrs attrs, Expr data, Expr mean) {
  const auto param = attrs.as<VarianceAttrs>();
  ICHECK(param);
  Expr sub_ = Subtract(data, mean);
  Expr mul_ = Multiply(sub_, sub_);
  Expr var = Mean(mul_, {param->axis}, true, false);
  return var;
}

class VarianceDecomposer : public MixedModeMutator {
 public:
  VarianceDecomposer()
      : variance_op_(Op::Get("variance")) {}


  Expr Rewrite_(const CallNode* n, const Expr& new_n) {
    if (n->op == variance_op_) {
      const auto* call = new_n.as<CallNode>();
      return VarianceToInferUnpack(call->attrs, call->args[0], call->args[1]);
    }
    return new_n;
  }

 private:
  // Cache the following ops. They will be used in the passes repeatedly for
  // operator equivalence checking so that the registry lookup overhead can be
  // reduced.

  const Op& variance_op_;
  std::unordered_map<Expr, Type, ObjectPtrHash, ObjectPtrEqual> ty_map_;
};

Expr DecomposeVariance(const Expr& e) { return VarianceDecomposer().Mutate(e); }

namespace transform {

Pass DecomposeVariance() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(DecomposeVariance(f));
      };
  return CreateFunctionPass(pass_func, 0, "DecomposeVariance", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.DecomposeVariance").set_body_typed(DecomposeVariance);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
