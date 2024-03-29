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
 * \file intrin_rule_aocl.cc
 * \brief AOCL intrinsic rules.
 */
#include <tvm/tir/op_attr_types.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
using tir::FLowerIntrinsic;

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.popcount")
    .set_attr<FLowerIntrinsic>("aocl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.popcount")
    .set_attr<FLowerIntrinsic>("aocl_sw_emu.FLowerIntrinsic", DispatchPureExtern<Direct>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
