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
    // BuildEngine();

    // ICHECK_EQ(consts.size(), const_idx_.size())
    //     << "The number of input constants must match the number of required.";

    // // Setup constants entries for weights.
    // SetupConstants(consts);
  }
  void Run() override {
    //   if (true) {
    //     return;
    //   }
    //   std::cout << "BudaRuntime::Run" << std::endl;

    //   const auto* pf = Registry::Get("retrieve_pybuda_graph");
    //   ICHECK(pf != nullptr) << "Cannot find retrieve_pybuda_graph";

    //   std::vector<TVMValue> values(input_nodes_.size() * 2 + 1);
    //   std::vector<int> codes(input_nodes_.size() * 2 + 1);
    //   TVMArgsSetter setter(values.data(), codes.data());

    //   for (size_t i = 0; i < input_nodes_.size(); ++i) {
    //     auto eid = EntryID(input_nodes_[i], 0);

    //     DLTensor *dl_tensor;

    //     TVMArrayAlloc(data_entry_[eid]->shape, data_entry_[eid]->ndim, data_entry_[eid]->dtype.code, data_entry_[eid]->dtype.bits, data_entry_[eid]->dtype.lanes, data_entry_[eid]->device.device_type, data_entry_[eid]->device.device_id, &dl_tensor);
    //     NDArray::CopyFromTo(data_entry_[eid], dl_tensor, NULL);
        
    //     tt::graphlib::NodeId node_id = std::get<0>(id_to_tensor_.at(eid));
    //     // Set Data
    //     setter(2*i, dl_tensor);

    //     // Set Type + Name String
    //     std::string input_string = graph_->node_by_id(node_id)->as<InputNode>()->input_type_string() + "|||" + graph_->node_by_id(node_id)->name();
    //     // Setter is by reference, need to dynamically allocate
    //     char* p = new char[input_string.size()];

    //     strcpy(p, input_string.c_str());
    //     setter(2*i + 1, p);
    //   }
    //   // setter(i, static_cast<void *>(graph_));
    //   values[values.size() - 1].v_handle = static_cast<void *>(graph_);
    //   codes[codes.size() - 1] = TVMArgTypeCode::kTVMOpaqueHandle;

    //   TVMRetValue rv;
    //   pf->CallPacked(TVMArgs(values.data(), codes.data(), values.size()), &rv);

    //   // TODO (arui) : Free the pointers passed to TVMArgs
    //   std::cout << "Returned" << std::endl;

    //   DLTensor *ret_value = static_cast<DLTensor *>(rv);
    //   size_t numel = 1;
    //   for (int i = 0; i < ret_value->ndim; i++) {
    //     numel = numel * ret_value->shape[i];
    //   }

    //   memcpy(data_entry_[outputs_[0].id_]->data , ret_value->data, numel * sizeof(float));      
  }
 private:
  std::map <uint32_t, std::tuple<tt::graphlib::NodeId, std::string, int, Shape>> id_to_tensor_;
  Graph* graph_;
  int buda_node_id_ = 0;

  const Shape MakeBudaShape(std::vector<int64_t> shape) {
    std::vector<int> shape_4d;
    for (size_t i = 0; i < shape.size(); i++) {
      shape_4d.push_back(static_cast<int>(shape[i]));
    }
    while (shape_4d.size() < 4) {
      shape_4d.insert(shape_4d.begin(), 1);
    }

    for (uint i = 2; i < shape_4d.size(); i++) {
        if (shape_4d[i] % 32 != 0){
          shape_4d[i] = (shape_4d[i] / 32 + 1) * 32;
        }
    }
    const Shape buda_shape = Shape(shape_4d);
    return buda_shape;
  }

  void PrintMap() {
    std::cout << "id_to_tensor_ entries:" << std::endl;
    for (std::pair<uint32_t, std::tuple<tt::graphlib::NodeId, std::string, int, Shape>> entry : id_to_tensor_) {
      std::cout << "  " << entry.first << ", " << std::get<0>(entry.second) << ", " << std::get<1>(entry.second) 
        <<  ", " << std::get<2>(entry.second) <<   ", " << std::get<3>(entry.second) << std::endl;
    }
  }
  // Build up the engine based on the input graph.
  void BuildEngine() {
    // std::cout << "BudaRuntime::Build" << std::endl;
    // graph_ = new Graph();

    // std::vector<tt::graphlib::NodeId> module_inputs;
    // for (size_t i = 0; i < input_nodes_.size(); ++i) {
    //   uint32_t eid = EntryID(input_nodes_[i], 0);
    //   auto shape = nodes_[eid].GetOpShape()[0];
    //   auto name = nodes_[eid].GetOpName();

    //   InputNodeType input_type;
    //   bool requires_grad = false;
    //   if (std::count(const_idx_.begin(), const_idx_.end(), eid) == 0) {
    //     input_type = InputNodeType::Activation;
    //     // std::cout << "Creating activations tensor: " << name << std::endl;
    //   }
    //   else {
    //     requires_grad = true;
    //     input_type = InputNodeType::Parameter;
    //     // std::cout << "Creating parameter tensor: " << name << std::endl;
    //   }

    //   auto node = graph_->add_node(create_node<InputNode>(name + "_" + std::to_string(buda_node_id_++), input_type, requires_grad));
    //   module_inputs.push_back(node->id());
      
    //   const Shape buda_shape = MakeBudaShape(shape);
    //   std::cout << "Node: " << node->id() << " shape: " << buda_shape<< " type: " << input_type << std::endl;
    //   node->set_shape(buda_shape);
      
    //   id_to_tensor_.emplace(eid, std::make_tuple(node->id(), name, 0, buda_shape));
    // }

    // graph_->register_module_inputs(module_inputs);
    // for (size_t nid = 0; nid < nodes_.size(); ++nid) {
    //   std::vector<int> attributes;
    //   const auto& node = nodes_[nid];
    //   if (node.GetOpType() == "kernel") {
    //     ICHECK_EQ(node.GetOpType(), "kernel");
    //     auto op_name = node.GetOpName();
    //     std::string op_type;
    //     if ("add" == op_name) {
    //       op_type = "add";
    //     } else if ("multiply" == op_name) {
    //       op_type = "multiply";
    //     } else if ("subtract" == op_name) {
    //       op_type = "subtract";
    //     } else if ("sqrt" == op_name) {
    //       op_type = "sqrt";
    //     } else if ("reciprocal" == op_name) {
    //       op_type = "reciprocal";
    //     } else if ("buda.matmul" == op_name) {
    //       op_type = "matmul";
    //     } else if ("nn.batch_matmul" == op_name) {
    //       op_type = "matmul";
    //     } else if ("reshape" == op_name) {
    //       if (DoSkipReshape(nid, &attributes)) {
    //         continue; 
    //       }
    //     } else if ("transpose" == op_name) {
    //       op_type = "transpose";
    //     } else if ("nn.softmax" == op_name) {
    //       std::string axis = node.GetAttr<std::vector<std::string>>("axis")[0];
    //       attributes.push_back(std::stoi(axis));
    //       ExpandCompoundOps(nid, &attributes);          
    //       continue; 
    //     } else if ("layernorm" == op_name) {
    //       std::string axis = node.GetAttr<std::vector<std::string>>("axis")[0];
    //       attributes.push_back(std::stoi(axis));
    //       std::string epsilon = node.GetAttr<std::vector<std::string>>("epsilon")[0];
    //       attributes.push_back(std::stoi(epsilon));
    //       ExpandCompoundOps(nid, &attributes);          
    //       continue; 
    //     } else if ("buda.hslice" == op_name) {
    //       op_type = "hslice";
    //       PopulateHSliceAttrs(nid, &attributes);
    //     } else if ("buda.hstack" == op_name) {
    //       op_type = "hstack";
    //       PopulateHStackAttrs(nid, &attributes);
    //     } else if ("mean" == op_name) {
    //       op_type = "reduce_avg";
    //       PopulateReduceAttrs(nid, &attributes);
    //     } else if ("gelu" == op_name) {
    //       op_type = "gelu";
    //     } else {
    //       LOG(FATAL) << "Unsupported op: " << op_name;
    //     }
    //     std::string buda_name = op_type + "_" + std::to_string(buda_node_id_++);
    //     auto shape = nodes_[nid].GetOpShape()[0];
    //     auto buda_node = graph_->add_node(create_node<OpNode>(buda_name, OpType{.op=op_type, .attr=attributes}));
    //     const Shape buda_shape = MakeBudaShape(shape);
    //     std::cout << "Node: " << buda_node->id() << " shape: " << buda_shape<< " type: " << buda_name << std::endl;
    //     buda_node->set_shape(buda_shape);

    //     id_to_tensor_.emplace(nid, std::make_tuple(buda_node->id(), buda_name, 0, buda_shape));

    //     auto inputs = node.GetInputs();
    //     for (unsigned int i = 0; i < inputs.size(); i++) {
    //       auto input_id = inputs[i].id_;
    //       // if this node is to be skipped
    //       while (std::get<0>(id_to_tensor_.at(input_id)) == -1) {
    //         auto inputs = nodes_[input_id].GetInputs();
    //         ICHECK_EQ(inputs.size(), 1);
    //         input_id = inputs[0].id_;
    //       }
          
    //       Edge edge(std::get<0>(id_to_tensor_.at(input_id)), 0, buda_node->id(), i, EdgeType::kData);
    //       std::cout << "Edge from: " << std::get<0>(id_to_tensor_.at(input_id)) << ":0 to " << buda_node->id() << ":" << i << std::endl;
    //       graph_->add_edge(edge);
    //     }
    //   }
    // }

    // std::vector<tt::graphlib::NodeId> module_outputs;
    // for (auto output : outputs_) {
    //   auto input_id = output.id_;
    //   // if this node is to be skipped
    //   while (std::get<0>(id_to_tensor_.at(input_id)) == -1) {
    //     auto inputs = nodes_[input_id].GetInputs();
    //     ICHECK_EQ(inputs.size(), 1);
    //     input_id = inputs[0].id_;
    //   }
    //   std::string buda_name = nodes_[input_id].GetOpName() + "_" + std::to_string(buda_node_id_++) + "_output";
    //   auto buda_node = graph_->add_node(create_node<OutputNode>(buda_name));
    //   module_outputs.push_back(buda_node->id());
    //   std::cout << "Node: " << buda_node->id() << " type: " << buda_name << std::endl;
    //   auto shape = nodes_[input_id].GetOpShape()[0];
    //   const Shape buda_shape = MakeBudaShape(shape);
    //   buda_node->set_shape(buda_shape);
      
    //   Edge edge(std::get<0>(id_to_tensor_.at(input_id)), 0, buda_node->id(), 0, EdgeType::kData);
    //   std::cout << "Edge from: " << std::get<0>(id_to_tensor_.at(input_id)) << ":0 to " << buda_node->id() << ":" << 0 << std::endl;
    //   graph_->add_edge(edge);

    // }
    // graph_->register_module_outputs(module_outputs);
  }

  void PopulateHSliceAttrs(const size_t& nid, std::vector<int> *attributes) {
    auto shape = nodes_[nid].GetOpShape()[0];
    const Shape buda_shape = MakeBudaShape(shape);
    attributes->push_back(buda_shape[1]);
  }

  void PopulateHStackAttrs(const size_t& nid, std::vector<int> *attributes) {
    auto inputs = nodes_[nid].GetInputs();
    ICHECK_EQ(inputs.size(), 1) << "HStack can only have one input";
    
    auto input_shape = nodes_[inputs[0].id_].GetOpShape()[0];
    const Shape buda_shape = MakeBudaShape(input_shape);
    attributes->push_back(buda_shape[1]);
  }

  void PopulateReduceAttrs(const size_t& nid, std::vector<int> *attributes) {
    std::string axis = nodes_[nid].GetAttr<std::vector<std::string>>("axis")[0];
    attributes->push_back(std::stoi(axis));
  }

  void ExpandCompoundOps(const size_t& nid, std::vector<int> *attributes) {
    // const auto* expand_compound_ops = Registry::Get("expand_compound_ops");
    // ICHECK(expand_compound_ops != nullptr) << "Cannot find expand_compound_ops";

    // auto inputs = nodes_[nid].GetInputs();

    // // arguments are passed as follows:
    // // num_inputs, input0, input1, ..., num_attributes, attribute0, attribute1, ..., op_name
    // size_t num_elements = inputs.size() * 4 + attributes->size() + 3;

    // std::vector<TVMValue> values(num_elements);
    // std::vector<int> codes(num_elements);
    // TVMArgsSetter setter(values.data(), codes.data());
    // int i = 0;

    // setter(i++, inputs.size());
    // for (auto input : inputs) {
    //     auto shape = nodes_[input.id_].GetOpShape()[0];
    //     auto buda_shape = MakeBudaShape(shape).as_vector();
    //     for (auto elem : buda_shape) {
    //       setter(i++, elem);
    //     }
    // }

    // setter(i++, attributes->size());
    // for (int attribute : *attributes) {
    //   setter(i++, attribute);
    // }

    // setter(i++, nodes_[nid].GetOpName());

    // TVMRetValue rv;
    // expand_compound_ops->CallPacked(TVMArgs(values.data(), codes.data(), values.size()), &rv);

    // Graph* subgraph; 

    // subgraph = reinterpret_cast<Graph *>((void *)rv);

    // std::map <NodeId, NodeId> subgraph_to_graph_node_id;

    // auto node_inputs = nodes_[nid].GetInputs();
    // std::vector<Node *> subgraph_inputs = subgraph->ordered_module_inputs();
    // ICHECK_EQ(node_inputs.size(), subgraph_inputs.size()) << "Number of inputs does not match, something went wrong!";

    // std::vector<Node *> subgraph_outputs = subgraph->ordered_module_outputs();

    // std::cout << "Got subgraph with num nodes: " << subgraph->num_nodes() << std::endl;
    // for (Node *node : topological_sort(*subgraph)) {
    //   if (node->node_type() != NodeType::kOp)
    //     continue;
      
    //   Node *buda_node;
    //   std::string buda_name;
    //   Shape buda_shape;
    //   buda_name = node->name() + "_" + std::to_string(buda_node_id_++);
    //   buda_shape = node->shape();

    //   OpNode *op_node = node->as<OpNode>();
    //   buda_node = graph_->add_node(create_node<OpNode>(buda_name, OpType{.op=op_node->op_type().op, .attr=op_node->op_attrs()}));
    //   subgraph_to_graph_node_id[node->id()] = buda_node->id();
    //   std::cout << "Node: " << buda_node->id() << " shape: " << buda_shape<< " type: " << buda_name << std::endl;
    //   buda_node->set_shape(buda_shape);
      
    //   size_t output_index = 0;
    //   for (Node *subgraph_output : subgraph_outputs) {
    //     auto last_op_node = subgraph->operand_data_edges(subgraph_output);
    //     ICHECK_EQ(last_op_node.size(), 1);
    //     if (last_op_node[0].producer_node_id == node->id()) {
    //       id_to_tensor_.emplace(nid, std::make_tuple(buda_node->id(), buda_name, output_index, buda_shape));
    //       std::cout << "Found subgraph output: " << output_index << " at id: " << buda_node->id() << std::endl;
    //       break;
    //     }
    //     output_index++;
    //   }

    //   int i = 0;
    //   for (Edge &input_edge : subgraph->operand_data_edges(node)) {
    //     auto subgraph_input_node_id = input_edge.producer_node_id;
    //     auto input_node = subgraph->node_by_id(subgraph_input_node_id);
    //     if (input_node->node_type() == NodeType::kInput) {
    //       size_t input_index = 0;
    //       for (Node *subgraph_input : subgraph_inputs) {
    //         if (subgraph_input->id() == subgraph_input_node_id) {
    //           std::cout << "Found subgraph input: " << input_index << " at id: " << buda_node->id() << std::endl;
    //           break;
    //         }
    //         input_index++;
    //       }
          
    //       auto input_id = inputs[input_index].id_;

    //       Edge edge(std::get<0>(id_to_tensor_.at(input_id)), 0, buda_node->id(), i, EdgeType::kData);
    //       std::cout << "Edge from: " << std::get<0>(id_to_tensor_.at(input_id)) << ":0 to " << buda_node->id() << ":" << i << std::endl;
    //       graph_->add_edge(edge);
    //       std::shared_ptr<EdgeAttributes> in_attr = subgraph->get_edge_attributes(input_edge);
    //       std::shared_ptr<EdgeAttributes> out_attr = graph_->get_edge_attributes(edge);
    //       out_attr->set_tms(in_attr->get_tms());

    //     }
    //     else {
    //       auto input_node_id = subgraph_to_graph_node_id[input_edge.producer_node_id];
    //       Edge edge(input_node_id, input_edge.producer_output_port_id, buda_node->id(), i, EdgeType::kData);
    //       std::cout << "Edge from: " << input_node_id << ":" << input_edge.producer_output_port_id << " to " << buda_node->id() << ":" << i << std::endl;
    //       graph_->add_edge(edge);
    //     }
    //     i++;
    //     //TODO: Broadcast
    //   }
    // }

    
  }

  bool DoSkipReshape(const size_t& nid, std::vector<int> *attributes) {
    auto output_shape = nodes_[nid].GetOpShape()[0];
    auto inputs = nodes_[nid].GetInputs();
    ICHECK_EQ(inputs.size(), 1);
    auto input_shape = nodes_[inputs[0].id_].GetOpShape()[0];

    bool can_remove = true;
    if (output_shape.size() > input_shape.size()) {
      for (size_t i = 0; i < input_shape.size(); i++) {
        if (i < (output_shape.size() - input_shape.size())) {
          if (output_shape[i] != 1) { 
            can_remove = false;
          }
        }
        if (input_shape[i] != output_shape[i + output_shape.size() - input_shape.size()]) {
          can_remove = false;
        }
      }
    }
    else if (input_shape.size() > output_shape.size()) {
      for (size_t i = 0; i < output_shape.size(); i++) {
        if (i < (input_shape.size() - output_shape.size())) {
          if (input_shape[i] != 1) { 
            can_remove = false;
          }
        }
        if (output_shape[i] != input_shape[i + input_shape.size() - output_shape.size()]) {
          can_remove = false;
        }
      }
    }
    // std::cout << " can be removed: " << can_remove << std::endl;

    // if we can't remove it, see if it can be replaced w/ a hslice
    if (can_remove) {
      id_to_tensor_.emplace(nid, std::make_tuple(-1, "", 0, Shape()));
    }
    else {
      // std::cout << "Here" << std::endl;
      auto hdim = output_shape.size() - 2;
      attributes->push_back(output_shape[hdim]);
    }
    ICHECK_EQ(can_remove, true)
        << "Reshape not supported at this time.";
    return can_remove;
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
