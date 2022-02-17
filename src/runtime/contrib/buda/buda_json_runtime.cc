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
    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
  }
  void Run() override {
      std::cout << "BudaRuntime::Run" << std::endl;

      const auto* pf = Registry::Get("my_py_packed_func");
      ICHECK(pf != nullptr) << "Cannot find my_py_packed_func";

    //   const auto* pf = Registry::Get("initialize_device_packed_func");
    //   ICHECK(pf != nullptr) << "Cannot find initialize_device_packed_func";
      // std::vector<py::object> torch_tensors;

      std::vector<TVMValue> values(input_nodes_.size() * 2 + 1);
      std::vector<int> codes(input_nodes_.size() * 2 + 1);
      TVMArgsSetter setter(values.data(), codes.data());

      size_t i;
      std::vector<Node *> ordered_inputs = graph_->ordered_inputs();

      // Cast to InputNode ptr
      std::vector<InputNode *> order_input_nodes;
      for (Node* node_ptr : ordered_inputs) {
        InputNode* inp_ptr = dynamic_cast<InputNode*>(node_ptr);
        if (inp_ptr == NULL){
            assert(("Cannot cast Node * to InputNode *", 0));
        }
        order_input_nodes.push_back(inp_ptr);
      }

      for (i = 0; i < input_nodes_.size(); ++i) {
        auto eid = EntryID(input_nodes_[i], 0);

        DLTensor *dl_tensor;

        std::vector<int64_t> reversed_shape;
        for (auto i = data_entry_[eid]->ndim - 1; i >= 0; i--) {
          reversed_shape.push_back(data_entry_[eid]->shape[i]);
        }

        TVMArrayAlloc(&reversed_shape[0], data_entry_[eid]->ndim, data_entry_[eid]->dtype.code, data_entry_[eid]->dtype.bits, data_entry_[eid]->dtype.lanes, data_entry_[eid]->device.device_type, data_entry_[eid]->device.device_id, &dl_tensor);
        
        NDArray::CopyFromTo(data_entry_[eid], dl_tensor, NULL);
        //TODO: loop over z and w
        auto shape = dl_tensor->shape;
        float *row_major_data = static_cast<float*>(dl_tensor->data);
        float *col_major_data = static_cast<float*>(data_entry_[eid]->data);
        
        for (int i = 0; i < shape[0]; i ++) {
          for (int j = 0; j < shape[1]; j++) {
            row_major_data[(i * shape[1]) + j] = col_major_data[(j * shape[0]) + i];
          }
        }
        
        tt::graphlib::NodeId node_id = std::get<0>(id_to_tensor_.at(eid));
        size_t pos = 0;
        for (InputNode *ordered_input : order_input_nodes) {
            std::cout << "Node type is " << ordered_input->input_type().c_str() << " with name " << ordered_input->name()<< std::endl;
          if (node_id == ordered_input->id()) {
            // Set Data
            std::cout << "Setting input index " << i << " into position " << pos << std::endl; 
            setter(pos, dl_tensor);

            // Set Type + Name String
            std::string input_string = ordered_input->input_type() + "|||" + ordered_input->name();
            // Setter is by reference, need to dynamically allocate
            char* p = new char[sizeof(input_string)];
            strcpy(p, input_string.c_str());
            setter(pos + 1, p);
            break;
          }
          pos += 2; // (arui) Hack: Every other entry is a string storing type + Name 
        }

        // tt::graphlib::NodeId node_id = std::get<0>(id_to_tensor_.at(eid));
        // Shape shape = graph_->node_by_id(node_id)->shape();
        // std::cout << "Rewriting tensor shape: (" << shape[2] << ", " << shape[3] << ") into row major" << std::endl;
        // size_t numel = shape[0] * shape[1] * shape[2] * shape[3];
        // float *row_major_data = static_cast<float*>(malloc(numel * sizeof(float)));
        // //TODO: loop over z and w
        // for (int i = 0; i < shape[2]; i ++) {
        //   for (int j = 0; j < shape[3]; j++) {
        //     row_major_data[(j * 32) + i] = data[(i * 32) + j];
        //   }
        // }

        // at::Tensor tensor = torch::from_blob(data, {shape[0], shape[1], shape[2], shape[3]}, torch::TensorOptions(torch::kFloat32));
        // std::cout << "Loaded torch tensor:" << std::endl << tensor[0][0][0][0] << ", " <<  tensor[0][0][0][1] << ", " << tensor[0][0][1][0] << ", " << tensor[0][0][1][1] << ", " << std::endl;
        // PyObject *pytensor = THPVariable_Wrap(tensor);
        // py::object pyobject = py::reinterpret_steal<py::object>(pytensor);
        // torch_tensors.push_back(pyobject);
      }
      i++;
      // setter(i, static_cast<void *>(graph_));
      values[values.size() - 1].v_handle = static_cast<void *>(graph_);
      codes[codes.size() - 1] = TVMArgTypeCode::kTVMOpaqueHandle;



      TVMRetValue rv;
      pf->CallPacked(TVMArgs(values.data(), codes.data(), values.size()), &rv);

      // TODO (arui) : Free the pointers passed to TVMArgs
      std::cout << "Returned" << std::endl;

      DLTensor *ret_value = static_cast<DLTensor *>(rv);
      size_t numel = 1;
      for (int i = 0; i < ret_value->ndim; i++) {
        numel = numel * ret_value->shape[i];
      }
      // auto shape = data_entry_[outputs_[0].id_]->shape;
      // float *row_major_data = static_cast<float*>(ret_value->data);
      // float *col_major_data = static_cast<float*>(data_entry_[outputs_[0].id_]->data);
      // for (int i = 0; i < shape[0]; i ++) {
      //   for (int j = 0; j < shape[1]; j++) {
      //     col_major_data[(j * 32) + i] = row_major_data[(i * 32) + j];
      //   }
      // }
      memcpy(data_entry_[outputs_[0].id_]->data , ret_value->data, numel * sizeof(float));
      
      // auto mod = (*pf)(graph_, input_nodes_[0]);

      // std::unordered_map<int, py::object> map = std::unordered_map<int, py::object>(); 
      // std::vector<py::object> results = tt::eval_graph(graph_, torch_tensors, map);

      // std::cout << "Size of results: " << results.size() << std::endl;
      // std::cout << "Size of outputs: " << outputs_.size() << std::endl;
      // PrintMap();
      // for (py::object result : results) {
      //   py::handle h = result;
      //   PyObject *pyobject = h.ptr();
      //   at::Tensor tensor = THPVariable_Unpack(pyobject);

      //   float *data = tensor.data<float>();
      //   // memcpy(data_entry_[output.id_]->data , col_major_data, numel * sizeof(float));
      // }

    //  for (auto output : outputs_) {
    //     std::cout << "  id: " << output.id_ << ", tesnor: " << std::get<0>(id_to_tensor_.at(output.id_)) << std::endl;

  }
 private:
  std::map <uint32_t, std::tuple<tt::graphlib::NodeId, std::string, int, Shape>> id_to_tensor_;
  Graph* graph_;

  const Shape MakeBudaShape(std::vector<int64_t> shape) {
    std::vector<int> shape_4d;
    for (auto it =  shape.rbegin(); it != shape.rend(); ++it) {
      shape_4d.push_back(static_cast<int>(*it));
    }
    int i = 0;
    while (shape_4d.size() < 4) {
      if (i++ > 10) {
        break;
      }
      shape_4d.insert(shape_4d.begin(), 1);
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
    std::cout << "BudaRuntime::Build" << std::endl;
    graph_ = new Graph();

    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      uint32_t eid = EntryID(input_nodes_[i], 0);
      auto shape = nodes_[eid].GetOpShape()[0];
      auto name = nodes_[eid].GetOpName();

      std::string input_type;
      bool requires_grad = false;
      if (std::count(const_idx_.begin(), const_idx_.end(), eid) == 0) {
        input_type = "activations";
        std::cout << "Creating activations tensor: " << name << std::endl;
      }
      else {
        requires_grad = true;
        input_type = "parameter";
        std::cout << "Creating parameter tensor: " << name << std::endl;
      }

      auto node = graph_->add_node(create_node<InputNode>(name, input_type, requires_grad));
      
      const Shape buda_shape = MakeBudaShape(shape);
      std::cout << "Tensor shape: " << buda_shape << std::endl;
      node->set_shape(buda_shape);
      
      id_to_tensor_.emplace(eid, std::make_tuple(node->id(), name, 0, buda_shape));
    }

    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        std::string op_type;
        if ("add" == op_name) {
          op_type = "add";
        } else if ("buda.matmul" == op_name) {
          op_type = "matmul";
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
        std::string buda_name = op_type + "_" + std::to_string(nid);
        auto shape = nodes_[nid].GetOpShape()[0];
        auto buda_node = graph_->add_node(create_node<OpNode>(buda_name, op_type));
        const Shape buda_shape = MakeBudaShape(shape);
        buda_node->set_shape(buda_shape);

        id_to_tensor_.emplace(nid, std::make_tuple(buda_node->id(), buda_name, 0, buda_shape));

        auto inputs = node.GetInputs();
        for (unsigned int i = 0; i < inputs.size(); i++) {
          std::cout << "Add node input: " << nodes_[inputs[i].id_].GetOpName() << " to op: " << node.GetOpName() << " in position: " << i << std::endl;
          
          Edge edge(std::get<0>(id_to_tensor_.at(inputs[i].id_)), 0, buda_node->id(), i, EdgeType::kData);
          graph_->add_edge(edge);
          
          //TODO: Better broadcast
          std::shared_ptr<EdgeAttributes> attr = graph_->get_edge_attributes(edge);

          auto input_shape = std::get<3>(id_to_tensor_.at(inputs[i].id_)).as_vector();
          for (size_t dim = 0; dim < input_shape.size(); dim++) {
            if ((input_shape[dim] == 1) && (buda_shape.as_vector()[dim] != 1)) {
              attr->set_broadcast_dim(dim, buda_node->shape()[dim]);
              std::cout << "Explicit Broadcasting: " << nodes_[inputs[i].id_].GetOpName() << " to: " << nodes_[nid].GetOpName() << " dim: " << dim << " port: " << i << std::endl;
              std::cout << "input_shape: " << std::get<3>(id_to_tensor_.at(EntryID(input_nodes_[i], 0))) << " output_shape: " << buda_shape << std::endl;
            }
          }
        }

      }
    }

    for (auto output : outputs_) {
      std::string buda_name = nodes_[output.id_].GetOpName() + "_output";
      auto buda_node = graph_->add_node(create_node<OutputNode>(buda_name));
      auto shape = nodes_[output.id_].GetOpShape()[0];
      const Shape buda_shape = MakeBudaShape(shape);
      buda_node->set_shape(buda_shape);
      
      Edge edge(std::get<0>(id_to_tensor_.at(output.id_)), 0, buda_node->id(), 0, EdgeType::kData);
      graph_->add_edge(edge);

    }
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
