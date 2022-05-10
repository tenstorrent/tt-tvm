import os
import json

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr_functor import ExprVisitor, ExprMutator

def get_default_reportify_path(test_name):
    home = os.environ["HOME"]
    return home + "/testify/ll-sw/" + test_name

def initialize_directory(path, test_name):
    filename = path + "/" + "summary.yaml"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"content:\n")
        f.write(f"  name: ll-sw.{test_name}\n")
        f.write(f"  output-dir: {path}\n")
        f.write(f"type: summary\n")

def get_tvm_reports_relpath():
    return "/tvm_reports/Passes/"

class CreateJson(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.node_idx = 0
        self.graph = {}
        self.graph["graph"] = {}
        self.graph["nodes"] = {}

        self.node_map = {}

    def get_default_op(self, name):
        op = {}
        op["name"] = name
        op["epoch"] = 0
        op["ir"] = "pybuda"
        op["unique_id"] = self.node_idx
        op["input_nodes"] = []
        op["output_nodes"] = []
        op["pybuda"] = 1
        self.graph["nodes"][name] = op
        self.node_idx += 1
        return op

    def visit_call(self, call):
        name = f"{call.op.name}_{self.node_idx}"
        op = self.get_default_op(name)
        if isinstance(call.checked_type, tvm.ir.type.TupleType):
            shape = []
            for field in call.checked_type.fields:
                shape.append([int(dim) for dim in field.shape])
            op["cache"] = {"shape": shape}
        else:
            op["cache"] = {"shape": [int(dim) for dim in call.checked_type.shape]}
        op["class"] = call.op.name
        op["type"] = call.op.name
        op["opcode"] = "RelayOp"
        self.node_map[call] = name
        return super().visit_call(call)

    def visit_var(self, call):
        name = call.name_hint
        op = self.get_default_op(name)
        op["cache"] = {"shape": [int(dim) for dim in call.checked_type.shape]}
        op["opcode"] = "Input"
        op["class"] = "Input::"
        op["type"] = "Input::input"
        self.node_map[call] = name
        return super().visit_var(call)


    def visit_constant(self, const):
        name = f"constant_{self.node_idx}"
        op = self.get_default_op(name)
        op["cahce"] = {"shape": [int(dim) for dim in const.checked_type.shape]}
        op["opcode"] = "Input"
        op["class"] = "Input::"
        op["type"] = "Input::constant"
        self.node_map[const] = name
        return super().visit_constant(const)

    def visit_tuple_getitem(self, t):
        if isinstance(t.tuple_value, tvm.relay.expr.Tuple):
            op_type = "tuple"
        else:
            op_type = t.tuple_value.op.name
        name = f"{op_type}_{self.node_idx}"
        op = self.get_default_op(name)
        op["cache"] = {"shape": [int(dim) for dim in t.checked_type.shape]}
        op["class"] = op_type
        op["type"] = op_type
        op["opcode"] = "RelayOp"
        self.node_map[t] = name
        return super().visit_tuple_getitem(t)

    def visit_tuple(self, tup):
        name = f"tuple_{self.node_idx}"
        op = self.get_default_op(name)
        shape = []
        for field in tup.checked_type.fields:
            shape.append([int(dim) for dim in field.shape])

        op["cache"] = {"shape": shape}
        op["class"] = "tuple"
        op["type"] = "tuple"
        op["opcode"] = "RelayOp"
        self.node_map[tup] = name
        return super().visit_tuple(tup)


def convert_serialized_tvm_to_reportify_graph(mod):
    json_creator = CreateJson()
    json_creator.visit(mod["main"])

    graph = json_creator.graph
    # because expr visitor does not operate in topologial order, link all
    # the nodes after they are visited
    for node, name in json_creator.node_map.items():
        if not hasattr(node, "args"):
            continue
        for in_node in node.args:
            graph["nodes"][name]["input_nodes"].append(json_creator.node_map[in_node])
            graph["nodes"][json_creator.node_map[in_node]]["output_nodes"].append(name)

    return json.dumps(graph, indent=4, sort_keys=True)


def dump_graph(mod, test_name, stage):
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    tvm_graph = convert_serialized_tvm_to_reportify_graph(mod)
    path = get_default_reportify_path(test_name)
    initialize_directory(path, test_name)
    tvm_subdir = path + get_tvm_reports_relpath()
    os.makedirs(tvm_subdir + stage + "_graphs", exist_ok=True)

    filename = tvm_subdir + stage + ".buda"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(tvm_graph)
        
    # assert(False)