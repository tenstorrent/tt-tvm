from tvm.relay import ExprVisitor
from tvm.relay import ExprVisitor
from tvm import relay
import tvm
import numpy as np
from tvm.relay.expr_functor import Function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.frontend.pytorch import _infer_type as infer_type


def get_op_info_0(expr, params=None):
    class ModelInspector(ExprVisitor):
        def __init__(self, params):
            super().__init__()
            self.inputs = []
            self.params = []
            self.constants = []
            self.constant_details = []  # To store details of constants
            self.ops = []
            self.call_details = []
            self.params_dict = params or {}  # Dictionary of parameters
            self.param_details = {}  # To store detailed info on parameters
            self.param_usage = {}  # Track which params are used by each op
            self.constant_usage = {}  # Track which ops use each constant

            # Populate initial parameter details from params dictionary
            for param_name, param_value in self.params_dict.items():
                self.param_details[param_name] = {
                    "shape": param_value.shape,
                    "dtype": param_value.dtype
                }
                self.param_usage[param_name] = []

        def visit_var(self, var):
            # Check if variable is a parameter using params dictionary
            if var.name_hint in self.params_dict:
                self.params.append(var.name_hint)
            else:
                self.inputs.append(var.name_hint)
            super().visit_var(var)

        def visit_constant(self, const):
            # Add constant as model constant with detailed info
            constant_data = const.data.asnumpy()  # Assuming const.data is a TVM tensor
            constant_info = {
                "value": constant_data,
                "dtype": const.data.dtype,
                "shape": constant_data.shape
            }
            self.constants.append(constant_data)
            self.constant_details.append(constant_info)  # Store details for each constant
            
            # Initialize constant usage with shape and empty usage list
            self.constant_usage[constant_data.tobytes()] = {
                "shape": constant_data.shape,
                "usage": []  # Store the list of operations that use this constant
            }
            super().visit_constant(const)

        def visit_call(self, call):
            # Extract operator details
            op_name = call.op.name if hasattr(call.op, "name") else str(call.op)

            # Extract input shapes and check parameter and constant usage
            input_shapes = []
            used_params = []
            used_constants = []
            for arg in call.args:
                if isinstance(arg, relay.expr.Expr):
                    try:
                        inferred_type = infer_type(arg)
                        shape = inferred_type.checked_type.shape
                        input_shapes.append(tuple(int(dim) for dim in shape))
                    except Exception:
                        input_shapes.append("Unknown")
                    
                    # Track if this argument is a parameter
                    if hasattr(arg, "name_hint") and arg.name_hint in self.params_dict:
                        used_params.append(arg.name_hint)
                        self.param_usage[arg.name_hint].append(op_name)
                    # Track if this argument is a constant
                    elif isinstance(arg, relay.Constant):
                        const_data = arg.data.asnumpy()
                        const_key = const_data.tobytes()  # Store bytes representation
                        used_constants.append(const_key)  # Keep the key for reference
                        
                        # Ensure the constant usage entry exists
                        if const_key not in self.constant_usage:
                            self.constant_usage[const_key] = {
                                "shape": const_data.shape,
                                "usage": []  # Initialize the usage list
                            }

                        # Append operation name to the usage list in constant_usage
                        self.constant_usage[const_key]["usage"].append(op_name)

                        # Add the constant's shape to the input shapes
                        # Handle the scalar case correctly
                        if const_data.ndim == 0:  # Scalar case
                            input_shapes.append(())  # Append an empty tuple for scalar
                        else:
                            input_shapes.append(tuple(int(dim) for dim in const_data.shape))

            # Extract attributes if available
            attrs = {}
            if hasattr(call, "attrs") and call.attrs is not None:
                try:
                    if hasattr(call.attrs, "keys"):
                        for attr_name in call.attrs.keys():
                            attrs[attr_name] = call.attrs[attr_name]
                    else:
                        for attr_name in dir(call.attrs):
                            if not attr_name.startswith("_"):
                                attrs[attr_name] = getattr(call.attrs, attr_name)
                except Exception as e:
                    print(f"Could not retrieve attributes for {op_name}: {e}")

            # Append all operator calls with details
            call_entry = {
                "Operator": op_name,
                "Input Shapes": input_shapes,
                "Attributes": dict(attrs),
                "Used Params": used_params,
                "Used Constants": used_constants  # Track used constants
            }
            self.call_details.append(call_entry)

            # Record the operation for summary purposes
            self.ops.append(op_name)

            super().visit_call(call)

        def visit_function(self, func):
            # Extract model inputs and output/return information
            for param in func.params:
                self.visit(param)
            super().visit_function(func)

        def get_model_info(self):
            # Return a comprehensive dictionary of model details
            return {
                "inputs": self.inputs,
                "params": self.params,
                "constants": self.constants,
                "constant_details": self.constant_details,  # Detailed constant info
                "ops": self.ops,
                "call_details": self.call_details,
                "param_details": self.param_details,  # Detailed parameter info
                "param_usage": self.param_usage,        # Which ops use each param
                "constant_usage": self.constant_usage     # Which ops use each constant
            }

    # Run inspection
    inspector = ModelInspector(params)
    inspector.visit(expr)



    # Return the gathered details
    return inspector.get_model_info()