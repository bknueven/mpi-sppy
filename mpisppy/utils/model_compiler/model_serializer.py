
import numpy as np

from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.environ import Objective, Param, Set, Expression

from mpisppy.utils.sputils import find_active_objective
from .model_transformation import compile_block_linear_constraints

def serialize_model(model, verbose=True, keep_expressions=True):

    VarIDToVarIdx = compile_block_linear_constraints(model,
                                     '_serialized_constraints',
                                     verbose=verbose)

    serialize_objective(model, VarIDToVarIdx,
                      verbose=verbose)

    if keep_expressions:
        serialize_expressions(model, VarIDToVarIdx,
                            verbose)

    remove_other_componenets(model, verbose, keep_expressions)

def deserialize_model(model, verbose=True):

    VarIdxtoVarID = model._serialized_constraints._x

    deserialize_objective(model, VarIdxtoVarID, verbose)
    deserialize_expressions(model, VarIdxtoVarID, verbose)
    deserialize_constraints(model, verbose)

def deserialize_constraints(model, verbose):
    serialized_constraints = model._serialized_constraints

    serialized_constraints._A_data = serialized_constraints._A_data.tolist()
    serialized_constraints._A_indices = serialized_constraints._A_indices.tolist()
    serialized_constraints._A_indptr = serialized_constraints._A_indptr.tolist()
    serialized_constraints._lower = serialized_constraints._lower.tolist()
    serialized_constraints._upper = serialized_constraints._upper.tolist()

def _replace_expr_with_linear_expr( _expr):
    repn = generate_standard_repn(_expr.expr)

    assert repn.nonlinear_expr is None

    expr = LinearExpression()

    expr.constant=repn.constant
    expr.linear_coefs=repn.linear_coefs
    expr.linear_vars=repn.linear_vars

    _expr.expr = expr

def _replace_linear_expr_with_serialized_linear_expr( _expr, VarIDToVarIdx ):

    expr = _expr.expr
    expr.linear_coefs=np.array(expr.linear_coefs, dtype=np.double)
    expr.linear_vars=np.fromiter((VarIDToVarIdx[id(var)] for var in expr.linear_vars),
                                    count=len(expr.linear_vars), dtype=np.uint64)

def _replace_serialized_linear_expr_with_linear_expr( _expr, VarIdxtoVarID ):

    expr = _expr.expr
    expr.linear_vars = [ VarIdxtoVarID[idx] for idx in expr.linear_vars ]
    expr.linear_coefs = expr.linear_coefs.tolist()

def deserialize_objective(model, VarIdxtoVarID, verbose):

    obj = find_active_objective(model)
    _replace_serialized_linear_expr_with_linear_expr( obj, VarIdxtoVarID )

def serialize_objective(model, VarIDToVarIdx, verbose):

    obj = find_active_objective(model)
    _replace_expr_with_linear_expr( obj )
    _replace_linear_expr_with_serialized_linear_expr( obj, VarIDToVarIdx )

def deserialize_expressions(model, VarIdxtoVarID, verbose):
    all_blocks = [_b for _b in model.block_data_objects(
        active=True,
        descend_into=True)]

    for block in all_blocks:
        for _expr in block.component_data_objects(Expression,
                                             descend_into=False):
            _replace_serialized_linear_expr_with_linear_expr( _expr, VarIdxtoVarID)

def serialize_expressions(model, VarIDToVarIdx, verbose):
    all_blocks = [_b for _b in model.block_data_objects(
        active=True,
        descend_into=True)]

    for block in all_blocks:
        for _expr in block.component_data_objects(Expression,
                                             descend_into=False):
            _replace_expr_with_linear_expr( _expr )

    for block in all_blocks:
        for _expr in block.component_data_objects(Expression,
                                             descend_into=False):
            _replace_linear_expr_with_serialized_linear_expr( _expr, VarIDToVarIdx )

def remove_other_componenets(model, verbose, keep_expressions):
    
    all_blocks = [_b for _b in model.block_data_objects(
        active=True,
        descend_into=True)]

    ctypes_to_remove = [Param, Set]
    if not keep_expressions:
        ctypes_to_remove.append(Expression)

    components_to_remove = []
    for block in all_blocks:
        for ctype in ctypes_to_remove:
            for _comp in block.component_objects(ctype,
                                                 descend_into=False):
                components_to_remove.append((block,_comp))

    for block, comp in components_to_remove:
        block.del_component(comp)
