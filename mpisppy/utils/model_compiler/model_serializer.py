
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


def _replace_expr_with_serialized_linear_expr( _expr, VarIDToVarIdx ):
    repn = generate_standard_repn(_expr.expr)

    assert repn.nonlinear_expr is None

    expr = LinearExpression()

    expr.constant=repn.constant
    expr.linear_coefs=np.array(repn.linear_coefs, dtype=np.double)
    expr.linear_vars=np.fromiter((VarIDToVarIdx[id(var)] for var in repn.linear_vars),
                                    count=len(repn.linear_vars), dtype=np.uint64)
    _expr.expr = expr

def _replace_serialized_linear_expr_with_linear_expr( _expr, VarIdxtoVarID ):

    expr = _expr.expr
    expr.linear_vars = [ VarIdxtoVarID[idx] for idx in expr.linear_vars ]
    #expr.linear_coefs = list(expr.linear_coefs)

def deserialize_objective(model, VarIdxtoVarID, verbose):

    obj = find_active_objective(model)
    _replace_serialized_linear_expr_with_linear_expr( obj, VarIdxtoVarID )

def serialize_objective(model, VarIDToVarIdx, verbose):

    obj = find_active_objective(model)
    _replace_expr_with_serialized_linear_expr( obj, VarIDToVarIdx )

def deserialize_expressions(model, VarIdxtoVarID, verbose):
    all_blocks = [_b for _b in model.block_data_objects(
        active=True,
        descend_into=True)]

    for block in all_blocks:
        for _expr in block.component_objects(Expression,
                                             descend_into=True):
            _replace_serialized_linear_expr_with_linear_expr( _expr, VarIdxtoVarID)

def serialize_expressions(model, VarIDToVarIdx, verbose):
    all_blocks = [_b for _b in model.block_data_objects(
        active=True,
        descend_into=True)]

    for block in all_blocks:
        for _expr in block.component_objects(Expression,
                                             descend_into=True):
            _replace_expr_with_serialized_linear_expr( _expr, VarIDToVarIdx )

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
