#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("compile_block_linear_constraints",)

import time
import logging
import array
from weakref import ref as weakref_ref

from pyomo.core.base.set_types import Any
from pyomo.core.base import (SortComponents,
                             Var)
from pyomo.core.base.numvalue import (is_fixed,
                                      value,
                                      ZeroConstant)
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.constraint import (Constraint,
                                        IndexedConstraint,
                                        SimpleConstraint,
                                        _ConstraintData)
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.base.matrix_constraint import MatrixConstraint
from pyomo.repn import generate_standard_repn

from six import iteritems, PY3
from six.moves import xrange

if PY3:
    from collections.abc import Mapping as collections_Mapping
else:
    from collections import Mapping as collections_Mapping

def _label_bytes(x):
    if x < 1e3:
        return str(x)+" B"
    if x < 1e6:
        return str(x / 1.0e3)+" KB"
    if x < 1e9:
        return str(x / 1.0e6)+" MB"
    return str(x / 1.0e9)+" GB"

#
# Compile a Pyomo constructed model in-place, storing the compiled
# sparse constraint object on the model under constraint_name.
#
def compile_block_linear_constraints(parent_block,
                                     constraint_name,
                                     skip_trivial_constraints=False,
                                     single_precision_storage=False,
                                     verbose=False,
                                     descend_into=True):

    if verbose:
        print("")
        print("Compiling linear constraints on block with name: %s"
              % (parent_block.name))

    if not parent_block.is_constructed():
        raise RuntimeError(
            "Attempting to compile block '%s' with unconstructed "
            "component(s)" % (parent_block.name))

    #
    # Linear MatrixConstraint in CSR format
    #
    A_data = []
    A_indices = []
    A_indptr = []
    LowerBounds = []
    UpperBounds = []
    Vars = []

    def _get_bound(exp):
        if exp is None:
            return None
        if is_fixed(exp):
            return value(exp)
        raise ValueError("non-fixed bound: " + str(exp))

    start_time = time.time()
    if verbose:
        print("Sorting active blocks...")

    sortOrder = SortComponents.indices | SortComponents.alphabetical
    all_blocks = [_b for _b in parent_block.block_data_objects(
        active=True,
        sort=sortOrder,
        descend_into=descend_into)]

    stop_time = time.time()
    if verbose:
        print("Time to sort active blocks: %.2f seconds"
              % (stop_time-start_time))

    start_time = time.time()
    if verbose:
        print("Collecting variables on active blocks...")

    #
    # First Pass: assign each variable a deterministic id
    #             (an index in a list)
    #
    Vars = []
    for block in all_blocks:
        Vars.extend(
            block.component_data_objects(Var,
                                         sort=sortOrder,
                                         descend_into=False))

    VarIDToVarIdx = { id(vardata) : index for index, vardata in enumerate(Vars) }

    stop_time = time.time()
    if verbose:
        print("Time to collect variables on active blocks: %.2f seconds"
              % (stop_time-start_time))

    start_time = time.time()
    if verbose:
        print("Compiling active linear constraints...")

    #
    # Second Pass: collect and remove active linear constraints
    #
    constraint_data_to_remove = []
    empty_constraint_containers_to_remove = []
    constraint_containers_to_remove = []
    constraint_containers_to_check = set()
    referenced_variable_symbols = set()
    nnz = 0
    nrows = 0
    SparseMat_pRows = [0]
    for block in all_blocks:

        if hasattr(block, '_repn'):
            del block._repn

        for constraint in block.component_objects(Constraint,
                                                  active=True,
                                                  sort=sortOrder,
                                                  descend_into=False):

            assert not isinstance(constraint, MatrixConstraint)

            if len(constraint) == 0:

                empty_constraint_containers_to_remove.append((block, constraint))

            else:

                singleton = isinstance(constraint, SimpleConstraint)

                # Note that as we may be removing items from the _data
                # dictionary, we need to make a copy of the items list
                # before iterating:
                for index, constraint_data in list(iteritems(constraint)):

                    if constraint_data.body.__class__ in native_numeric_types or constraint_data.body.polynomial_degree() <= 1:

                        # collect for removal
                        if singleton:
                            constraint_containers_to_remove.append((block, constraint))
                        else:
                            constraint_data_to_remove.append((constraint, index))
                            constraint_containers_to_check.add((block, constraint))

                        repn = generate_standard_repn(constraint_data.body)

                        assert repn.nonlinear_expr is None

                        row_variable_symbols = []
                        row_coefficients = []
                        if len(repn.linear_vars) == 0:
                            if skip_trivial_constraints:
                                continue
                        else:
                            row_variable_indices = \
                                [VarIDToVarIdx[id(vardata)]
                                 for vardata in repn.linear_vars]
                            assert repn.linear_coefs is not None
                            row_coefficients = repn.linear_coefs

                        A_indptr.append(A_indptr[-1] + len(row_variable_symbols))
                        A_indices.extend(row_variable_indices)
                        A_data.extend(row_coefficients)

                        nnz += len(row_variable_symbols)
                        nrows += 1

                        L = _get_bound(constraint_data.lower)
                        U = _get_bound(constraint_data.upper)
                        constant = value(repn.constant)

                        if L is None:
                            LowerBounds.append(None)
                        else:
                            LowerBounds.append(L - constant)

                        if U is None:
                            UpperBounds.append(None)
                        else:
                            UpperBounds.append(U - constant)

                        # Start freeing up memory
                        constraint[index] = Constraint.Skip

    stop_time = time.time()
    if verbose:
        print("Time to compile active linear constraints: %.2f seconds"
              % (stop_time-start_time))

    start_time = time.time()
    if verbose:
        print("Removing compiled constraint objects...")

    #
    # Remove compiled constraints
    #
    constraints_removed = 0
    constraint_containers_removed = 0
    for block, constraint in empty_constraint_containers_to_remove:
        block.del_component(constraint)
        constraint_containers_removed += 1
    for constraint, index in constraint_data_to_remove:
        # Note that this del is not needed: assigning Constraint.Skip
        # above removes the _ConstraintData from the _data dict.
        #del constraint[index]
        constraints_removed += 1
    for block, constraint in constraint_containers_to_remove:
        block.del_component(constraint)
        constraints_removed += 1
        constraint_containers_removed += 1
    for block, constraint in constraint_containers_to_check:
        if len(constraint) == 0:
            block.del_component(constraint)
            constraint_containers_removed += 1

    stop_time = time.time()
    if verbose:
        print("Eliminated %s constraints and %s Constraint container objects"
              % (constraints_removed, constraint_containers_removed))
        print("Time to remove compiled constraint objects: %.2f seconds"
              % (stop_time-start_time))

    parent_block.add_component(constraint_name,
                               MatrixConstraint(A_data,
                                                A_indices,
                                                A_indptr,
                                                LowerBounds,
                                                UpperBounds,
                                                Vars))
