###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import numpy as np

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces import pyomo_nlp
from pyomo.contrib.pynumero.linalg.scipy_interface import ScipyLU

from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

def nonant_sensitivities(s, ph):
    """Compute the sensitivities of noants (w.r.t. the Lagrangian for s)
    Args:
        s: (Pyomo ConcreteModel): the scenario
       ph: (PHBase Object): to deal with bundles (that are not proper)
    Returns:
        nonant_sensis (dict): [ndn_i]: sensitivity for the Var
    """

    solution_cache = pyo.ComponentMap()
    for var in s.component_data_objects(pyo.Var):
        solution_cache[var] = var._value

    relax_int = pyo.TransformationFactory("core.relax_integer_vars")
    relax_int.apply_to(s)
    assert hasattr(s, "_relaxed_integer_vars")

    nonant_sensis = _compute_primal_sensitivities(s, _ph_var_source(s, ph))

    relax_int.apply_to(s, options={"undo": True})
    assert not hasattr(s, "_relaxed_integer_vars")
    for var, val in solution_cache.items():
        var._value = val

    return nonant_sensis


def _ph_var_source(s, ph):
    # bundles?
    for scenario_name in s.scen_list:
        for ndn_i, v in ph.local_scenarios[
            scenario_name
        ]._mpisppy_data.nonant_indices.items():
            yield (ndn_i, v)


def _compute_primal_sensitivities(model, var_source, active_constr_tol=1e-6):
    """Compute the sensitivities of the vars in var_source
    (w.r.t the Lagrangian for mdoel)
    Args:
        model: Pyomo ConcreteModel
        var_source: A generator of (keys, pyomo variables) for which
            the sensitivities are computed.
    Returns:
        primal_sensitivities (dict): (key, sensitivity) for each var in var_source.
    """

    slack_suffix = model.component("slack")
    if not slack_suffix or slack_suffix.ctype is not pyo.Suffix:
        raise RuntimeError(f"_compute_primal_sensitivities needs a slack Suffix")

    nlp = pyomo_nlp.PyomoNLP(model, nl_file_options={'skip_trivial_constraints': True})

    x = nlp.init_primals()
    nlp.set_primals(x)
    jac = nlp.evaluate_jacobian().tocsr()

    # equality constraints are always active
    row_indices = list(nlp._condata_to_eq_idx.values())
    # get the active inequality constraints
    for c, i in nlp._condata_to_ineq_idx.items():
        if abs(slack_suffix[c]) <= active_constr_tol:
            row_indices.append(i)

    jac = jac[row_indices]

    nx = jac.shape[1]
    assert nx == nlp.n_primals()
    nc = jac.shape[0]
    kkt = BlockMatrix(2,2)

    kkt.set_block(0, 0, identity(nx))
    kkt.set_block(1, 0, -jac)
    kkt.set_block(0, 1, jac.transpose())
    kkt.set_block(1, 1, -1e-08 * identity(nc))

    kkt_lu = ScipyLU()
    kkt_lu.do_numeric_factorization(kkt, raise_on_error=True)

    grad_vec = np.zeros(kkt.shape[1])
    grad_vec[0 : nx] = (
        nlp.evaluate_grad_objective()
    )

    grad_vec_kkt_inv = kkt_lu._lu.solve(grad_vec, "T")

    primal_sensitivities = {}
    for key, var in var_source:
        var_idx = nlp._vardata_to_idx[var]

        y_vec = np.zeros(kkt.shape[0])
        y_vec[var_idx] = 1.0

        x_denom = y_vec.T @ kkt_lu._lu.solve(y_vec)
        x = -1 / x_denom
        e_x = x * y_vec

        primal_sensitivities[key] = grad_vec_kkt_inv @ -e_x

    return primal_sensitivities
