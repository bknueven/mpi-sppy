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
from pyomo.contrib.pynumero.linalg.scipy_interface import ScipyLU

from mpisppy.utils.kkt.interface import InteriorPointInterface


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


def _compute_primal_sensitivities(model, var_source):
    """Compute the sensitivities of the vars in var_source
    (w.r.t the Lagrangian for mdoel)
    Args:
        model: Pyomo ConcreteModel
        var_source: A generator of (keys, pyomo variables) for which
            the sensitivities are computed.
    Returns:
        primal_sensitivities (dict): (key, sensitivity) for each var in var_source.
    """
    # first, solve the subproblems with Ipopt,
    # and gather sensitivity information
    ipopt = pyo.SolverFactory("ipopt")

    # add the needed suffixes / remove later
    # TODO: keep the suffix and restore its value if it already exists
    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    results = ipopt.solve(model)
    pyo.assert_optimal_termination(results)

    kkt_builder = InteriorPointInterface(model)
    kkt_builder.set_barrier_parameter(1e-9)
    kkt_builder.set_bounds_relaxation_factor(1e-8)
    # rhs = kkt_builder.evaluate_primal_dual_kkt_rhs()
    # print(f"{rhs}")
    # print(f"{rhs.flatten()}")
    kkt = kkt_builder.evaluate_primal_dual_kkt_matrix()

    # print(f"{kkt=}")
    # could do better than SuperLU
    kkt_lu = ScipyLU()
    # always regularize equality constraints
    kkt_builder.regularize_equality_gradient(kkt=kkt, coef=-1e-8, copy_kkt=False)
    # always regularize the Hessian too
    kkt_builder.regularize_hessian(kkt=kkt, coef=1e-8, copy_kkt=False)
    kkt_lu.do_numeric_factorization(kkt, raise_on_error=True)

    grad_vec = np.zeros(kkt.shape[1])
    grad_vec[0 : kkt_builder._nlp.n_primals()] = (
        kkt_builder._nlp.evaluate_grad_objective()
    )

    grad_vec_kkt_inv = kkt_lu._lu.solve(grad_vec, "T")

    primal_sensitivities = {}
    for key, var in var_source:
        var_idx = kkt_builder._nlp._vardata_to_idx[var]

        y_vec = np.zeros(kkt.shape[0])
        y_vec[var_idx] = 1.0

        x_denom = y_vec.T @ kkt_lu._lu.solve(y_vec)
        x = -1 / x_denom
        e_x = x * y_vec

        primal_sensitivities[key] = grad_vec_kkt_inv @ -e_x

    del model.ipopt_zL_out
    del model.ipopt_zU_out
    del model.dual

    return primal_sensitivities
