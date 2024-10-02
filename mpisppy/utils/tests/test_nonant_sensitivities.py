###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import math

import pyomo.environ as pyo
import mpisppy.utils.nonant_sensitivities as ns


def test_DLW_example():
    model = pyo.ConcreteModel("trivial_xy")
    model.x = pyo.Var(within=pyo.NonNegativeReals)
    model.y = pyo.Var(within=pyo.NonNegativeReals)
    model.f = pyo.Objective(expr=model.x * model.x)
    model.x_ge_y = pyo.Constraint(expr=model.x >= model.y)
    model.b = pyo.Param(initialize=10, mutable=True)
    model.y_ge_b = pyo.Constraint(expr=model.y >= model.b)

    var_source = [("x", model.x), ("y", model.y)]

    for bval in (12, 20):
        model.b.value = bval
        # feasible solution
        model.y.value = bval
        model.x.value = bval

        sensis = ns._compute_primal_sensitivities(model, var_source)

        assert math.isclose(sensis["x"], 2 * bval, rel_tol=1e-7)
        assert math.isclose(sensis["y"], 2 * bval, rel_tol=1e-7)


def _1UC_problem(demand):
    m = pyo.ConcreteModel("1UC")
    m.d = pyo.Param(initialize=demand, mutable=True)

    m.u = pyo.Var([1, 2], within=pyo.Binary)
    m.cu = pyo.Param([1, 2], initialize=lambda m, g: 2 * g, mutable=True)
    m.p = pyo.Var([1, 2], within=pyo.NonNegativeReals)

    m.pp = pyo.Set(initialize=[(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)])
    m.pl = pyo.Var(m.pp, within=pyo.NonNegativeReals)
    m.cl = pyo.Param(m.pp, initialize=lambda m, g, p: (g + g) * p, mutable=True)
    m.max_pl = pyo.Param(m.pp, initialize=lambda m, *_: 1, mutable=True)

    @m.Constraint(m.pp)
    def pl_max_constr(m, g, i):
        return m.pl[g, i] <= m.max_pl[g, i] * m.u[g]

    @m.Constraint([1, 2])
    def pl_link_constr(m, g):
        return sum(m.pl[g, :]) == m.p[g]

    @m.Constraint()
    def demand_constr(m):
        return sum(m.p[:]) == m.d

    @m.Objective()
    def objective(m):
        return sum(m.cl[g, i] * m.pl[g, i] for g, i in m.pp) + sum(
            m.cu[g] * m.u[g] for g in [1, 2]
        )

    return m


def _1ED_problem(demand):
    m = pyo.ConcreteModel("1UC")
    m.d = pyo.Param(initialize=demand, mutable=True)

    m.p = pyo.Var([1, 2], within=pyo.NonNegativeReals)

    m.pp = pyo.Set(initialize=[(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)])
    m.pl = pyo.Var(m.pp, within=pyo.NonNegativeReals)
    m.cl = pyo.Param(m.pp, initialize=lambda m, g, p: (g + g) * p, mutable=True)
    m.max_pl = pyo.Param(m.pp, initialize=lambda m, *_: 1, mutable=True)

    @m.Constraint(m.pp)
    def pl_max_constr(m, g, i):
        return m.pl[g, i] <= m.max_pl[g, i]

    @m.Constraint([1, 2])
    def pl_link_constr(m, g):
        return sum(m.pl[g, :]) == m.p[g]

    @m.Constraint()
    def demand_constr(m):
        return sum(m.p[:]) == m.d

    @m.Objective()
    def objective(m):
        return sum(m.cl[g, i] * m.pl[g, i] for g, i in m.pp)

    return m


def test_1UC_problem():
    m = _1UC_problem(3.5)
    m.u.domain = pyo.UnitInterval
    # solve the relaxed problem
    result = pyo.SolverFactory("cbc").solve(m)
    pyo.assert_optimal_termination(result)

    result = ns._compute_primal_sensitivities(m, ((u.name, u) for u in m.u.values()))

    assert math.isclose(result["u[1]"], -8.0, abs_tol=1e-6)
    assert math.isclose(result["u[2]"], 0.0, abs_tol=1e-6)


def test_1ED_problem():
    m = _1ED_problem(3.5)

    result = pyo.SolverFactory("cbc").solve(m)
    pyo.assert_optimal_termination(result)

    result = ns._compute_primal_sensitivities(
        m, ((pl.name, pl) for pl in m.pl.values())
    )

    assert math.isclose(result["pl[1,1]"], -6, abs_tol=1e-6)
    assert math.isclose(result["pl[1,2]"], -4, abs_tol=1e-6)
    assert math.isclose(result["pl[2,1]"], -4, abs_tol=1e-6)
    assert math.isclose(result["pl[2,3]"], 4, abs_tol=1e-6)
