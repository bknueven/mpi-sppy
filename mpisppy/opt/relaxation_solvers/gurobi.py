###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import math
from pyomo.environ import SolverFactory
from pyomo.opt.results.solution import SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent, gurobipy

@SolverFactory.register(
    "mip-dual-gurobi",
    doc="GurobiPersistent, but load the best relaxation solution from the root MIP node",
)
class GurobiRelaxationSolver(GurobiPersistent):
    def make_root_callback(self):
        def get_root_node_solution(gurobi_model, where):
            if where == gurobipy.GRB.Callback.MIP:
                self._best_bound = gurobi_model.cbGet(gurobipy.GRB.Callback.MIP_OBJBND)
            if where == gurobipy.GRB.Callback.MIPSOL:
                if gurobi_model.cbGet(gurobipy.GRB.Callback.MIPSOL_NODCNT) != 0:
                    # TODO: when  to terminate 
                    gurobi_model.terminate()
                sol_val = self._multiplier * gurobi_model.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJ)
                best_incumbent = self._multiplier * gurobi_model.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJBST)
                self._best_bound = gurobi_model.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJBND)
                if sol_val < best_incumbent:
                    self._incumbent_value = sol_val
                    self._incumbent_solution = gurobi_model.cbGetSolution(self._gurobipy_var_list)
                return
            if where == gurobipy.GRB.Callback.MIPNODE:
                if gurobi_model.cbGet(gurobipy.GRB.Callback.MIPNODE_NODCNT) != 0:
                    # TODO: when  to terminate 
                    gurobi_model.terminate()
                status = gurobi_model.cbGet(gurobipy.GRB.Callback.MIPNODE_STATUS)
                if status != gurobipy.GRB.OPTIMAL:
                    return
                self._relaxation_solution = gurobi_model.cbGetNodeRel(self._gurobipy_var_list)
                self._relaxation_value = gurobi_model.cbGet(gurobipy.GRB.Callback.MIPNODE_OBJBND)
                # print(f"{self._relaxation_value=}")
                return
        return get_root_node_solution

    def set_gurobi_callback(self):
        self._gurobipy_var_list = list(self._solver_var_to_pyomo_var_map.keys())
        # use the existing callback setting in GurobiDirect
        self._callback = self.make_root_callback()
        self._relaxation_solution = None
        self._relaxation_value = None
        self._incumbent_solution = None
        self._incumbent_value = None
        self._best_bound = None
        self._multiplier = self._solver_model.ModelSense

    def _presolve(self, **kwds):
        # disable warmstart
        kwds.pop("warmstart", None)
        super()._presolve(**kwds)

    def _apply_solver(self):
        gurobi_model = self._solver_model
        if self._needs_updated:
            self._update()
        if not gurobi_model.IsMIP:
            return super()._apply_solver()
        self.set_gurobi_callback()
        # turn off heuristics
        if self.options.Heuristics is None:
            self.options.Heuristics = 0.0
        # don't process MIP Starts
        if self.options.StartNodeLimit is None:
            self.options.StartNodeLimit = -3
        # never go beyond the root node
        self.options.NodeLimit = 1
        if self.options.PreQLinearize is None:
            # This turns off PreQLinearize, which will linearize quadratic terms
            # containing interger values. This is a good idea for the primal
            # but makes the relaxation invalid for the proximal term
            self.options.PreQLinearize = 0
        # TODO: is this a good idea?
        # if self.options.Cuts is None:
        #     self.options.Cuts = 3
        # TODO: consider giving the user better
        #       direct control over the number
        #       of cut passes
        # if self.options.CutPasses is None:
        #     self.options.CutPasses = 50
        return super()._apply_solver()

    def _postsolve(self):
        gprob = self._solver_model
        if not gprob.IsMIP:
            return super()._postsolve()
        # overwrite some attributes
        if self._save_results:
            self._save_results = False
            print(
                f"WARNING: Not saving results. Use {self.__class__.__name__}.load_vars() to load a relaxed primal solution"
            )
        load_solutions = False
        if self._load_solutions:
            # don't load the solution using
            # the base class method; use
            # our own method below
            self._load_solutions = False
            load_solutions = True
        ret = super()._postsolve()
        # Sometimes Gurobi might not finish processing the root node before cutoff
        # If a solution is cut off in the root node then is a CH solution 
        if self._incumbent_value is not None and math.isclose(self._incumbent_value, self._best_bound, abs_tol=1e-6):
            self._relaxation_solution = self._incumbent_solution
            self._relaxation_value = self._incumbent_value
        if self._relaxation_solution is None and self._incumbent_solution is not None:
            self._relaxation_solution = self._incumbent_solution
            self._relaxation_value = self._incumbent_value
        if self._relaxation_solution is not None:
            # print(f"{self._relaxation_value=}")
            self.results.problem.number_of_solutions = 1
            if load_solutions:
                self._load_solutions = True
                self._load_vars()
            # since we don't add *all* the cuts, this
            # warning message is appropriate.
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Unable to satisfy optimality tolerances; a sub-optimal "
                "solution is available."
            )
            self.results.solver.termination_condition = TerminationCondition.other
            self.results.solution(0).status = SolutionStatus.feasible
        return ret

    def _load_vars(self, vars_to_load=None):
        gprob = self._solver_model
        if not gprob.IsMIP:
            return super()._load_vars(vars_to_load=vars_to_load)
        assert self._relaxation_solution is not None
        if vars_to_load is not None:
            print("WARNING: loading all the variable values")
        # get all the variables
        gurobi_vars = self._gurobipy_var_list
        # now zip(xpress_vars, self._relaxation_solution)
        # as the xpress_var, sol_val pairs
        var_map = self._solver_var_to_pyomo_var_map
        ref_vars = self._referenced_variables
        assert len(gurobi_vars) == len(self._relaxation_solution)
        for gp_var, val in zip(gurobi_vars, self._relaxation_solution):
            pyo_var = var_map[gp_var]
            if ref_vars[pyo_var] > 0:
                pyo_var.set_value(val, skip_validation=True)
        # for v in self._pyomo_model._mpisppy_data.nonant_indices.values():
        #     print(v.name, v.value)
        # raise RuntimeError

    def _load_rc(self, vars_to_load=None):
        gprob = self._solver_model
        if not gprob.IsMIP:
            return super()._load_rc(vars_to_load=vars_to_load)
        print("Cannot extract reduced costs from relaxed MIP solution")

    def _load_duals(self, cons_to_load=None):
        gprob = self._solver_model
        if not gprob.IsMIP:
            return super()._load_duals(cons_to_load=cons_to_load)
        print("Cannot extract duals from relaxed MIP solution")

    def _load_slacks(self, cons_to_load=None):
        gprob = self._solver_model
        if not gprob.IsMIP:
            return super()._load_slacks(cons_to_load=cons_to_load)
        print("Cannot extract slacks from relaxed MIP solution")
