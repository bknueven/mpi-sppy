###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
from pyomo.environ import SolverFactory
from pyomo.opt.results.solution import SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.xpress_persistent import XpressPersistent


@SolverFactory.register(
    "mip-dual-xpress",
    doc="XpressPersistent, but load the best relaxation solution from the root MIP node"
)
class XpressRelaxationSolver(XpressPersistent):

    @staticmethod
    def get_root_node_solution(xpress_problem, self):
        print("here!")
        # when the root node is finished processing
        #node = xpress_problem.attributes.currentnode
        #print("NodeOptimal: node number", node)
        #objval = xpress_problem.attributes.lpobjval
        #print("Objective function value =", objval)
        # TODO: could get duals, rc, and slacks too
        sol = []
        xpress_problem.getlpsol(x=sol)
        self._relaxation_solution = sol

        # print(f"{self._relaxation_solution=}")

        # TODO: put solution into self._relaxation_solution
        # returning 1 tells the solver to terminate
        return 1

    def set_xpress_callback(self):
        xpress_problem = self._solver_model
        xpress_problem.addcboptnode(XpressRelaxationSolver.get_root_node_solution, self, 0)
        self._relaxation_solution = None

    def _apply_solver(self):
        self.set_xpress_callback()
        # turn off heuristics
        if self.options.heuremphasis is None:
            self.options.heuremphasis = 0
        return super()._apply_solver()

    # TODO
    def _get_mip_results(self, results, soln):
        """Sets up `results` and `soln` and returns whether there is a solution
        to query.
        Returns `True` if a feasible solution is available, `False` otherwise.
        """
        # overwrite some attributes
        if self._save_results:
            self._save_results = False
            print(f"WARNING: Not saving results. Use {self.__class__.__name__}.load_vars() to load a relaxed primal solution")
        load_solutions = False 
        if self._load_solutions:
            # don't load the solution using
            # the base class method; use
            # our own methoe
            self._load_solutions = False
            load_solutions = True 
        # loads the bounds, which are important
        super()._get_mip_results(results, soln)
        if self._relaxation_solution is not None:
            if load_solutions:
                self._load_solutions = True
                self._load_vars()
            # since we don't add *all* the cuts, this
            # warning message is appropriate.
            results.solver.status = SolverStatus.warning
            results.solver.termination_message = (
                "Unable to satisfy optimality tolerances; a sub-optimal "
                "solution is available."
            )
            results.solver.termination_condition = TerminationCondition.other
            soln.status = SolutionStatus.feasible
            return True
        return False

    def _postsolve(self):
        ret = super()._postsolve()
        if self._relaxation_solution is not None:
            self.results.problem.number_of_solutions = 1
        return ret

    def _load_vars(self, vars_to_load=None):
        assert self._relaxation_solution is not None
        if vars_to_load is not None:
            print("WARNING: loading all the variable values")
        # get all the variables
        xpress_vars = self._solver_model.getVariable()
        # now zip(xpress_vars, self._relaxation_solution)
        # as the xpress_var, sol_val pairs
        var_map = self._solver_var_to_pyomo_var_map
        ref_vars = self._referenced_variables
        assert len(xpress_vars) == len(self._relaxation_solution)
        for xp_var, val in zip(xpress_vars, self._relaxation_solution):
            pyo_var = var_map[xp_var]
            if ref_vars[pyo_var] > 0:
                pyo_var.set_value(val, skip_validation=True)

    def _load_rc(self, vars_to_load=None):
        print(f"Cannot extract reduced costs from relaxed MIP solution")

    def _load_duals(self, cons_to_load=None):
        print(f"Cannot extract duals from relaxed MIP solution")

    def _load_slacks(self, cons_to_load=None):
        print(f"Cannot extract slacks from relaxed MIP solution")
