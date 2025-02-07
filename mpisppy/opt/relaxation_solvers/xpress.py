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
    doc="XpressPersistent, but load the best relaxation solution from the root MIP node",
)
class XpressRelaxationSolver(XpressPersistent):
    @staticmethod
    def get_root_node_solution(xpress_problem, self):
        print("start get_root_node_solution")
        # when the root node is finished processing
        node = xpress_problem.attributes.currentnode
        print(f"in cboptnode, node number {node}")
        objval = xpress_problem.attributes.lpobjval
        print(f"objective function value {objval}")
        # TODO: could get duals, rc, and slacks too
        sol = []
        xpress_problem.getlpsol(x=sol)
        self._relaxation_solution = sol
        self._relaxation_value = objval

        print("end get_root_node_solution")
        # returning 1 tells the solver to terminate
        return 1

    def set_xpress_callback(self):
        xpress_problem = self._solver_model
        xpress_problem.addcboptnode(
            XpressRelaxationSolver.get_root_node_solution, self, 0
        )

    def _set_instance(self, model, kwds={}):
        super()._set_instance(model, kwds)
        self.set_xpress_callback()

    def _reset_solution(self):
        self._relaxation_solution = None
        self._relaxation_value = None

    def _apply_solver(self):
        xprob = self._solver_model
        is_mip = (xprob.attributes.mipents > 0) or (xprob.attributes.sets > 0)
        if not is_mip:
            return super()._apply_solver()
        self._reset_solution()
        # turn off heuristics
        if self.options.heuremphasis is None:
            self.options.heuremphasis = 0
        if self.options.presolveops is None:
            # This turns off IP presolve, which will linearize quadratic terms,
            # and do a bunch of other reductions not valid for the dual.
            #                               1111 1111 1100 0000 0000
            #                               9876 5432 1098 7654 3210
            self.options.presolveops = int("0000_0000_0011_1111_1111", 2)
        # TODO: is this a good idea?
        # if self.options.cutstrategy is None:
        #     self.options.cutstrategy = 3
        # TODO: consider giving the user better
        #       direct control over the number
        #       of cut passes
        # if self.options.covercuts is None:
        #     self.options.covercuts = 25
        # if self.options.gomcuts is None:
        #     self.options.gomcuts = 25
        return super()._apply_solver()

    def _get_mip_results(self, results, soln):
        """Sets up `results` and `soln` and returns whether there is a solution
        to query.
        Returns `True` if a feasible solution is available, `False` otherwise.
        """
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
            xprob_attrs = self._solver_model.attributes
            if xprob_attrs.objsense == 1.0:  # minimizing MIP
                results.problem.lower_bound = self._relaxation_value
            elif xprob_attrs.objsense == -1.0:  # maximizing MIP
                results.problem.upper_bound = self._relaxation_value
            return True
        return False

    def _postsolve(self):
        ret = super()._postsolve()
        xprob = self._solver_model
        is_mip = (xprob.attributes.mipents > 0) or (xprob.attributes.sets > 0)
        if not is_mip:
            return ret
        if self._relaxation_solution is not None:
            self.results.problem.number_of_solutions = 1
        return ret

    def _load_vars(self, vars_to_load=None):
        xprob = self._solver_model
        is_mip = (xprob.attributes.mipents > 0) or (xprob.attributes.sets > 0)
        if not is_mip:
            return super()._load_vars(vars_to_load=vars_to_load)
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
        # for v in self._pyomo_model._mpisppy_data.nonant_indices.values():
        #     print(v.name, v.value)
        # raise RuntimeError

    def _load_rc(self, vars_to_load=None):
        xprob = self._solver_model
        is_mip = (xprob.attributes.mipents > 0) or (xprob.attributes.sets > 0)
        if not is_mip:
            return super()._load_rc(vars_to_load=vars_to_load)
        print("Cannot extract reduced costs from relaxed MIP solution")

    def _load_duals(self, cons_to_load=None):
        xprob = self._solver_model
        is_mip = (xprob.attributes.mipents > 0) or (xprob.attributes.sets > 0)
        if not is_mip:
            return super()._load_duals(cons_to_load=cons_to_load)
        print("Cannot extract duals from relaxed MIP solution")

    def _load_slacks(self, cons_to_load=None):
        xprob = self._solver_model
        is_mip = (xprob.attributes.mipents > 0) or (xprob.attributes.sets > 0)
        if not is_mip:
            return super()._load_slacks(cons_to_load=cons_to_load)
        print("Cannot extract slacks from relaxed MIP solution")
