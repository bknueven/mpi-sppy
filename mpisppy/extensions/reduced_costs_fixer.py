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

from mpisppy.extensions.extension import Extension
from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke
from mpisppy.utils.sputils import is_persistent, find_active_objective

from mpisppy.cylinders.spwindow import Field


class ReducedCostsFixer(Extension):

    def __init__(self, spobj):
        super().__init__(spobj)

        ph_options = spobj.options
        rc_options = ph_options['rc_options']
        self.verbose = ph_options['verbose'] or rc_options['verbose']
        self.debug = rc_options['debug']

        # reduced costs less than this in absolute value
        # will be considered 0
        self.zero_rc_tol = rc_options['zero_rc_tol']
        # Percentage of variables which are at the bound we will target
        # to fix. We never fix varibles with reduced costs less than
        # the `zero_rc_tol` in absolute value
        self._fix_fraction_target_pre_iter0 = rc_options.get('fix_fraction_target_pre_iter0', 0)
        if self._fix_fraction_target_pre_iter0 < 0 or self._fix_fraction_target_pre_iter0 > 1:
            raise ValueError("fix_fraction_target_pre_iter0 must be between 0 and 1")
        self._fix_fraction_target_iter0 = rc_options['fix_fraction_target_iter0']
        if self._fix_fraction_target_iter0 < 0 or self._fix_fraction_target_iter0 > 1:
            raise ValueError("fix_fraction_target_iter0 must be between 0 and 1")
        self._fix_fraction_target_iterK = rc_options['fix_fraction_target_iterK']
        if self._fix_fraction_target_iterK < 0 or self._fix_fraction_target_iterK > 1:
            raise ValueError("fix_fraction_target_iterK must be between 0 and 1")
        self.fix_fraction_target = self._fix_fraction_target_pre_iter0

        # TODO: This should be same as in rc spoke?
        self.bound_tol = rc_options['rc_bound_tol']

        self._integer_relaxer = pyo.TransformationFactory("core.relax_integer_vars")

    def pre_iter0(self):
        self._modeler_fixed_nonants = set()
        self.nonant_length = self.opt.nonant_length
        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants.add(ndn_i)

    def pre_solve(self, sub):
        reduced_costs = self._extract_reduced_costs(sub)
        if self.opt._PHIter == 0:
            self.reduced_costs_fixing(sub, reduced_costs, pre_iter0 = True)
        else:
            self.reduced_costs_fixing(sub, reduced_costs, pre_iter0 = False)

    def _extract_reduced_costs(self, sub):
        solver = sub._solver_plugin
        persistent_solver = is_persistent(sub._solver_plugin)

        # unfix nonants
        for sn in sub.scen_list:
            s = self.opt.local_scenarios[sn]
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed and ndn_i not in self._modeler_fixed_nonants:
                    xvar.unfix()
                    if persistent_solver:
                        solver.update_var(xvar)

        # relax integers
        self._integer_relaxer.apply_to(sub)
        if persistent_solver:
            _relaxed_integer_vars = [v for v, d in sub._relaxed_integer_vars[None].values()]
            for var in _relaxed_integer_vars:
                solver.update_var(var)

        # solve; turn off crossover, extract RCs, delete suffix...
        _prior_crossover_setting = self.opt.current_solver_options.get("crossover", -1)
        self.opt.current_solver_options["crossover"] = 0
        sub.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        ### TODO: maybe we need to set the objective here, and again *after*
        ###       we enforce the integer variables...
        if persistent_solver:
            active_objective = find_active_objective(s)
            solver.set_objective(active_objective)

        solve_keyword_args = {}
        if persistent_solver:
            solve_keyword_args["save_results"] = False
        for k, v in self.opt.current_solver_options.items():
            solver.options[k] = v

        results = solver.solve(s, **solve_keyword_args)

        pyo.assert_optimal_termination(results)

        vars_to_load = []
        vars_indicies = []
        for sn in sub.scen_list:
            s = self.opt.local_scenarios[sn]
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                vars_to_load.append(xvar)
                vars_indicies.append(ndn_i)

        if persistent_solver:
            solver.load_vars(vars_to_load=vars_to_load)
            solver.load_rc(vars_to_load=vars_to_load)
        else:
            sub.solutions.load_from(results)

        reduced_costs = np.zeros(len(vars_to_load))
        for ci, (ndn_i, xvar) in enumerate(zip(vars_indicies, vars_to_load)):
            if ndn_i in self._modeler_fixed_nonants:
                continue
            reduced_costs[ci] = sub.rc[xvar]

        sub.del_component("rc")
        self.opt.current_solver_options["crossover"] = _prior_crossover_setting
        # make integer
        self._integer_relaxer.apply_to(sub, undo=True)
        if persistent_solver:
            for var in _relaxed_integer_vars:
                solver.update_var(var)

        if persistent_solver:
            active_objective = find_active_objective(s)
            solver.set_objective(active_objective)

        return reduced_costs

    def reduced_costs_fixing(self, sub, reduced_costs, pre_iter0 = False):

        # compute the quantile target
        abs_reduced_costs = np.abs(reduced_costs)

        # fix_fraction_target = self.fix_fraction_target

        # # excludes nan
        # nonzero_rc = abs_reduced_costs[abs_reduced_costs > self.zero_rc_tol]
        # if len(nonzero_rc) == 0:
        #     # still need to continue, for unfixing
        #     target = self.zero_rc_tol
        # else:
        # TODO: need to rethink as there could be multiple subproblem reduced costs!
        #     target = np.nanquantile(nonzero_rc, 1 - fix_fraction_target, method="median_unbiased")

        # if target < self.zero_rc_tol:
        # shouldn't be reached
        target = self.zero_rc_tol

        ci = 0 # buffer index
        if self.verbose:
            print(f"Subproblem {sub.name}, heuristic fixing reduced cost cutoff: {target:.2e}")
        raw_fixed_this_iter = 0
        persistent_solver = is_persistent(sub._solver_plugin)
        for sn in sub.scen_list:
            s = self.opt.local_scenarios[sn]
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                this_rc = reduced_costs[ci]
                abs_this_rc = abs_reduced_costs[ci]
                # next index, in case we break
                ci += 1
                if ndn_i in self._modeler_fixed_nonants:
                    continue
                if xvar in s._mpisppy_data.all_surrogate_nonants:
                    continue
                if pre_iter0:
                    x_bar = None
                else:
                    x_bar = s._mpisppy_model.xbars[ndn_i].value
                x_val = xvar.value
                update_var = False
                # print(f"rank: {self.opt.cylinder_rank}, var {xvar.name}, value {x_val}, rc {this_rc}, xbar {x_bar}")
                if xvar.fixed and not pre_iter0:
                    if abs_this_rc <= target or abs(x_bar - x_val) > self.bound_tol:
                        xvar.unfix()
                        update_var = True
                        raw_fixed_this_iter -= 1
                        if self.debug and self.opt.cylinder_rank == 0:
                            msg = f"unfixing var {xvar.name}; "
                            if abs(x_bar - x_val) > self.bound_tol:
                                msg += f"{x_bar=} is differs from the fixed value {x_val=}"
                            else:
                                msg += f"reduced cost {this_rc} is zero/below target"
                            print(msg)
                elif (pre_iter0 or abs(x_val - x_bar) <= self.bound_tol) and (abs_this_rc >= target):
                    if this_rc > self.zero_rc_tol and (pre_iter0 or (x_bar - xvar.lb <= self.bound_tol)):
                        xvar.fix(xvar.lb)
                        if self.debug and self.opt.cylinder_rank == 0:
                            print(f"fixing var {xvar.name} to lb {xvar.lb}; reduced cost is {this_rc}, {x_val=}")
                        update_var = True
                        raw_fixed_this_iter += 1
                    elif (this_rc < -self.zero_rc_tol) and (pre_iter0 or (xvar.ub - x_bar <= self.bound_tol)):
                        xvar.fix(xvar.ub)
                        if self.debug and self.opt.cylinder_rank == 0:
                            print(f"fixing var {xvar.name} to ub {xvar.ub}; reduced cost is {this_rc}, {x_val=}")
                        update_var = True
                        raw_fixed_this_iter += 1
                    # ???
                    # elif (not pre_iter0) and abs(s._mpisppy_model.W[ndn_i].value) > self.zero_rc_tol:
                    #     xvar.fix(x_bar)
                    #     if self.debug and self.opt.cylinder_rank == 0:
                    #         print(f"fixing var {xvar.name} to x_bar {x_bar}; W is {s._mpisppy_model.W[ndn_i].value}")
                    #     update_var = True
                    #     raw_fixed_this_iter += 1
                    else:
                        # rc is near zero or x_bar is away from the bound
                        pass

                if update_var and persistent_solver:
                    sub._solver_plugin.update_var(xvar)

        if self.verbose:
            print(f"Subproblem {sub.name}, total unique vars fixed by heuristic: {int(round(raw_fixed_this_iter))}/{self.nonant_length}")
