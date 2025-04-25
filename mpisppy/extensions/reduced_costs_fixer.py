###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import numpy as np


from mpisppy.extensions.extension import Extension
from mpisppy.cylinders.reduced_costs_spoke import ReducedCostsSpoke
from mpisppy.utils.sputils import is_persistent

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
        self._rc_fixer_require_improving_lagrangian = rc_options.get('rc_fixer_require_improving_lagrangian', True)
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

        self._heuristic_fixed_vars = {}
        if spobj.is_minimizing:
            self._best_outer_bound = -float("inf")
            self._outer_bound_update = lambda new, old : (new > old)
        else:
            self._best_outer_bound = float("inf")
            self._outer_bound_update = lambda new, old : (new < old)

        self._current_reduced_costs = None

    def _update_best_outer_bound(self, new_outer_bound):
        if self._outer_bound_update(new_outer_bound, self._best_outer_bound):
            self._best_outer_bound = new_outer_bound
            return True
        return False

    def pre_iter0(self):
        self._modeler_fixed_nonants = set()
        self._integer_nonants = set()
        self.nonant_length = self.opt.nonant_length
        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants.add(ndn_i)
                    continue
                if xvar.is_integer():
                    self._integer_nonants.add(ndn_i)

    def iter0_post_solver_creation(self):
        self.fix_fraction_target = self._fix_fraction_target_pre_iter0
        if self.fix_fraction_target > 0:
            # wait for the reduced costs
            if self.opt.cylinder_rank == 0 and self.verbose:
                print("Fixing based on reduced costs prior to iteration 0!")
            if self.reduced_cost_buf.id() == 0:
                while not self.opt.spcomm.get_receive_buffer(self.outer_bound_buf, Field.OBJECTIVE_OUTER_BOUND, self.reduced_costs_spoke_index):
                    continue
            self.sync_with_spokes(pre_iter0 = True)
        self.fix_fraction_target = self._fix_fraction_target_iter0

    def post_iter0_after_sync(self):
        self.fix_fraction_target = self._fix_fraction_target_iterK

    def register_receive_fields(self):
        spcomm = self.opt.spcomm
        expected_reduced_cost_ranks = spcomm.fields_to_ranks[Field.EXPECTED_REDUCED_COST]
        assert len(expected_reduced_cost_ranks) == 1
        index = expected_reduced_cost_ranks[0]

        self.reduced_costs_spoke_index = index

        self.reduced_cost_buf = spcomm.register_recv_field(
            Field.SCENARIO_REDUCED_COST,
            self.reduced_costs_spoke_index,
        )
        self.outer_bound_buf = spcomm.register_recv_field(
            Field.OBJECTIVE_OUTER_BOUND,
            self.reduced_costs_spoke_index,
        )

        return

    def sync_with_spokes(self, pre_iter0 = False):
        # TODO: If we calculate the new bounds in the spoke we don't need to
        #       check if the buffers have the same ID.
        # NOTE: If we do this, then the heuristic reduced cost fixing might fix
        #       different variables in different subproblems. But this might be
        #       fine.
        self.opt.spcomm.get_receive_buffer(
            self.reduced_cost_buf,
            Field.SCENARIO_REDUCED_COST,
            self.reduced_costs_spoke_index,
        )
        self.opt.spcomm.get_receive_buffer(
            self.outer_bound_buf,
            Field.OBJECTIVE_OUTER_BOUND,
            self.reduced_costs_spoke_index,
        )
        if self.reduced_cost_buf.is_new() and self.reduced_cost_buf.id() == self.outer_bound_buf.id():
            reduced_costs = self.reduced_cost_buf.value_array()
            this_outer_bound = self.outer_bound_buf.value_array()[0]
            is_new_outer_bound = self._update_best_outer_bound(this_outer_bound)
            if pre_iter0:
                # make sure we set the bound we compute prior to iteration 0
                self.opt.spcomm.BestOuterBound = self.opt.spcomm.OuterBoundUpdate(
                    self._best_outer_bound,
                    cls=ReducedCostsSpoke,
                    idx=self.reduced_costs_spoke_index,
                )
            if is_new_outer_bound or not self._rc_fixer_require_improving_lagrangian:
                self._current_reduced_costs = np.array(reduced_costs[:])
                if not self.opt.is_minimizing: # reverse the sign for ease of implementation
                    self._current_reduced_costs *= -1.0
        else:
            if self.opt.cylinder_rank == 0 and self.verbose:
                print("No new reduced costs!")
            ## End if
        ## End if
        if self.fix_fraction_target > 0.0 and self._current_reduced_costs is not None:
            # makes sense to run this every iteration becuase xbar can change!!
            self.reduced_costs_fixing(self._current_reduced_costs)
        ## End if

        return


    def reduced_costs_fixing(self, reduced_costs, pre_iter0 = False):

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
        for k, sub in self.opt.local_subproblems.items():
            if k not in self._heuristic_fixed_vars:
                self._heuristic_fixed_vars[k] = 0
            if self.verbose:
                print(f"Subproblem {k}, heuristic fixing reduced cost cutoff: {target:.2e}")
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

            # Note: might count incorrectly with bundling?
            self._heuristic_fixed_vars[k] += raw_fixed_this_iter
            if self.verbose:
                print(f"Subproblem {k}, total unique vars fixed by heuristic: {int(round(self._heuristic_fixed_vars[k]))}/{self.nonant_length}")
