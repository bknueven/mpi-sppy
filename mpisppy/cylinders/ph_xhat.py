###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import copy
import logging
import math
import mpisppy.log

import pyomo.environ as pyo

from mpisppy.cylinders.xhatshufflelooper_bounder import (
    XhatShuffleInnerBound,
    ScenarioCycler,
)
from mpisppy.phbase import PHBase

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger(
    "mpisppy.cylinders.ph_xhat", "xhatclp.log", level=logging.CRITICAL
)
logger = logging.getLogger("mpisppy.cylinders.ph_xhat")


class PHXhat(XhatShuffleInnerBound):
    converger_spoke_char = "P"

    def _vb(self, msg):
        if self.verbose and self.opt.cylinder_rank == 0:
            print("(rank0) " + msg)

    def _new_nonant_debug_msg(self, xh_iter):
        # similar to above, not all ranks will agree on
        # when there are new_nonants (in the same loop)
        logger.debug(f"   *Xhatshuffle loop iter={xh_iter}")
        logger.debug(f"   *got a new one! on rank {self.global_rank}")
        logger.debug(f"   *localnonants={str(self.localnonants)}")
        self._vb("  New nonants")

    def _loop_debug_msg(self, xh_iter):
        logger.debug(
            f"   Xhatshuffle loop iter={xh_iter} on rank {self.global_rank}"
        )
        logger.debug(f"   Xhatshuffle got from opt on rank {self.global_rank}")

    def _set_options(self):
        ph_xhat_options = self.opt.options.get("ph_xhat_options", {})
        self.reverse = ph_xhat_options.get("reverse", True)
        self.iter_step = ph_xhat_options.get("iter_step", None)
        self.fixtol = ph_xhat_options.get("fixtol", 1e-6)
        self.rho_multiplier = ph_xhat_options.get("rho_multiplier", 1.0)
        self.restart_iters = ph_xhat_options.get("restart_iters", 10)
        self.solver_options = ph_xhat_options.get("xhat_solver_options", {})
        self.verbose = ph_xhat_options.get("verbose", True)

    def xhat_opt(self):
        return PHBase

    def restart_ph(self):
        self._vb("   Restarting PH")
        # set the nonants to those coming from Hub
        self.opt._put_nonant_cache(self.localnonants)
        self.opt._restore_nonants(update_persistent=False)
        self.opt._restore_original_fixedness()

        # reset Ws
        # for s in self.opt.local_scenarios.values():
        #     for w in s._mpisppy_model.W.values():
        #         w._value = 0
        self.opt.Compute_Xbar()
        self.opt.Update_W(verbose=False)
        # TODO: add smoothing
        smoothed = False
        if smoothed:
            self.opt.Update_z(self.verbose)
        # fix a bunch
        # first = True
        raw_fixed = 0
        for k, s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.is_fixed():
                    continue
                xb = pyo.value(s._mpisppy_model.xbars[ndn_i])
                diff = xb * xb - pyo.value(s._mpisppy_model.xsqbars[ndn_i])
                totval = self.fixtol * self.fixtol
                # if first:
                #     print(f"{xvar.name}, {xb=}, {diff=}, {totval=}")
                if -diff < totval and diff < totval:
                    # if first:
                    #     print("\tfixing")
                    if xvar.is_integer():
                        if math.isclose(
                            int(xb), xb, abs_tol=1e-5
                        ):
                            xvar.fix(int(xb))
                            raw_fixed += 1
                    elif xvar.lb is not None and xvar.lb > xb:
                        xvar.fix(xvar.lb)
                        raw_fixed += 1
                    elif xvar.ub is not None and xvar.ub < xb:
                        xvar.fix(xvar.ub)
                        raw_fixed += 1
                    else:
                        xvar.fix(xb)
                        raw_fixed += 1
                # else:
                #     if first:
                #         print("\tnot fixing")
            # first = False

        number_fixed = int(raw_fixed / len(self.opt.local_scenarios))
        self._vb(f"  Fixed {number_fixed} non-anticipative varibles")
        self.opt._save_nonants()

    def fix_unfix_new_nonants(self):
        nonant_caches = {}
        fixedness_caches = {}
        for k, s in self.opt.local_scenarios.items():
            nonant_caches[s] = copy.deepcopy(s._mpisppy_data.nonant_cache)
            fixedness_caches[s] = copy.deepcopy(s._mpisppy_data.fixedness_cache)

        # set the nonants to those coming from Hub, compute *their* xbar
        self.opt._put_nonant_cache(self.localnonants)
        self.opt._restore_nonants(update_persistent=False)
        self.opt.Compute_Xbar()

        # arbitrary scenario
        xbar_new = {k: p._value for k, p in s._mpisppy_model.xbars.items()}
        #xsqbars_new = {k: p._value for k, p in s._mpisppy_model.xsqbars.items()}

        for k, s in self.opt.local_scenarios.items():
            s._mpisppy_data.nonant_cache = nonant_caches[s]
            s._mpisppy_data.fixedness_cache = fixedness_caches[s]

        # now we're back to the last PH iterate
        self.opt._restore_nonants(update_persistent=False)
        self.opt.Compute_Xbar()

        raw_fixed = 0
        raw_unfixed = 0
        for k, s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                xb = pyo.value(s._mpisppy_model.xbars[ndn_i])
                diff = abs( xb - xbar_new[ndn_i] )
                if xvar.is_fixed():
                    if diff > self.fixtol:
                        xvar.unfix()
                        raw_unfixed += 1
                        continue
                elif diff < self.fixtol: # not currently fixed
                    # fix if this variable seems to have converged
                    # and it has the same xbar as the hub
                    xb = pyo.value(s._mpisppy_model.xbars[ndn_i])
                    diff = xb * xb - pyo.value(s._mpisppy_model.xsqbars[ndn_i])
                    totval = self.fixtol * self.fixtol
                    # if first:
                    #     print(f"{xvar.name}, {xb=}, {diff=}, {totval=}")
                    if -diff < totval and diff < totval:
                        # if first:
                        #     print("\tfixing")
                        if xvar.is_integer():
                            if math.isclose(
                                int(xb), xb, abs_tol=1e-5
                            ):
                                xvar.fix(int(xb))
                                raw_fixed += 1
                        elif xvar.lb is not None and xvar.lb > xb:
                            xvar.fix(xvar.lb)
                            raw_fixed += 1
                        elif xvar.ub is not None and xvar.ub < xb:
                            xvar.fix(xvar.ub)
                            raw_fixed += 1
                        else:
                            xvar.fix(xb)
                            raw_fixed += 1
                # else:
                #     if first:
                #         print("\tnot fixing")
            # first = False

        number_fixed = int(raw_fixed / len(self.opt.local_scenarios))
        number_unfixed = int(raw_unfixed / len(self.opt.local_scenarios))
        self._vb(f"  Fixed {number_fixed} non-anticipative varibles")
        self._vb(f"  Unfixed {number_unfixed} non-anticipative varibles")
        self.opt._save_nonants()


    def ph_iter(self):
        if self.opt.extensions:
            self.opt.extobject.miditer()

        self._vb("  Doing PH iteration")
        teeme = False
        dtiming = False
        self.opt.solve_loop(
            solver_options=self.solver_options,
            dtiming=dtiming,
            gripe=True,
            disable_pyomo_signal_handling=False,
            tee=teeme,
            verbose=False,  # self.verbose
        )
        self.opt._save_nonants()

        if self.opt.extensions:
            self.opt.extobject.enditer()
            self.opt.extobject.enditer_after_sync()

        self.opt.Compute_Xbar()
        self.opt.Update_W(verbose=False)
        # TODO: add smoothing
        smoothed = False
        if smoothed:
            self.opt.Update_z(self.verbose)

    def main(self):
        logger.debug(f"Entering main on ph_xhat spoke rank {self.global_rank}")

        self.opt.PH_Prep()
        self.xhat_prep()
        self._set_options()

        # give all ranks the same seed
        self.random_stream.seed(self.random_seed)

        # We need to keep track of the way scenario_names were sorted
        scen_names = list(enumerate(self.opt.all_scenario_names))

        # shuffle the scenarios associated (i.e., sample without replacement)
        shuffled_scenarios = self.random_stream.sample(scen_names, len(scen_names))

        scenario_cycler = ScenarioCycler(
            shuffled_scenarios, self.opt.nonleaves, self.reverse, self.iter_step
        )

        if self.opt.rho_setter is not None:
            self._vb("PHXhat calling rho setter")
            self.opt._use_rho_setter(False)

        # update rho
        rf = self.rho_multiplier
        for scenario in self.opt.local_scenarios.values():
            for ndn_i in scenario._mpisppy_model.rho:
                scenario._mpisppy_model.rho[ndn_i] *= rf

        xh_iter = 1
        iter_since_restart = 1_000_000_000
        conv = float("inf")
        restart_new_nonants = False
        while not self.got_kill_signal():
            # When there is no iter0, the serial number must be checked.
            # (unrelated: uncomment the next line to see the source of delay getting an xhat)
            if self.get_serial_number() == 0:
                continue

            if (xh_iter - 1) % 100 == 0:
                self._loop_debug_msg(xh_iter)

            # check if we don't already have new nonants
            if self.new_nonants:
                self._new_nonant_debug_msg(xh_iter)
            if not restart_new_nonants:
                restart_new_nonants = self.new_nonants
            if (restart_new_nonants and iter_since_restart >= self.restart_iters) or (conv < self.opt.options["convthresh"]):
                restart_new_nonants = False
                best_obj_this_nonants = float("inf")
                self.restart_ph()
                iter_since_restart = 0

            elif self.new_nonants:
                # best_obj_this_nonants = float("inf")
                self.fix_unfix_new_nonants()

            self._vb(f"    scenario_cycler._scenarios_this_epoch {scenario_cycler._scenarios_this_epoch}")
            # Restore nonants; compute xbar, W, fix lots of things, solve
            self.opt._restore_nonants(update_persistent=True)
            self.opt.reenable_W_and_prox()

            self.ph_iter()
            conv = self.opt.convergence_diff()

            self.opt.disable_W_and_prox()

            old_best_obj = best_obj_this_nonants
            scenario_cycler.begin_epoch()
            next_scendict = scenario_cycler.get_next()
            self._vb(f"   Trying next {next_scendict}")
            update, obj = self.try_scenario_dict(next_scendict)
            if obj is not None:
                obj = obj if self.is_minimizing else -obj
                if obj < best_obj_this_nonants:
                    best_obj_this_nonants = obj
                    self._vb(f"   Updating best to {next_scendict}")
                    scenario_cycler.best = next_scendict["ROOT"]

            xh_iter += 1

            next_scendict = scenario_cycler.get_next()
            if next_scendict is None or conv < self.opt.options["convthresh"]:
                continue

            self.opt._restore_nonants(update_persistent=True)
            self._vb(f"   Trying next {next_scendict}")
            update, obj = self.try_scenario_dict(next_scendict)
            obj = obj if self.is_minimizing else -obj
            if obj is not None:
                if obj < best_obj_this_nonants:
                    best_obj_this_nonants = obj
                    self._vb(f"   Updating best to {next_scendict}")
                    scenario_cycler.best = next_scendict["ROOT"]

            iter_since_restart += 1

            self._vb(f"   iter_since_restart: {iter_since_restart}")

            xh_iter += 1
