###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
import logging
import math
import mpisppy.log

import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
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

    def xhat_opt(self):
        return PHBase

    def main(self):
        logger.debug(f"Entering main on ph_xhat spoke rank {self.global_rank}")

        self.opt.PH_Prep()
        self.xhat_prep()
        if "reverse" in self.opt.options["ph_xhat_options"]:
            self.reverse = self.opt.options["ph_xhat_options"]["reverse"]
        else:
            self.reverse = True
        if "iter_step" in self.opt.options["ph_xhat_options"]:
            self.iter_step = self.opt.options["ph_xhat_options"]["iter_step"]
        else:
            self.iter_step = None
        if "fixtol" in self.opt.options["ph_xhat_options"]:
            self.fixtol = self.opt.options["ph_xhat_options"]["fixtol"]
        else:
            self.fixtol = 1e-6
        self.solver_options = self.opt.options["ph_xhat_options"]["xhat_solver_options"]

        self.verbose = True
        # give all ranks the same seed
        self.random_stream.seed(self.random_seed)

        # We need to keep track of the way scenario_names were sorted
        scen_names = list(enumerate(self.opt.all_scenario_names))

        # shuffle the scenarios associated (i.e., sample without replacement)
        shuffled_scenarios = self.random_stream.sample(scen_names, len(scen_names))

        scenario_cycler = ScenarioCycler(
            shuffled_scenarios, self.opt.nonleaves, self.reverse, self.iter_step
        )

        def _vb(msg):
            if self.verbose and self.opt.cylinder_rank == 0:
                print("(rank0) " + msg)

        if self.opt.rho_setter is not None:
            _vb("PHXhat calling rho setter")
            self.opt._use_rho_setter(False)
        # _vb(f"  Doing PH iter 0")
        # teeme = False
        # dtiming = False
        # self.opt.solve_loop(
        #     solver_options=self.solver_options,
        #     dtiming=dtiming,
        #     gripe=True,
        #     disable_pyomo_signal_handling=False,
        #     tee=teeme,
        #     verbose=False, #self.verbose
        # )

        xh_iter = 1
        while not self.got_kill_signal():
            # When there is no iter0, the serial number must be checked.
            # (unrelated: uncomment the next line to see the source of delay getting an xhat)
            if self.get_serial_number() == 0:
                continue

            if (xh_iter - 1) % 100 == 0:
                logger.debug(
                    f"   Xhatshuffle loop iter={xh_iter} on rank {self.global_rank}"
                )
                logger.debug(f"   Xhatshuffle got from opt on rank {self.global_rank}")

            if self.new_nonants:
                # similar to above, not all ranks will agree on
                # when there are new_nonants (in the same loop)
                logger.debug(f"   *Xhatshuffle loop iter={xh_iter}")
                logger.debug(f"   *got a new one! on rank {self.global_rank}")
                logger.debug(f"   *localnonants={str(self.localnonants)}")
                _vb("  New nonants")

                # update the caches
                self.opt._put_nonant_cache(self.localnonants)
                self.opt._restore_nonants(update_persistent=False)
                self.opt._restore_original_fixedness()

                # reset Ws
                for s in self.opt.local_scenarios.values():
                    for w in s._mpisppy_model.W.values():
                        w._value = 0

                scenario_cycler.begin_epoch()
                next_scendict = scenario_cycler.get_next()
                if next_scendict is not None:
                    _vb(f"   Trying next {next_scendict}")
                    update = self.try_scenario_dict(next_scendict)
                    if update:
                        _vb(f"   Updating best to {next_scendict}")
                        scenario_cycler.best = next_scendict["ROOT"]

                self.opt._restore_nonants(update_persistent=False)

                self.opt.Compute_Xbar()
                # fix a bunch
                # first = True
                fixed_vars = {}
                raw_fixed = 0
                for k, s in self.opt.local_scenarios.items():
                    fixed_vars[s] = pyo.ComponentMap()
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
                                    fixed_vars[s][xvar] = int(xb)
                                    raw_fixed += 1
                            elif xvar.lb is not None and xvar.lb > xb:
                                fixed_vars[s][xvar] = xvar.lb
                                raw_fixed += 1
                            elif xvar.ub is not None and xvar.ub < xb:
                                fixed_vars[s][xvar] = xvar.ub
                                raw_fixed += 1
                            else:
                                fixed_vars[s][xvar] = xb
                                raw_fixed += 1
                        # else:
                        #     if first:
                        #         print("\tnot fixing")
                    # first = False

                number_fixed = int(raw_fixed / len(self.opt.local_scenarios))
                _vb(f"  Fixed {number_fixed} non-anticipative varibles")

            _vb(f"    scenario_cycler._scenarios_this_epoch {scenario_cycler._scenarios_this_epoch}")
            # Restore nonants; compute xbar, W, fix lots of things, solve
            self.opt._restore_nonants(update_persistent=True)
            self.opt.reenable_W_and_prox()
            self.opt.Compute_Xbar()
            self.opt.Update_W(verbose=False)

            for s in fixed_vars:
                for var, val in fixed_vars[s].items():
                    var.fix(val)
                if (sputils.is_persistent(s._solver_plugin)):
                    for var in fixed_vars[s]:
                        s._solver_plugin.update_var(var)

            # TODO: add smoothing
            smoothed = False
            if smoothed:
                self.opt.Update_z(self.verbose)

            if self.opt.extensions:
                self.opt.extobject.miditer()

            _vb("  Doing PH iteration")
            _vb(f"  Solver: {s._solver_plugin}")
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

            if self.opt.extensions:
                self.opt.extobject.enditer()
                self.opt.extobject.enditer_after_sync()

            self.opt.disable_W_and_prox()
            next_scendict = scenario_cycler.get_next()
            if next_scendict is None:
                scenario_cycler.begin_epoch()
                next_scendict = scenario_cycler.get_next()
            _vb(f"   Trying next {next_scendict}")
            update = self.try_scenario_dict(next_scendict)
            if update:
                _vb(f"   Updating best to {next_scendict}")
                scenario_cycler.best = next_scendict["ROOT"]

            xh_iter += 1
