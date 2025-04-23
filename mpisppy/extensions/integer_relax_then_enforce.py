###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################

import time
import pyomo.environ as pyo
import mpisppy.extensions.extension
from mpisppy.utils.sputils import is_persistent
from mpisppy import global_toc

class IntegerRelaxThenEnforce(mpisppy.extensions.extension.Extension):
    """ Class for relaxing integer variables, running PH, and then
        enforcing the integality constraints after some condition.
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.integer_relaxer = pyo.TransformationFactory('core.relax_integer_vars')
        options = opt.options.get("integer_relax_then_enforce_options", {})
        # fraction of iterations or time to spend in relaxed mode
        self.ratio = options.get("ratio", 0.5)
        self.agg_on_relax_only = options.get("agg_on_relax_only", True)
        self._non_agg_rhos = {}
        self._reset_non_agg_rhos_next_iter = False


    def pre_iter0(self):
        global_toc(f"{self.__class__.__name__}: relaxing integrality constraints", self.opt.cylinder_rank == 0)
        for s in self.opt.local_scenarios.values():
            self.integer_relaxer.apply_to(s) 
        self._integers_relaxed = True

    def _cache_rho_reset_W(self):
        print(f"caching rho")
        for s in self.opt.local_scenarios.values():
            self._non_agg_rhos[s] = {}
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar not in s._mpisppy_data.all_surrogate_nonants:
                    self._non_agg_rhos[s][ndn_i] = s._mpisppy_model.rho[ndn_i]._value
                    s._mpisppy_model.rho[ndn_i]._value = 0
                    s._mpisppy_model.W[ndn_i]._value = 0

    def _reset_rho(self):
        print(f"resetting rho")
        for s, rhos in self._non_agg_rhos.items():
            for ndn_i, val in rhos.items():
                s._mpisppy_model.rho[ndn_i]._value = val
            # for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
            #     if xvar in s._mpisppy_data.all_surrogate_nonants:
            #         s._mpisppy_model.rho[ndn_i]._value *= 1e-1
        self._reset_non_agg_rhos_next_iter = False

    def _unrelax_integers(self):
        for sub in self.opt.local_subproblems.values():
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                subproblem_solver = sub._solver_plugin
                vlist = None
                if is_persistent(subproblem_solver):
                    vlist = list(v for v,d in s._relaxed_integer_vars[None].values())
                self.integer_relaxer.apply_to(s, options={"undo":True}) 
                if is_persistent(subproblem_solver):
                    for v in vlist:
                        subproblem_solver.update_var(v)
        self._integers_relaxed = False
        self._reset_non_agg_rhos_next_iter = True

    def miditer(self):
        if self.agg_on_relax_only:
            if not self._non_agg_rhos:
                self._cache_rho_reset_W()
                return
        if self.agg_on_relax_only and self._reset_non_agg_rhos_next_iter:
            self._reset_rho()
            return
        if not self._integers_relaxed:
            return
        # time is running out
        if self.opt.options["time_limit"] is not None and ( time.perf_counter() - self.opt.start_time ) > (self.opt.options["time_limit"] * self.ratio):
            global_toc(f"{self.__class__.__name__}: enforcing integrality constraints, ran so far for more than {self.opt.options['time_limit']*self.ratio} seconds", self.opt.cylinder_rank == 0)
            self._unrelax_integers()
        # iterations are running out
        if self.opt._PHIter > self.opt.options["PHIterLimit"] * self.ratio:
            global_toc(f"{self.__class__.__name__}: enforcing integrality constraints, ran so far for {self.opt._PHIter - 1} iterations", self.opt.cylinder_rank == 0)
            self._unrelax_integers()
        # nearly converged
        if self.opt.conv < (self.opt.options["convthresh"] * 1.1):
            global_toc(f"{self.__class__.__name__}: Enforcing integrality constraints, PH is nearly converged", self.opt.cylinder_rank == 0)
            self._unrelax_integers()
