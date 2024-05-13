# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
from math import isfinite

import mpisppy.cylinders.spoke
from mpisppy.cylinders.lagrangian_bounder import _LagrangianMixin
from mpisppy.convergers.norms_and_residuals import primal_residuals_norm

class SubgradientOuterBound(_LagrangianMixin, mpisppy.cylinders.spoke.OuterBoundSpoke):

    converger_spoke_char = 'G'

    def main(self):
        # The rho_setter should be attached to the opt object
        rho_setter = None
        if hasattr(self.opt, 'rho_setter'):
            rho_setter = self.opt.rho_setter
        extensions = self.opt.extensions is not None
        verbose = self.opt.options['verbose']

        self.lagrangian_prep()

        if extensions:
            self.opt.extobject.pre_iter0()
        self.dk_iter = 1
        self.trivial_bound = self.lagrangian()
        if extensions:
            self.opt.extobject.post_iter0()

        self.bound = self.trivial_bound
        if extensions:
            self.opt.extobject.post_iter0_after_sync()

        self.opt.current_solver_options = self.opt.iterk_solver_options

        # update rho / alpha
        if self.opt.options.get('subgradient_rho_multiplier') is not None:
            rf = self.opt.options['subgradient_rho_multiplier']
            for scenario in self.opt.local_scenarios.values():
                for ndn_i in scenario._mpisppy_model.rho:
                    scenario._mpisppy_model.rho[ndn_i] *= rf

        polyak_stepsize = self.opt.options.get('polyak_stepsize', True)

        bound = self.bound
        while not self.got_kill_signal():
            # compute a subgradient step
            self.opt.Compute_Xbar(verbose)
            if polyak_stepsize:
                inner_bound = self.hub_inner_bound
                if isfinite(inner_bound):
                    primal_resid = primal_residuals_norm(self.opt)
                    alpha_k = abs( inner_bound - bound ) / max(1e-10, (primal_resid*primal_resid))
                    for scenario in self.opt.local_scenarios.values():
                        for ndn_i in scenario._mpisppy_model.rho:
                            scenario._mpisppy_model.rho[ndn_i] = alpha_k
            self.opt.Update_W(verbose)
            if extensions:
                self.opt.extobject.miditer()
            bound = self.lagrangian()
            if extensions:
                self.opt.extobject.enditer()
            if bound is not None:
                self.bound = bound
            if extensions:
                self.opt.extobject.enditer_after_sync()
