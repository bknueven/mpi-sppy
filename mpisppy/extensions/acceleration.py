# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

from mpisppy.extensions.norm_rho_updater import NormRhoUpdater

class PHAcceleration(NormRhoUpdater):
    """ PH Acceleration via overrelaxation
        Overwrite all Extension methods to
        only get what we need from NormRhoUpdater
    """
    def __init__(self, spopt_object):
        super().__init__(spopt_object)
        self.ph = spopt_object
        self.alpha = 0.9 # acceleration term
        self.eta = 0.9 # guard/restart condition
        self._gamma_init = None # first convergence criterion
        self._gamma = None # convergence criterion
        self._prev_avg = {}
        self._prev_W = {}

    def get_current_step(self):
        current_step = {}
        for s in self.ph.local_scenarios.values():
            for ndn_i, xbar in s._mpisppy_model.xbars.items():
                current_step[ndn_i] = xbar.value
            return current_step

    def setup_hub(self):
        '''
        Method called when the Hub SPCommunicator is set up (if used)

        Returns
        -------
        None
        '''
        pass

    def initialize_spoke_indices(self):
        '''
        Method called when the Hub SPCommunicator initializes its spoke indices

        Returns
        -------
        None
        '''
        pass

    def sync_with_spokes(self):
        '''
        Method called when the Hub SPCommunicator syncs with spokes

        Returns
        -------
        None
        '''
        pass

    def pre_solve(self, subproblem):
        '''
        Method called before every subproblem solve

        Inputs
        ------
        subproblem : Pyomo subproblem (could be a scenario or bundle)

        Returns
        -------
        None
        '''
        pass

    def post_solve(self, subproblem, results):
        '''
        Method called after every subproblem solve

        Inputs
        ------
        subproblem : Pyomo subproblem (could be a scenario or bundle)
        results : Pyomo results object from initial solve or None if solve failed

        Returns
        -------
        results : Pyomo results objects from most recent solve
        '''
        return results

    def pre_solve_loop(self):
        ''' Method called before every solve loop within
            mpisppy.spot.SPOpt.solve_loop()
        '''
        pass

    def post_solve_loop(self):
        ''' Method called after every solve loop within
            mpisppy.spot.SPOpt.solve_loop()
        '''
        pass

    def pre_iter0(self):
        ''' Method called at the end of PH_Prep().
            When this method is called, all scenarios have been created, and
            the dual/prox terms have been attached to the objective, but the
            solvers have not yet been created.
        '''
        pass

    def post_iter0(self):
        ''' Method called after the first PH iteration.
            When this method is called, one call to solve_loop() has been
            completed, and we have ensured that none of the models are
            infeasible. The rho_setter, if present, has not yet been applied.
        '''
        pass

    def post_iter0_after_sync(self):
        ''' Method called after the first PH iteration, after the
            synchronization of sending messages between cylinders
            has completed.
        '''
        pass

    def _compute_gamma_init(self):
        primal_residual = self._compute_primal_residual_norm(self.ph)
        dual_residual = self._compute_dual_residual_norm(self.ph)
        self._gamma_init = {ndn_i : primal_residual[ndn_i] + dual_residual[ndn_i] for ndn_i in primal_residual}
        return

    def _compute_gamma(self):
        primal_residual = self._compute_primal_residual_norm(self.ph)
        dual_residual = self._compute_dual_residual_norm(self.ph)
        return {ndn_i : primal_residual[ndn_i] + dual_residual[ndn_i] for ndn_i in primal_residual}

    def _snapshot_W(self, ph):
        for s in ph.local_scenarios.values():
            self._prev_W[s] = {}
            for ndn_i, w in s._mpisppy_model.W.items():
                self._prev_W[s][ndn_i] = w._value

    def miditer(self):
        ''' Method called after x-bar has been computed and the dual weights
            have been updated, but before solve_loop().
            If a converger is present, this method is called between the
            convergence_value() method and the is_converged() method.
        '''

        if not self._prev_avg:
            self._snapshot_avg(self.ph)
            self._snapshot_W(self.ph)
            return
        if self._gamma_init is None:
            self._compute_gamma_init()

        # compute acceleration
        current_step = self.get_current_step()

        # for the gamma_k computation
        for s in self.ph.local_scenarios.values():
            for ndn_i, xbar in s._mpisppy_model.xbars.items():
                xbar._value = current_step[ndn_i] + self.alpha * (current_step[ndn_i] - self._prev_avg[ndn_i])

        gamma_k = self._compute_gamma()

        for ndn_i in gamma_k:
            #print(f"{ndn_i}, gamma_k: {gamma_k[ndn_i]}, initial: {self._gamma_init[ndn_i]}")
            if gamma_k[ndn_i] < self._gamma_init[ndn_i] * self.eta:
                # keep the inertia
                print(f"over relaxing {ndn_i}")
                current_step[ndn_i] = current_step[ndn_i] + self.alpha * (current_step[ndn_i] - self._prev_avg[ndn_i])
                for s in self.ph.local_scenarios.values():
                    s._mpisppy_model.W[ndn_i]._value = s._mpisppy_model.W[ndn_i]._value + self.alpha * (s._mpisppy_model.W[ndn_i]._value - self._prev_W[s][ndn_i])
            #else:
                #print(f"not over relaxing {ndn_i}")
                # else we will keep the current_step

        # store (new) xbars
        for s in self.ph.local_scenarios.values():
            for ndn_i, xbar in s._mpisppy_model.xbars.items():
                xbar._value = current_step[ndn_i]

        #self.ph.Update_W(verbose=False)

        self._prev_avg = current_step
        self._snapshot_W(self.ph)
        self._gamma_init = gamma_k

    def enditer(self):
        ''' Method called after the solve_loop(), but before the next x-bar and
            weight update.
        '''
        pass

    def enditer_after_sync(self):
        ''' Method called after the solve_loop(), after the
            synchronization of sending messages between cylinders
            has completed.
        '''
        pass

    def post_everything(self):
        ''' Method called after the termination of the algorithm.
            This method is called after the scenario_denouement, if a
            denouement is present. This function will not begin on any rank
            within self.opt.mpicomm until the scenario_denouement has completed
            on all other ranks.
        '''
        pass
