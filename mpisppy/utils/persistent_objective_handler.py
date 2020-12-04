# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import abc

## ABC for Objective Handlers
class PersistentObjectiveHandler(abc.ABC)

    def __init__(self, ph, scenario):
        self.solver_original_obj = None
        self.solver_obj_sense = None
        self.prior_linear_offsets = None
        self.prior_quadratic_offsets = None
        self.prior_objective_offsets = 0.
        self.is_minimizing = ph.is_minimizing
        self.scenario = scenario

    @abc.abstractmethod
    def _set_solver_original_obj(self, scenario):
        '''should set self.solver_original_obj and self.solver_obj_sense based on PersistentSolver'''
        pass

    @abc.abstractmethod
    def _update_quadratic(self, scenario, new_quadratic_offsets, new_linear_offsets, new_objective_offset):
        '''does a full update in the scenario._solver_plugin based on new_quadratic_offsets, new_linear_offsets, and new_objective_offset'''
        pass
    
    @abc.abstractmethod
    def _update_linear(self, scenario, new_linear_offsets, new_objective_offset):
        '''does a partial update in the scenario._solver_plugin based on new_linear_offsets and new_objective_offset'''
        pass

    @abc.abstractmethod
    def _reset_objective(self, scenario):
        '''resets the objective to self.solver_original_obj'''
        pass

    def set_solver_original_obj(self, scenario):
        '''should set self.solver_original_obj and self.solver_obj_sense based on PersistentSolver'''
        if scenario is not self.scenario:
            raise Exception("Scenario subproblem is different from the PersistentObjectiveHandler's instance scenario subproblem")
        self._set_solver_original_obj(self, scenario)

    def update_quadratic(self, scenario, new_quadratic_offsets, new_linear_offsets, new_objective_offset):
        '''does a full update in the scenario._solver_plugin based on new_quadratic_offsets, new_linear_offsets, and new_objective_offset'''
        if scenario is not self.scenario:
            raise Exception("Scenario subproblem is different from the PersistentObjectiveHandler's instance scenario subproblem")
        self._update_quadratic(self, scenario, new_quadratic_offsets, new_linear_offsets, new_objective_offset):
    
    def update_linear(self, scenario, new_linear_offsets, new_objective_offset):
        '''does a partial update in the scenario._solver_plugin based on new_linear_offsets and new_objective_offset'''
        if scenario is not self.scenario:
            raise Exception("Scenario subproblem is different from the PersistentObjectiveHandler's instance scenario subproblem")
        self._update_linear(self, scenario, new_linear_offsets, new_objective_offset):

    def reset_objective(self, scenario):
        '''resets the objective to self.solver_original_obj'''
        if scenario is not self.scenario:
            raise Exception("Scenario subproblem is different from the PersistentObjectiveHandler's instance scenario subproblem")
        self._reset_objective(self, scenario):

    ## for diff-ing objective term offset dictionaries
    def diff_offsets(self, offset1, offset2):
        return offset1 != offset2

    def set_objective(self, scenario, PHoptions):
        if scenario is not self.scenario:
            raise Exception("Scenario subproblem is different from the PersistentObjectiveHandler's instance scenario subproblem")
        linearize_binary_proximal_terms = False
        if "linearize_binary_proximal_terms" in PHoptions:
            linearize_binary_proximal_terms = PHoptions["linearize_binary_proximal_terms"]

        if self.solver_original_obj is None:
            self.set_solver_original_obj(scenario)

        new_quadratic_offsets = {}
        new_linear_offsets = {}
        new_objective_offset = 0.

        is_min_problem = self.is_minimizing

        for ndn_i, xvar in scenario._nonant_indexes.items():
            xbar = scenario._xbars[ndn_i].value
            if xvar.is_binary() and linearize_binary_proximal_terms:
                new_linear_offsets[ndn_i] = scenario._PHprox_on[ndn_i].value * (scenario._PHrho[ndn_i].value /2.0)*(1.-2.*xbar)
            else:
                new_quadratic_offsets[ndn_i] = scenario._PHprox_on[ndn_i].value * (scenario._PHrho[ndn_i].value /2.0)
                new_linear_offsets[ndn_i] = new_quadratic_offsets[ndn_i]*(-2.*xbar)

            if is_min_problem:
                new_linear_offsets[ndn_i] += scenario._PHW_on[ndn_i].value * scenario._Ws[ndn_i].value
            else:
                new_linear_offsets[ndn_i] += -1.* scenario._PHW_on[ndn_i].value * scenario._Ws[ndn_i].value
            new_objective_offset += xbar*xbar

        if not new_quadratic_offsets:
            update_quadratic = False
        elif self.prior_quadratic_offsets is None:
            update_quadratic = True
        else:
            update_quadratic = self.diff_offsets(self.prior_quadratic_offsets, new_quadratic_offsets)

        if update_quadratic:
            #print("In PersistentObjectiveHandler, need to update quadratic terms")
            self.update_quadratic(scenario, new_quadratic_offsets, new_linear_offsets, new_objective_offset)

        else:
            #print("In PersistentObjectiveHandler, NO need to update quadratic terms")
            self.update_linear(scenario, new_linear_offsets, new_objective_offset)

        self.prior_quadratic_offsets = new_quadratic_offsets
        self.prior_linear_offsets = new_linear_offsets
        self.prior_objective_offset =  new_objective_offset


class GurobiPersistentObjectiveHandler(PersistentObjectiveHandler):

    def set_solver_original_obj(self, scenario):
        self.solver_original_obj = scenario._solver_plugin._solver_model.getObjective()
        self.solver_obj_sense = scenario._solver_plugin._solver_model.ModelSense

    def update_quadratic(self, scenario, new_quadratic_offsets, new_linear_offsets, new_objective_offset):
        scenario_solver = scenario._solver_plugin

        gurobi_expr = 0.

        for ndn_i, var in scenario._nonant_indexes.items():
            grb_xvar = scenario_solver._pyomo_var_to_solver_var_map[var]
            gurobi_expr += new_linear_offsets[ndn_i] * grb_xvar
            if ndn_i in new_quadratic_offsets:
                gurobi_expr += new_quadratic_offsets[ndn_i] * grb_xvar * grb_xvar

        scenario_solver._solver_model.setObjective(self.solver_original_obj + gurobi_expr + new_objective_offset, sense = self.solver_obj_sense)

    def update_linear(self, scenario, new_linear_offsets, new_objective_offset):

        prior_linear_offsets = self.prior_linear_offsets
        scenario_solver = scenario._solver_plugin

        ## this can happen on iter 1
        if prior_linear_offsets is None:
            prior_linear_offsets = {}
            for ndn_i, var in scenario._nonant_indexes.items():
                prior_linear_offsets[ndn_i] = 0.

        for ndn_i, var in scenario._nonant_indexes.items():
            grb_xvar = scenario_solver._pyomo_var_to_solver_var_map[var]
            grb_xvar.obj += (new_linear_offsets[ndn_i] - prior_linear_offsets[ndn_i])
        scenario_solver._solver_model.ObjCon += (new_objective_offset - self.prior_objective_offset)

    def reset_objective(self, scenario):
        if self.solver_original_obj is None:
            raise RuntimeError("No objective to reset")
        scenario._solver_plugin._solver_model.setObjective(self.solver_original_obj, sense = self.solver_obj_sense)
