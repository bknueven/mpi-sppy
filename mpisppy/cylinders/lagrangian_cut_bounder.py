# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import math
import numpy as np
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.repn.standard_repn import generate_standard_repn
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.utils.sputils import is_persistent, find_active_objective
from mpisppy import MPI

class LagrangianCutSpoke(LagrangianOuterBound):

    converger_spoke_char = 'L'

    def make_windows(self):
        if not hasattr(self.opt, "local_scenarios"):
            raise RuntimeError("Provided SPBase object does not have local_scenarios attribute")

        if len(self.opt.local_scenarios) == 0:
            raise RuntimeError("Rank has zero local_scenarios")

        vbuflen = 2
        for s in self.opt.local_scenarios.values():
            vbuflen += len(s._mpisppy_data.nonant_indices)

        self.nonant_length = self.opt.nonant_length
        self.total_number_scenarios = len(self.opt.all_scenario_names)

                           #         send               
                           # |S| * ( const + nonants )                           , 2 + nonants for every local scenario
        self._make_windows(
            # send bound + |S| * ( const + |nonants| )
            1 + self.total_number_scenarios * (1 + self.nonant_length),
            # recieve 2 + |local scenarios| * |nonants|  
            vbuflen,
        )
        self._locals = np.zeros(vbuflen + 1)

        # over load the _bound attribute here
        # so the rest of the class works as expected
        # first float will be the bound we're sending
        # indices 1:-1 will be the scenario lagrangian bound and weights 
        # the last index will be the serial number
        self._bound = np.zeros(1 + self.total_number_scenarios * (1 + self.nonant_length) + 1)

    def lagrangian_prep(self):
        # TODO: want to extract the objective coefficients *HERE*
        #       before the Ws are attached in super().lagrangian_prep()
        # TODO: extract c (coefficients of nonants); send to hub??
        # TODO: maybe instead of using allgather we should just verify
        #       that the coefficients are the same in every node of the
        #       scenario tree.
        value = pyo.value
        self.c_vector = None
        for k, obj in self.opt.saved_objectives.items():
            varid_to_nonant_index = (
                self.opt.local_scenarios[k]._mpisppy_data.varid_to_nonant_index
            )
            repn = generate_standard_repn(obj.expr, compute_values=False, quadratic=False)

            # non-linear non-nonant vars *might* be okay, but
            # non-linear nonant vars are definitely an issue.
            # So we'll check in this case
            nonlinear_var_ids = set(id(v) for v in repn.nonlinear_vars)
            if nonlinear_var_ids and (
                    nonlinear_var_ids.intersection(set(varid_to_nonant_index.keys()))
                ):
                raise RuntimeError(f"Found nonlinear nonants in objective function."
                                    " This is not supported by the LagrangianCutSpoke")

            # just linear terms on the nonants, find what they are
            ndn_id_to_coefs = {}
            for c, v in zip(repn.linear_coefs, repn.linear_vars):
                if id(v) in varid_to_nonant_index:
                    ndn_id_to_coefs[id(v)] = value(c)

            # TODO: for multistage this will need to be broken down by node
            c_vector_k = np.fromiter(
                (ndn_id_to_coefs[var_id] for var_id in varid_to_nonant_index),
                dtype=float,
                count=len(varid_to_nonant_index),
            )

            if self.c_vector is None:
                self.c_vector = c_vector_k
                # TODO: should we just break here?
                # break
            else:
                assert np.allclose(self.c_vector, c_vector_k)

        # TODO: for multistage this will need to be broken down by node
        global_c_vector = np.zeros(len(self.c_vector))
        self.cylinder_comm.Allreduce(self.c_vector, global_c_vector, op=MPI.SUM)
        assert np.allclose(self.c_vector, global_c_vector / self.cylinder_comm.size)

        super().lagrangian_prep()


    def lagrangian(self):
        bound = super().lagrangian()

        scenario_Ws_bound_slices = {}
        for idx, sn in enumerate(self.opt.all_scenario_names):
            scenario_Ws_bound_slices[sn] = slice(1 + idx*(self.nonant_length + 1), (1 + (idx+1)*(self.nonant_length + 1)))

        # NOTE: for *bundles* we would like to just do everything by scenario
        #       This works fine if we use the actual objective function, but for
        #       MIPs we would like to use the subproblem *bound*.
        # NOTE: Can we just compute the gap (inner - outer) and distribute it to all the scenarios?
        assert not self.opt.bundling

        # populate the local portion of the vector
        for sn, s in self.opt.local_scenarios.items():
            sn_Ws_bound = self._bound[scenario_Ws_bound_slices[sn]] 
            for ci, ix in enumerate(s._mpisppy_data.nonant_indices):
                sn_Ws_bound[ci] = s._mpisppy_model.W[ix]._value
            ci += 1
            sn_Ws_bound[ci] = s._mpisppy_data.outer_bound
            ci += 1
            assert ci == len(sn_Ws_bound)
            # TODO: not clear is this a good idea *here*
            sn_Ws_bound[:-1] += self.c_vector
            sn_Ws_bound *= s._mpisppy_probability


        # TODO: We should probably convert this into an Allgatherv or Allreducev,
        #       but proceeding by scenario is safer, and allows for easy indexing
        #       if the scenarios are scattered across ranks in a weird way
        for idx, sn in enumerate(self.opt.all_scenario_names):
            source_rank = self.opt.scenario_names_to_rank["ROOT"][sn]
            self.opt.comms["ROOT"].Bcast(self._bound[scenario_Ws_bound_slices[sn]], root=source_rank)

        return bound
