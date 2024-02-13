# This software is distributed under the 3-clause BSD License.
# Replace baseparsers.py and enhance functionality.
# A class drived form pyomo.common.config is defined with
#   supporting member functions.
# NOTE: the xxxx_args() naming convention is used by amalgamator.py

""" Notes
The default for all 'with' options is False and we are dropping the with_
       (and we are dropping the `no` side that was in baseparsers.py)
       (so we are also dropping the use of with_)

Now you assemble the args you want and call the create_parser function,
   which returns an argparse object, E.g.:
parser = cfg.create_parser("myprog")
although most program use
cfg.parse_command_line("program_name")
which create the parser and does the parsing.

If you want to add args, you need to call the add_to_config function

If you want a required arg see num_scens_required() in this file.

If you want a positional arg, you have to DIY:
    parser = cfg.create_parser("tester")
    parser.add_argument(
            "num_scens", help="Number of scenarios", type=int,
        )
    args=parser.parse_args(['3', '--max-iterations', '99', '--solver-name', 'cplex'])
    print(f"{args.num_scens =}")
(Note: you can still attach it to a Config object, but that is also DIY)

    cfg.add_to_config("num_scens",
                         description="Number of Scenarios (required, positional)",
                         domain=int,
                         default=-1,
                         argparse=False)   # special
    # final special treatment of num_scens
    cfg.num_scens = args.num_scens


"""

import argparse
import pyomo.common.config as pyofig

# class to inherit from ConfigDict with a name field
class Config(pyofig.ConfigDict):
    # remember that the parent uses slots

    #===============
    def add_to_config(self, name, description, domain, default,
                      argparse=True,
                      complain=False,
                      argparse_args=None):
        """ Add an arg to the self dict.
        Args:
            name (str): the argument name, underscore seperated
            description (str): free text description
            domain (type): see pyomo config docs
            default (domain): value before argparse
            argparse (bool): if True put on command ine
            complain (bool): if True, output a message for a duplicate
            argparse_args (dict): args to pass to argpars (option; e.g. required, or group)
        """
        if name in self:
            if complain:
                print(f"Duplicate {name} will not be added to self.")
                # raise RuntimeError(f"Trying to add duplicate {name} to self.")
        else:
            c = self.declare(name, pyofig.ConfigValue(
                description = description,
                domain = domain,
                default = default))
            if argparse:
                if argparse_args is not None:
                    c.declare_as_argument(**argparse_args)
                else:
                    c.declare_as_argument()


    #===============
    def add_and_assign(self, name, description, domain, default, value, complain=False):
        """ Add an arg to the self dict and assign it a value
        Args:
             name (str): the argument name, underscore separated
            description (str): free text description
            domain (type): see pyomo config docs
            default (domain): probably unused, but here to avoid cut-and-paste errors
            value (domain): the value to assign
            complain (bool): if True, output a message for a duplicate
        """
        if name in self:
            if complain:
                print(f"Duplicate {name} will not be added to self by add_and_assign {value}.")
                # raise RuntimeError(f"Trying to add duplicate {name} to self.")
        else:
            self.add_to_config(name, description, domain, default, argparse=False)
            self[name] = value


    #===============
    def dict_assign(self, name, description, domain, default, value):
        """ mimic dict assignment
        Args:
            name (str): the argument name, underscore separated
            description (str): free text description
            domain (type): see pyomo config docs
            default (domain): probably unused, but here to avoid cut-and-paste errors
            value (domain): the value to assign
        """
        if name not in self:
            self.add_and_assign(name, description, domain, default, value)
        else:
            self[name] = value


    #===============
    def quick_assign(self, name, domain, value):
        """ mimic dict assignment with fewer args
        Args:
            name (str): the argument name, underscore separated
            domain (type): see pyomo config docs
            value (domain): the value to assign
        """
        self.dict_assign(name, f"field for {name}", domain, None, value)


    #===============
    def get(self, name, ifmissing=None):
        """ replcate the behavior of dict get"""
        if name in self:
            return self[name]
        else:
            return ifmissing

    def add_solver_specs(self, prefix=""):
        sstr = f"{prefix}_solver" if prefix != "" else "solver"
        self.add_to_config(f"{sstr}_name",
                            description= "solver name (default None)",
                            domain = str,
                            default=None)

        self.add_to_config(f"{sstr}_options",
                            description= "solver options; space delimited with = for values (default None)",
                            domain = str,
                            default=None)

    def _common_args(self):
        raise RuntimeError("_common_args is no longer used. See comments at top of config.py")

    def popular_args(self):
        self.add_to_config("max_iterations",
                            description="hub max iiterations (default 1)",
                            domain=int,
                            default=1)

        self.add_solver_specs(prefix="")

        self.add_to_config("seed",
                            description="Seed for random numbers (default is 1134)",
                            domain=int,
                            default=1134)

        self.add_to_config("default_rho",
                            description="Global rho for PH (default None)",
                            domain=float,
                            default=None)

        self.add_to_config("bundles_per_rank",
                            description="bundles per rank (default 0 (no bundles))",
                            domain=int,
                            default=0)

        self.add_to_config('verbose',
                              description="verbose output",
                              domain=bool,
                              default=False)

        self.add_to_config('display_progress',
                              description="display progress at each iteration",
                              domain=bool,
                              default=False)

        self.add_to_config('display_convergence_detail',
                              description="display non-anticipative variable convergence statistics at each iteration",
                              domain=bool,
                              default=False)

        self.add_to_config("max_solver_threads",
                            description="Limit on threads per solver (default None)",
                            domain=int,
                            default=None)

        self.add_to_config("intra_hub_conv_thresh",
                            description="Within hub convergence threshold (default 1e-10)",
                            domain=float,
                            default=1e-10)

        self.add_to_config("trace_prefix",
                            description="Prefix for bound spoke trace files. If None "
                                 "bound spoke trace files are not written.",
                            domain=str,
                            default=None)

        self.add_to_config("tee_rank0_solves",
                              description="Some cylinders support tee of rank 0 solves."
                              "(With multiple cylinders this could be confusing.)",
                              domain=bool,
                              default=False)

        self.add_to_config("auxilliary",
                            description="Free text for use by hackers (default '').",
                            domain=str,
                            default='')

    def ph_args(self):
        self.add_to_config("linearize_binary_proximal_terms",
                              description="For PH, linearize the proximal terms for "
                              "all binary nonanticipative variables",
                              domain=bool,
                              default=False)


        self.add_to_config("linearize_proximal_terms",
                              description="For PH, linearize the proximal terms for "
                              "all nonanticipative variables",
                              domain=bool,
                              default=False)


        self.add_to_config("proximal_linearization_tolerance",
                            description="For PH, when linearizing proximal terms, "
                            "a cut will be added if the proximal term approximation "
                            "is looser than this value (default 1e-1)",
                            domain=float,
                            default=1.e-1)

        self.add_to_config("initial_proximal_cut_count",
                            description="For PH, when linearizing proximal terms, "
                            "the is the number of additional proximal cuts initially "
                            "added (default 0)",
                            domain=int,
                            default=0)

    def make_parser(self, progname=None, num_scens_reqd=False):
        raise RuntimeError("make_parser is no longer used. See comments at top of config.py")


    def num_scens_optional(self):
        self.add_to_config(
            "num_scens",
            description="Number of scenarios (default None)",
            domain=int,
            default=None,
        )

    def num_scens_required(self):
        # required, but not postional
        self.add_to_config(
            "num_scens",
            description="Number of scenarios (default None)",
            domain=int,
            default=None,
            argparse_args = {"required": True}
        )


    def _basic_multistage(self, progname=None, num_scens_reqd=False):
        raise RuntimeError("_basic_multistage is no longer used. See comments at top of config.py")

    def add_branching_factors(self):
        self.add_to_config("branching_factors",
                            description="Spaces delimited branching factors (e.g., 2 2)",
                            domain=pyofig.ListOf(int, pyofig.PositiveInt),
                            default=None)


    def make_multistage_parser(self, progname=None):
        raise RuntimeError("make_multistage_parser is no longer used. See comments at top of config.py")

    def multistage(self):
        self.add_branching_factors()
        self.popular_args()


    #### EF ####
    def make_EF2_parser(self, progname=None, num_scens_reqd=False):
        raise RuntimeError("make_EF2_parser is no longer used. See comments at top of config.py")

    def _EF_base(self):

        self.add_solver_specs(prefix="EF")

        self.add_to_config("EF_mipgap",
                           description="mip gap option for the solver if needed (default None)",
                           domain=float,
                           default=None)


    def EF2(self):
        self._EF_base()
        self.add_to_config("num_scens",
                           description="Number of scenarios (default None)",
                           domain=int,
                           default=None)


    def make_EF_multistage_parser(self, progname=None, num_scens_reqd=False):
        raise RuntimeError("make_EF_multistage_parser is no longer used. See comments at top of config.py")

    def EF_multistage(self):

        self._EF_base()
        # branching factors???

    ##### common additions to the command line #####

    def two_sided_args(self):
        # add commands to  and also return the result

        self.add_to_config("rel_gap",
                            description="relative termination gap (default 0.05)",
                            domain=float,
                            default=0.05)

        self.add_to_config("abs_gap",
                            description="absolute termination gap (default 0)",
                            domain=float,
                            default=0.)

        self.add_to_config("max_stalled_iters",
                            description="maximum iterations with no reduction in gap (default 100)",
                            domain=int,
                            default=100)



    def mip_options(self):

        self.add_to_config("iter0_mipgap",
                            description="mip gap option for iteration 0 (default None)",
                            domain=float,
                            default=None)

        self.add_to_config("iterk_mipgap",
                            description="mip gap option non-zero iterations (default None)",
                            domain=float,
                            default=None)


    def aph_args(self):

        self.add_to_config('aph_gamma',
                            description='Gamma parameter associated with asychronous projective hedging (default 1.0)',
                            domain=float,
                            default=1.0)
        self.add_to_config('aph_nu',
                            description='Nu parameter associated with asychronous projective hedging (default 1.0)',
                            domain=float,
                            default=1.0)
        self.add_to_config('aph_frac_needed',
                            description='Fraction of sub-problems required before computing projective step (default 1.0)',
                            domain=float,
                            default=1.0)
        self.add_to_config('aph_dispatch_frac',
                            description='Fraction of sub-problems to dispatch at each step of asychronous projective hedging (default 1.0)',
                            domain=float,
                            default=1.0)
        self.add_to_config('aph_sleep_seconds',
                            description='Spin-lock sleep time for APH (default 0.01)',
                            domain=float,
                            default=0.01)


    def fixer_args(self):

        self.add_to_config('fixer',
                           description="have an integer fixer extension",
                           domain=bool,
                           default=False)

        self.add_to_config("fixer_tol",
                           description="fixer bounds tolerance  (default 1e-4)",
                           domain=float,
                           default=1e-2)



    def fwph_args(self):

        self.add_to_config('fwph',
                           description="have an fwph spoke",
                           domain=bool,
                           default=False)

        self.add_to_config("fwph_iter_limit",
                            description="maximum fwph iterations (default 10)",
                            domain=int,
                            default=10)

        self.add_to_config("fwph_weight",
                            description="fwph weight (default 0)",
                            domain=float,
                            default=0.0)

        self.add_to_config("fwph_conv_thresh",
                            description="fwph convergence threshold  (default 1e-4)",
                            domain=float,
                            default=1e-4)

        self.add_to_config("fwph_stop_check_tol",
                            description="fwph tolerance for Gamma^t (default 1e-4)",
                            domain=float,
                            default=1e-4)

        self.add_to_config("fwph_mipgap",
                            description="mip gap option FW subproblems iterations (default None)",
                            domain=float,
                            default=None)



    def lagrangian_args(self):

        self.add_to_config('lagrangian',
                              description="have a lagrangian spoke",
                              domain=bool,
                              default=False)

        self.add_to_config("lagrangian_iter0_mipgap",
                            description="lgr. iter0 solver option mipgap (default None)",
                            domain=float,
                            default=None)

        self.add_to_config("lagrangian_iterk_mipgap",
                            description="lgr. iterk solver option mipgap (default None)",
                            domain=float,
                            default=None)


    def lagranger_args(self):

        self.add_to_config('lagranger',
                            description="have a special lagranger spoke",
                              domain=bool,
                              default=False)

        self.add_to_config("lagranger_iter0_mipgap",
                            description="lagranger iter0 mipgap (default None)",
                            domain=float,
                            default=None)

        self.add_to_config("lagranger_iterk_mipgap",
                            description="lagranger iterk mipgap (default None)",
                            domain=float,
                            default=None)

        self.add_to_config("lagranger_rho_rescale_factors_json",
                            description="json file: rho rescale factors (default None)",
                            domain=str,
                            default=None)


    def ph_ob_args(self):

        self.add_to_config("ph_ob",
                            description="use PH to compute outer bound",
                            domain=bool,
                            default=False)
        self.add_to_config("ph_ob_rho_rescale_factors_json",
                            description="json file: rho rescale factors (default None)",
                            domain=str,
                            default=None)
        self.add_to_config("ph_ob_gradient_rho",
                            description="use gradient-based rho in PH OB",
                            domain=bool,
                            default=False)


    def xhatlooper_args(self):

        self.add_to_config('xhatlooper',
                              description="have an xhatlooper spoke",
                              domain=bool,
                              default=False)

        self.add_to_config("xhat_scen_limit",
                            description="scenario limit xhat looper to try (default 3)",
                            domain=int,
                            default=3)

    def xhatshuffle_args(self):

        self.add_to_config('xhatshuffle',
                           description="have an xhatshuffle spoke",
                           domain=bool,
                           default=False)

        self.add_to_config('add_reversed_shuffle',
                           description="using also the reversed shuffling (multistage only, default True)",
                           domain=bool,
                           default=False)

        self.add_to_config('xhatshuffle_iter_step',
                           description="step in shuffled list between 2 scenarios to try (default None)",
                           domain=int,
                           default=None)


    def mult_rho_args(self):

        self.add_to_config('mult_rho',
                              description="Have mult_rho extension (default False)",
                              domain=bool,
                              default=False)

        self.add_to_config('mult_rho_convergence_tolerance',
                            description="rhomult does nothing with convergence below this (default 1e-4)",
                              domain=float,
                              default=1e-4)

        self.add_to_config('mult_rho_update_stop_iteration',
                            description="stop doing rhomult rho updates after this iteration (default None)",
                            domain=int,
                            default=None)

        self.add_to_config('mult_rho_update_start_iteration',
                            description="start doing rhomult rho updates on this iteration (default 2)",
                            domain=int,
                            default=2)

    def mult_rho_to_dict(self):
        assert hasattr(self, "mult_rho")
        return {"mult_rho": self.mult_rho,
                "convergence_tolerance": self.mult_rho_convergence_tolerance,
                "rho_update_stop_iteration": self.mult_rho_update_stop_iteration,
                "rho_update_start_iteration": self.mult_rho_update_start_iteration,
                "verbose": False}



    def xhatspecific_args(self):
        # we will not try to get the specification from the command line

        self.add_to_config('xhatspecific',
                              description="have an xhatspecific spoke",
                              domain=bool,
                              default=False)


    def xhatxbar_args(self):

        self.add_to_config('xhatxbar',
                              description="have an xhatxbar spoke",
                              domain=bool,
                              default=False)




    def xhatlshaped_args(self):
        # we will not try to get the specification from the command line

        self.add_to_config('xhatlshaped',
                              description="have an xhatlshaped spoke",
                              domain=bool,
                              default=False)

    def wtracker_args(self):

        self.add_to_config('wtracker',
                              description="Use a wtracker extension",
                              domain=bool,
                              default=False)

        self.add_to_config('wtracker_file_prefix',
                            description="prefix for rank by rank wtracker files (default '')",
                            domain=str,
                            default='')

        self.add_to_config('wtracker_wlen',
                            description="max length of iteration window for xtracker (default 20)",
                            domain=int,
                            default=20)

        self.add_to_config('wtracker_reportlen',
                            description="max length of long reports for xtracker (default 100)",
                            domain=int,
                            default=100)

        self.add_to_config('wtracker_stdevthresh',
                            description="Ignore moving std dev below this value (default None)",
                            domain=float,
                            default=None)


    def slammax_args(self):
        # we will not try to get the specification from the command line

        self.add_to_config('slammax',
                            description="have a slammax spoke",
                              domain=bool,
                              default=False)


    def slammin_args(self):
        # we will not try to get the specification from the command line

        self.add_to_config('slammin',
                            description="have a slammin spoke",
                              domain=bool,
                              default=False)


    def cross_scenario_cuts_args(self):
        # we will not try to get the specification from the command line

        self.add_to_config('cross_scenario_cuts',
                              description="have a cross scenario cuts spoke",
                              domain=bool,
                              default=False)

        self.add_to_config("cross_scenario_iter_cnt",
                              description="cross scen check bound improve iterations "
                              "(default 4)",
                              domain=int,
                              default=4)

        self.add_to_config("eta_bounds_mipgap",
                              description="mipgap for determining eta bounds for cross "
                              "scenario cuts (default 0.01)",
                              domain=float,
                              default=0.01)


    def gradient_args(self):
         # we will not try to get the specification from the command line

        self.add_to_config("xhatpath",
                           description="path to npy file with xhat",
                           domain=str,
                           default='')
        self.add_to_config("grad_cost_file",
                           description="name of the gradient cost file (must be csv)",
                           domain=str,
                           default='')
        self.add_to_config("grad_rho_file",
                           description="name of the gradient rho file (must be csv)",
                           domain=str,
                           default='')
        self.add_to_config("grad_display_rho",
                           description="display rho during gradient calcs (default True)",
                           domain=bool,
                           default=True)
        self.add_to_config("grad_primal_thresh",
                           description="primal threshold for diff during gradient calcs",
                           domain=float,
                           default=0.001)
        self.add_to_config("grad_dual_thresh",
                           description="dual threshold for abs norm during gradient calcs",
                           domain=float,
                           default=0.1)
        self.add_to_config("grad_pd_thresh",
                           description="threshold for dual/primal during gradient calcs",
                           domain=float,
                           default=0.1)
        self.add_to_config("grad_order_stat",
                           description="order statistic for gradient based rho (must be between 0 and 1)",
                           domain=float,
                           default=0.5)

    def grad_rho_args(self):
        self.add_to_config("grad_whatpath",
                           description="path to csv file with what",
                           domain=str,
                           default='')
        self.add_to_config('grad_rho_setter',
                           description="use rho setter from a rho file",
                           domain=bool,
                           default=False)
        
        self.add_to_config("grad_rho_path",
                           description="csv file for the the grad based rho file (???)",
                           domain=str,
                           default='')
        self.add_to_config("grad_order_stat",
                           description="order statistic for rho: must be between 0 (the min) and 1 (the max); 0.5 is the average",
                           domain=float,
                           default=-1.0)
        self.add_to_config("grad_rho_relative_bound",
                           description="factor that bounds rho/cost",
                           domain=float,
                           default=1e3)

    def converger_args(self):
        self.add_to_config("use_norm_rho_converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)
        self.add_to_config("primal_dual_converger",
                            description="Use the primal dual converger",
                            domain=bool,
                            default=False)
        self.add_to_config("primal_dual_converger_tol",
                            description="Tolerance for primal dual converger (default 1e-2)",
                            domain=float,
                            default=1e-2)

    def tracking_args(self):
        self.add_to_config("tracking_folder",
                            description="Path of results folder (default results)",
                            domain=str,
                            default="results")
        self.add_to_config("ph_track_progress",
                            description="Adds tracking extension to all"
                            " ph opt cylinders (default False). Use --track_*"
                            " to specificy what and how to track."
                            " See mpisppy.utils.cfg_vanilla.add_ph_tracking for details",
                            domain=bool,
                            default=False)
        self.add_to_config("track_convergence",
                            description="Adds convergence tracking ie"
                                " gaps and bounds (default 0)",
                            domain=int,
                            default=0)
        self.add_to_config("track_xbars",
                            description="Adds xbar tracking (default 0)",
                            domain=int,
                            default=0)
        self.add_to_config("track_duals",
                            description="Adds w tracking (default 0)",
                            domain=int,
                            default=0)
        self.add_to_config("track_nonants",
                            description="Adds nonant tracking (default 0)",
                            domain=int,
                            default=0)
        self.add_to_config('track_scen_gaps',
                            description="Adds scenario gap tracking (default 0)",
                            domain=int,
                            default=0)

    def wxbar_read_write_args(self):
        self.add_to_config("init_W_fname",
                                description="Path of initial W file (default None)",
                                domain=str,
                                default=None)
        self.add_to_config("init_Xbar_fname",
                                description="Path of initial Xbar file (default None)",
                                domain=str,
                                default=None)
        self.add_to_config("init_separate_W_files",
                                description="If True, W is read from separate files (default False)",
                                domain=bool,
                                default=False)
        self.add_to_config("W_fname",
                                description="Path of final W file (default None)",
                                domain=str,
                                default=None)
        self.add_to_config("Xbar_fname",
                                description="Path of final Xbar file (default None)",
                                domain=str,
                                default=None)
        self.add_to_config("separate_W_files",
                                description="If True, writes W to separate files (default False)",
                                domain=bool,
                                default=False)


    #================
    def create_parser(self,progname=None):
        # seldom used
        if len(self) == 0:
            raise RuntimeError("create parser called before Config is populated")
        parser = argparse.ArgumentParser(progname, conflict_handler="resolve")
        self.initialize_argparse(parser)
        return parser

    #================
    def parse_command_line(self, progname=None):
        # often used, but the return value less so
        if len(self) == 0:
            raise RuntimeError("create parser called before Config is populated")
        parser = self.create_parser(progname)
        args = parser.parse_args()
        args = self.import_argparse(args)
        return args

#=================
if __name__ == "__main__":
    # a place for ad hoc testing by developers
    config = Config()
    config.popular_args() # populates self
    config.display()
    for i,j in config.items():
        print(i, j)
    print(dir(config))
    print(config._all_slots)
    print(config._domain)
    print(f"max_iterations {config['max_iterations']}")

    # most codes do not use create_parser; they use parse_command_line instead
    parser = config.create_parser("tester")
    parser.add_argument(
            "num_scens", help="Number of scenarios", type=int,
        )

    args=parser.parse_args(['3', '--max-iterations', '99', '--solver-name', 'cplex'])

    args = config.import_argparse(args)

    config.display()

    #parser.parse_args(['--help'])
