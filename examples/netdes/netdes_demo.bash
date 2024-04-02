#!/bin/bash

SOLVER=xpress_persistent
#SOLVER=cplex

mpiexec -host=localhost -n 6 python -u -m mpi4py netdes_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=10000.0 --instance-name=network-10-20-L-01 --max-iterations=100 --rel-gap=-1 --abs-gap=-1 --xhatshuffle --intra-hub-conv-thresh=-1.0 --presolve --lr-cross-scenario-cuts
#mpiexec -host=localhost -n 6 python -u -m mpi4py netdes_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=10000.0 --instance-name=network-10-20-L-01 --max-iterations=30 --rel-gap=0.01 --xhatshuffle --intra-hub-conv-thresh=-1.0 --presolve --cross-scenario-cuts
#mpiexec -host=localhost -n 6 python -u -m mpi4py netdes_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=10000.0 --instance-name=network-10-20-L-01 --max-iterations=30 --rel-gap=0.01 --xhatshuffle --intra-hub-conv-thresh=-1.0 --presolve --lagrangian
#mpiexec -host=localhost -n 6 python -u -m mpi4py netdes_cylinders.py --solver-name=${SOLVER} --max-solver-threads=1 --default-rho=10000.0 --instance-name=network-10-20-L-01 --max-iterations=30 --rel-gap=0.01 --xhatshuffle --intra-hub-conv-thresh=-1.0 --presolve --fwph
