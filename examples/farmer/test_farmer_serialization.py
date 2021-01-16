from farmer import scenario_creator

from pyutilib.misc.timing import TicTocTimer
from mpisppy.utils.model_compiler.model_serializer import *
from pyomo.environ import SolverFactory

tt = TicTocTimer()
g = SolverFactory('gurobi_direct')

sn = 'Scenario2'

f = scenario_creator(sn)

f.pprint()
tt.tic("farmer before serialization")
g.solve(f, tee=True)

serialize_model(f, verbose=True, keep_expressions=False)
deserialize_model(f, verbose=True)

f.pprint()
tt.tic("farmer after serialization")
g.solve(f, tee=True)

print("\n")
import ray
ray.init()

f = scenario_creator(sn)

serialize_model(f, verbose=True, keep_expressions=True)
tt.tic()
f_id = ray.put(f)
tt.toc("serialized farmer")

f_deserialized = ray.get(f_id)
deserialize_model(f_deserialized, verbose=True)
tt.toc("deserialized farmer")

f_deserialized.pprint()
print("farmer after deserialization")
g.solve(f_deserialized, tee=True)

'''
for k in f_deserialized.DevotedAcreage:
    print(k)
'''

ray.shutdown()
