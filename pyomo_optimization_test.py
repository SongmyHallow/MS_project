from pyomo.environ import *
from pyomo.opt import SolverFactory
model = ConcreteModel(name='test')

model.nVar = Param(initialize=4)
model.N = RangeSet(model.nVar)
model.x = Var(model.N, within=Binary)

model.obj = Objective(expr=summation(model.x))
model.cuts = ConstraintList()
opt = SolverFactory('baron')
opt.solve(model)

for i in range(5):
    expr = 0
    for j in model.x:
        if value(model.x[j]) < 0.5:
            expr+= model.x[j]
        else:
            expr += (1-model.x[j])
    model.cuts.add(expr >= 1)
    results = opt.solve(model)
    print("n======iteration\n",i)
    model.display()