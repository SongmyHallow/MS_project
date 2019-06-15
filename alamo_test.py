import alamopy
import math
import numpy as np
import matplotlib.pyplot as plt
import examples
ndata=30
# x = np.random.uniform([-2,-1],[2,1],(ndata,2))
x = np.random.random((30,20))
z = [0]*ndata

sim = examples.sixcamel
for i in range(ndata):
    z[i] = sim(x[i][0],x[i][1])
almsim = alamopy.wrapwriter(sim)
# res = alamopy.doalamo(x,z,almname='cam6',monomialpower=(1,2,3,4,5,6),multi2power=(1,2),simulator=almsim, expandoutput=True,maxiter=20,cvfun=True)
res = alamopy.doalamo(x,z,almname='cam6',monomialpower=(1,2,3,4),multi2power=(1,2),simulator=almsim,expandoutput=True,maxiter=20)
print(res['model'])

expression = res['model']