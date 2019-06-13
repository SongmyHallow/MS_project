import alamopy
import math
import numpy as np
import matplotlib.pyplot as plt
import examples
ndata=20
# x = np.random.uniform([-2,-1],[2,1],(ndata,2))
x = np.random.random((20,3))
z = [0]*ndata

sim = examples.sixcamel
for i in range(ndata):
    z[i] = sim(x[i][0],x[i][1])
almsim = alamopy.wrapwriter(sim)
# print(x,z)
res = alamopy.doalamo(x,z,almname='cam6',monomialpower=(1,2,3,4,5,6),multi2power=(1,2),simulator=almsim, expandoutput=True,maxiter=20,cvfun=True)
print(res)