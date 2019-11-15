import os
import sys
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from Sampling import halton_sequence,hammersley_sequence,van_der_corput,latin_random_sequence,sobol_sequence
from pyomo.environ import *
from pyomo.opt import SolverFactory
import alamopy
'''
Definition of the core class
'''
class blackBox(object):
    def __init__(self, name=None,cycles=0,radius=None,numOfSample=None,numOfVar=0,
                 lowBound=[] ,upBound=[],
                 iniLocation=[]):
        # name of model
        self.name = name
        # number of cycles
        self.cycles = cycles
        # number of samples and variables
        self.numOfSample = numOfSample
        self.numOfVar = numOfVar

        # search radius
        self.radius = radius
        self.direction = None
        
        # Boundaries of variables
        self.lowBound = lowBound
        self.upBound = upBound
        # Initial starting point
        self.iniLocation = iniLocation
        self.actualLocation = iniLocation

        # Initialize the number of evaluations
        self.totalCall = 0

        # Store results
        self.optimalValues = []
        self.optimalPoints = []
        self.calls = []

    '''
    Read boundaries, starting points and number of variables
    :param string filename: name of c file (with out extension)
    '''
    def readDataFile(self):
        path = self.name + ".problem.data"
        infile = open(path,'r')
        lines = infile.readlines()
        # The first line
        for num in lines[0].split():
            self.numOfVar = int(num.strip())
            self.radius = [2] * self.numOfVar
        # The second line
        for i in lines[1].split():
            self.lowBound.append(float(i.strip()))
        # The third line
        for j in lines[2].split():
            self.upBound.append(float(j.strip()))
        # The fourth line
        for k in lines[3].split():
            self.iniLocation.append(float(k.strip()))
        infile.close()


'''
Helper functions
'''
def writeInput(filename,values):
    infile = open(filename, 'w')
    for val in values:
        infile.write(str(val)+'\n')
    infile.close()

def readOutput(filename):
    readfile = open(filename, 'r')
    line = readfile.readline()
    if(line.strip() == "1.#INF00000000000"):
        return "INF"
    else:
        output_value = float(line.strip())
        return output_value

def evaluate(box,Xdata,index):

    if type(Xdata) == int or type(Xdata) == float:
        ydata = []
        tempLocation = box.actualLocation[:]
        tempLocation[index] = Xdata
        writeInput(filename='input.in',values=tempLocation)
        os.system('.\\'+box.name)
        ydata.append(readOutput(filename='output.out'))
        return ydata[0]
    else:
        ydata = []
        for val in Xdata:
            tempLocation = box.actualLocation[:]
            tempLocation[index] = val
            writeInput(filename='input.in',values=tempLocation)
            os.system('.\\'+box.name)
            ydata.append(readOutput(filename='output.out'))
        return ydata


def callAlamopy(Xdata,ydata,lowBound,upBound):
    # print(input_values)
    alamo_result = alamopy.alamo(xdata=Xdata,zdata=ydata,xmin=lowBound,xmax=upBound,monomialpower=(1,2))
#     print("===============================================================")
#     print("ALAMO results")
#     print("===============================================================")
#     print("#Model expression: ",alamo_result['model'])
#     print("#Rhe sum of squared residuals: ",alamo_result['ssr'])
#     print("#R squared: ",alamo_result['R2'])
#     print("#Root Mean Square Error: ",alamo_result['rmse'])
#     print("---------------------------------------------------------------")
    labels = alamo_result['xlabels']
    expr = alamo_result['f(model)']
    return labels,expr

def callBaron(box,labels,expr,index):
        model = ConcreteModel(name='blackbox')
        lowBound_dic = {labels[0]:box.lowBound[index]}
        upBound_dic = {labels[0]:box.upBound[index]}

        def fb(model,i):
            return (lowBound_dic[i],upBound_dic[i])
        model.A = Set(initialize=labels)
        model.x = Var(model.A,within=Reals,bounds=fb)
        
        def objrule(model):
            var_lst = []
            for var_name in model.x:
                var_lst.append(model.x[var_name])
            return expr(var_lst)
        model.obj = Objective(rule=objrule,sense=minimize)
        opt = SolverFactory('baron')
        solution = opt.solve(model)
        # solution.write()
        # model.pprint()
        # model.display()

        tempPoint = []
        for i in box.actualLocation:
            tempPoint.append(i)

        try:
            tempPoint[index] = value(model.x[labels[0]])
            # print(value(model.x[labels[index]]))
        except:
            pass
        tempMinimal = value(model.obj)
        return tempPoint,tempMinimal


def coordinateSearch(filename,cycles,sample_method,sample_ini):
    infile = 'input.in'
    outfile = 'output.out'

    box = blackBox(name=filename,cycles=cycles)
    box.readDataFile()
    box.numOfSample = [sample_ini] * box.numOfVar

    for cycle in range(cycles):
        print('The No.',cycle+1,'cycle...')

        for i in range(box.numOfVar):
            print('The No.',i+1,'variable...')

            tempXdata = []
            tempydata = []

            preventInifinite = 0
            while(True):
                preventInifinite += 1
                r = box.radius[i]
                # lower bound
                if box.actualLocation[i] - r < box.lowBound[i]:
                    lb = box.lowBound[i]
                else:
                    lb = box.actualLocation[i] - r
                # upper bound
                if box.actualLocation[i] + r > box.upBound[i]:
                    ub = box.upBound[i]
                else:
                    ub = box.actualLocation[i] + r

                # Sampling along the single direction within [lb,ub]
                if(sample_method=="vander"):
                    Xdata,_ = halton_sequence(lb,ub,box.numOfSample[i])
                elif(sample_method=="hammersley"):
                    Xdata,_ = hammersley_sequence(lb,ub,box.numOfSample[i])
                elif(sample_method=="halton"):
                    Xdata,_=van_der_corput(lb,ub,box.numOfSample[i],2)
                elif(sample_method=="latin"):
                    Xdata,_ = latin_random_sequence(lb,ub,box.numOfSample[i],1,1)
                elif(sample_method=="sobol"):
                    Xdata = sobol_sequence(lb,ub,1,box.numOfSample[i])

                box.totalCall += box.numOfSample[i]

                # generate output values and read out
                ydata = evaluate(box,Xdata,i)

                tempXdata += Xdata
                tempydata += ydata

                labels,expr = callAlamopy(tempXdata,tempydata,box.lowBound[i],box.upBound[i])
                tempPoint,tempOptimal = callBaron(box,labels,expr,i)
                print('Baron point:',tempPoint,'...')
                print('Baron value:',tempOptimal,'...')

                boxVal = evaluate(box,tempPoint[i],i)
                print('evalution value:',boxVal,'...')
                if boxVal == 0:
                    boxVal += 1e-5
                ratio = tempOptimal / boxVal

                # Exit while loop
                if (ratio>0.5 and ratio<1.5) or (np.abs(tempOptimal-boxVal)<0.1):
                    box.actualLocation = tempPoint
                    if len(box.optimalValues)<1 or boxVal<box.optimalValues[-1]:
                        box.optimalValues.append(boxVal)
                        box.optimalPoints.append(tempPoint)
                        box.calls.append(box.totalCall)
                        print('================================================')
                        print('Current optimal values:',box.optimalValues,'...')
                        print('Number of evaluations used until:',box.totalCall,'...')
                        print('================================================')

                    box.radius[i] *= 2

                    # decrease the number of sampling
                    if box.numOfSample[i] > 6:
                        box.numOfSample[i] = int(box.numOfSample[i]*0.5)
                    else:
                        box.numOfSample[i] = 3
                    break
                else:
                    box.numOfSample[i] += 4
                    box.radius[i] *= 0.8

                    if preventInifinite > 10:
                        break

        # Finish condition
        if len(box.optimalValues)>1 and box.optimalValues[-2]-box.optimalValues[-1]<1e-4:
            return box

    return box

'''
==============================================================
'''

def makePlot(box,sample_method):
    plt.plot(box.calls,box.optimalValues,'-o')
    plt.xlabel('Number of evaluations')
    plt.ylabel('Objective values')
    plt.title(box.name+'('+sample_method+')')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('plots\\'+box.name+'('+sample_method+')'+'.eps')
    print('Plot is saved')

def main():
    box = coordinateSearch(filename,cycles,sample_method,sample_ini)
    makePlot(box,sample_method)
    return

if __name__ == "__main__":
    filename = sys.argv[1]
    cycles = int(sys.argv[2])
    sample_method = sys.argv[3]
    sample_ini = int(sys.argv[4])

main()