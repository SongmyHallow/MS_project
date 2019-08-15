'''
MS project algorithm
Muyi Song (muyis)
Latest update: 8/1/2019

command line syntax: python BlackBox.py (name of model file) (number of iterations)
                                            sys.argv[1]             sys.argv[2]
sampling methods:
1. Hammersley sequence
2. Ver der Corput sequence
3. Halton sequence
4. Latin Random

search algorithm:
1. Coordinate search (shuffle the order of variables every iteration)
=================================================================================
'''
import os
import sys
from pyomo.environ import *
from pyomo.opt import SolverFactory
"""
================================================================================
support functions
================================================================================
"""
def writeInput(filename,input_values):
    infile = open(filename, 'w')
    for val in input_values:
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
    
def genBlackBoxValue(filename,input_values):
    input_filename = "input.in"
    output_filename = "output.out"
    
    writeInput(input_filename,input_values)
    os.system('.\\'+filename)
    output_value = readOutput(output_filename)
    return output_value

def genBlackBoxValuesSeq(filename,point,sequence,index):
    input_filename = "input.in"
    output_filename = "output.out"
    output_values = []
    for val in sequence:
        input_copy = point[:]
        input_copy[index] = val
        writeInput(input_filename,input_copy)
        os.system('.\\'+filename)
        output_values.append(readOutput(output_filename))
    return output_values

def boundary_dic(labels,lb,ub):
    lowerBound = {labels[0]:lb}
    upperBound = {labels[0]:ub}
    return lowerBound,upperBound
'''
Regression, use alamopy package to get the numerical expression
:param list input_values: values of variables
:param list output_values: values generated by black box models
:paran real lowerBound: lower boundary of specific variable
:param real upperBound: upper boundary of specific variable
:ruturn: labels of variables and expression of the function
'''
def callAlamopy(input_values,output_values,lowBound,upBound):
    import alamopy
    alamo_result = alamopy.alamo(xdata=input_values,zdata=output_values,xmin=lowBound,xmax=upBound,monomialpower=(1,2))
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
  
"""
================================================================================
definition of the core class
================================================================================
"""
class blackBox(object):
    def __init__(self,name=None,cycles=0,radius=0,
                 numOfVar=0,
                 lowBound=[],upBound=[],
                 iniStart=[],backUpStart=[],actualStart=[],
                 minimalCoordinate=[],minimalValue=[],allValue=[],
                 totalCalls=0,calls=[],allCalls=[]):
        self.name = name
        self.cycles = cycles
        self.radius = radius
        self.numOfVar = numOfVar
        self.lowBound = lowBound
        self.upBound = upBound
        self.iniStart = iniStart
        self.backUpStart = backUpStart
        self.actualStart = actualStart
        self.totalCalls = totalCalls
        self.calls = calls
        self.allCalls = allCalls

        self.minimalCoordinate = minimalCoordinate
        self.minimalValue = minimalValue  
        self.allValue = allValue

    def clear(self):
        self.name=None
        self.cycles=0
        self.radius=0
        self.numOfVar = 0
        self.lowBound = []
        self.upBound = []
        self.iniStart = []
        self.backUpStart = []
        self.actualStart = []
    
    def showParameter(self):
        print("Name:",self.name)
        print("Total number of cycles:",self.cycles)
        print("Search radius:",self.radius)
        print("Number of variables:",self.numOfVar)
        print("lowBound:",self.lowBound)
        print("upBound:",self.upBound)
        print("Initial starting point from data file:",self.iniStart)
        print("Back up starting points:",self.backUpStart)
        print("Actual starting point is:",self.actualStart)

    def getCalls(self):
        print("Total call:",self.totalCalls)
        print("Calls list:",self.calls)

    def getResult(self):
        print("Optimal values:",self.minimalValue)
        print("Optimal points:",self.minimalCoordinate)
        
    '''
    Compile .c file
    :param string filename: name of c file (with out extension)
    ''' 
    def compileCode(self):
        os.system('gcc source_princetonlibgloballib/'+self.name+'.c -lm -o '+self.name)
        after_name = self.name+".exe"
        print("Blackbox Model Name: ",self.name)
        if(os.path.exists(after_name)):
            print("Compilation finished")
        else:
            print("Compilation failed")
    
    '''
    Read boundaries, starting points and number of variables
    :param string filename: name of c file (with out extension)
    '''
    def readDataFile(self):
        infile = open("problemdata/"+self.name+".problem.data",'r')
        lines = infile.readlines()
        # The first line
        for num in lines[0].split():
            self.numOfVar = int(num.strip())
        # The second line
        for i in lines[1].split():
            self.lowBound.append(float(i.strip()))
        # The third line
        for j in lines[2].split():
            self.upBound.append(float(j.strip()))
        # The fourth line
        for k in lines[3].split():
            self.iniStart.append(float(k.strip()))
        infile.close()

    '''
    Generate starting points that are ready to be chose
    '''
    def genBackupStart(self):
        import numpy as np    
        # TODO: Finally I want to generate 10 backup starting points, which can be modifed later
        for i in range(len(self.lowBound)):
            arr = np.linspace(self.lowBound[i],self.upBound[0],17)
            self.backUpStart.append(arr)
        self.backUpStart = np.transpose(self.backUpStart)[1:-1]
    
    '''
    Choose a starting point from backup and check if it is valid,
    otherwise remove that point from list and choose another one
    '''
    def genActualStart(self,method):
        import random
        if(method=="random"):
            self.actualStart = random.choice(self.backUpStart)
            output = genBlackBoxValue(self.name,self.actualStart)
            while(output == "INF"):
                self.backUpStart.remove(self.actualStart)
                self.actualStart = random.choice(self.backUpStart)
                output = genBlackBoxValue(self.name,self.actualStart)
        elif(method=="origin"):
            self.actualStart = self.iniStart

    '''
    Generate the boundary of specific variable at current iteration
    '''
    def genVariableBound(self,index):
        import random

        if(self.actualStart[index]-self.radius<self.lowBound[index]):
            lb = self.lowBound[index]
        else:
            lb = self.actualStart[index]-self.radius
        
        if(self.actualStart[index]+self.radius>self.upBound[index]):
            ub = self.upBound[index]
        else:
            ub = self.actualStart[index]+self.radius

        # self.tempLB = lb
        # self.tempUB = ub
        # offset = random.uniform(lb,ub)
        return lb,ub
    
    '''
    Generate sampling points
    1. The Halton Quasi Monte Carlo (QMC) Sequence
    2. The Hammersley Quasi Monte Carlo (QMC) Sequence
    3. The van der Corput Quasi Monte Carlo (QMC) sequence
    4. Latin Random Squares in M dimensions
    5. Adaptive sampling from package 'adaptive'
    '''
    def genSamplePoints(self,method,num,lowBound,upBound):
        from Sampling import halton_sequence,hammersley_sequence,van_der_corput,latin_random_sequence
        import sobol_seq
        if(method=="halton"):
            Xdata,_ = halton_sequence(lowBound,upBound,num)
        elif(method=="hammersley"):
            Xdata,_ = hammersley_sequence(lowBound,upBound,num)
        elif(method=="vander"):
            Xdata,_=van_der_corput(lowBound,upBound,num,2)
        elif(method=="latin"):
            Xdata,_ = latin_random_sequence(lowBound,upBound,num,1,1)
        elif(method=="sobol"):
            Xdata = sobol_seq.i4_sobol_generate(1,num)
        self.totalCalls+=num
        return Xdata

    '''
    Optimization, call baron by Pyomo to get optimal solution
    :param labels: labels of variables generated by alamopy
    :param expr: numerical expression generated by alamopy
    :param lowerBound
    :Param upperBound
    :param list startPoint: list of coordinate of the point
    :param integer index: index of the specific variable
    '''
    def callBaron(self,labels,expr,lowBound,upBound,index):
        model = ConcreteModel(name='blackbox')
        lBound_dic,uBound_dic = boundary_dic(labels,lowBound,upBound)
        def fb(model,i):
            return (lBound_dic[i],uBound_dic[i])
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
        tempPoint = self.actualStart[:]
        try:
            tempPoint[index] = value(model.x[labels[0]])
            # print(value(model.x[labels[index]]))
        except:
            pass
        tempMinimal = value(model.obj)
        return tempPoint,tempMinimal

    '''
    Update the flag indicating the accuracy of surrogate model
    '''
    def updateFlag(self,tempPoint,tempMinimal):
        boxVal = genBlackBoxValue(self.name,tempPoint)
        print("Box value:",boxVal)
        print("Temp minimal value:",tempMinimal)

        if(boxVal==0):
            boxVal+=1e-5

        ratio = tempMinimal/boxVal
        if(ratio<=1.1 and ratio>=0.9):
            if(len(self.minimalValue)<1 or boxVal<self.minimalValue[-1]):
                self.actualStart = tempPoint
                self.minimalValue.append(boxVal)
                self.minimalCoordinate.append(tempPoint)
                self.calls.append(self.totalCalls)
            self.radius *= 2.0
            return True,boxVal
        else:
            self.radius *= 0.85
            return False,boxVal
    
    def checkEnd(self):
        if(len(self.minimalValue)>1 and self.minimalValue[-2]-self.minimalValue[-1]<1e-5):
            return True
        else:
            return False

    '''
    make a plot, optimal values vs calls of model
    :param list values
    :param list calls
    :param string name: name of model
    '''
    def makePlot(self):
        import matplotlib.pyplot as plt
        self.getResult()
        self.getCalls()
        plt.plot(self.allCalls, self.allValue, '-o')
        plt.xlabel("Number of calls")
        plt.ylabel("Optimal values")
        for x, y in zip(self.allCalls, self.allValue):
            plt.text(x, y+0.3, '%.5f'%y, ha='center', va='bottom', fontsize=10.5)
        plt.title(self.name)
        plt.savefig("plots\\"+self.name+".png")
        print("Plot of model "+ self.name +" is saved")

    '''
    Write data into csv file
    '''
    def make_csv(name,values,calls,time,points,cycle):
        from pandas import DataFrame
        import csv
        csvfile = open('experimentData.csv','a+',newline='')
        fieldsnames = ['model_name','time','cycle','values','calls','point']
        # writer = csv.writer(csvfile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer = csv.DictWriter(csvfile,fieldnames=fieldsnames)
        # values = list_int2str(values)
        # calls = list_int2str(calls)
        # writer.writerow([name,calls[-1]]+values)
        if(len(points)>0):
            writer.writerow({
                'model_name':name,
                'time':time,
                'cycle':cycle,
                'values':values[-1],
                'calls':calls[-1],
                'point':points[-1]
            })
        else:
            writer.writerow({
                'model_name':name,
                'time':time,
                'cycle':cycle,
                'values':values[-1],
                'calls':calls[-1]
            })
        csvfile.close()
    
    '''
    Algorithms
    '''
    '''
    1. Coordinate search
    :param integer cycles: execution loops
    :param list startPoint: 
    '''
    def coordinateSearch(self):
        import random
        for cycle in range(self.cycles):
            print("The No.",cycle+1,"Cycle")
            shuffleOrder = list(range(self.numOfVar))
            random.shuffle(shuffleOrder)
            for indexOfVar in shuffleOrder:
                print("The No.",indexOfVar,"Variable")
                
                flag = False
                numOfSample = 20
                while(flag==False):
                    lb,ub = self.genVariableBound(indexOfVar)
                    Xdata = self.genSamplePoints("vander",numOfSample,lb,ub)
                    ydata = genBlackBoxValuesSeq(self.name,self.actualStart,Xdata,indexOfVar)
                    labels,expr = callAlamopy(Xdata,ydata,lb,ub)
                    tempPoint,tempMinimal = self.callBaron(labels,expr,lb,ub,indexOfVar)
                    flag,boxVal = self.updateFlag(tempPoint,tempMinimal)
                    print("Flag",flag)

                    if(flag==False):
                        numOfSample += 10
                    else:
                        numOfSample -=5
                        self.allCalls.append(self.totalCalls)
                        self.allValue.append(boxVal)
            if(self.checkEnd()==True):
                    return

"""
================================================================================
main function
================================================================================
"""
if __name__ == "__main__":
    fileName = sys.argv[1]
    cycles = int(sys.argv[2])

def main():
    import time
    box = blackBox(name=fileName,cycles=cycles,radius=1.0)
    box.compileCode()
    box.readDataFile()
    box.genBackupStart()
    box.genActualStart("random")
    # for start in box.backUpStart:
        # box.actualStart = start
    box.coordinateSearch()
    # box.showParameter()
    # box.getResult()
    box.makePlot()

main()