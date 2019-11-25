import os
import sys
import win32com.client as win32
import numpy as np
import time
import random
from collections import Counter
from pyomo.environ import *
from pyomo.opt import SolverFactory
from Sampling import halton_sequence,hammersley_sequence,van_der_corput,latin_random_sequence,sobol_sequence
import alamopy

'''
Definition of the core class
'''
class blackBox(object):
    def __init__(self, cycles=0,radius=None,numOfSample=None,numOfVar=0,
                 lowBound=[] ,upBound=[],
                 iniLocation=[]):
        # number of cycles
        self.cycles = cycles
        # number of samples and variables
        self.numOfSample = numOfSample
        self.numOfVar = numOfVar

        # search radius
        self.radius = radius
        self.direction = ['both'] * numOfVar
        
        # Boundaries of variables
        self.lowBound = lowBound
        self.upBound = upBound
        # Initial starting point
        self.iniLocation = iniLocation
        self.actualLocation = None

        # Initialize the number of evaluations
        self.totalCall = 0

        # Store results
        self.optimalValues = []
        self.optimalPoints = []
        self.calls = []


'''
Helper functions
'''

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

def callBaron(box,labels,expr,lowBound,upBound,index):
        model = ConcreteModel(name='blackbox')
        lowBound_dic = {labels[0]:lowBound}
        upBound_dic = {labels[0]:upBound}

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

        # tempPoint = box.actualLocation[:]
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

def hy_distinguish(hysolver):
        i=4
        P=np.linspace(0,0,num=1001)
        k=0
        timeover=0
        while k<2:
            i+=1
            solveboolean = 1-hysolver.issolving
            P[i] = solveboolean
            k = P[i-2] + P[i-1] + P[i]
            time.sleep(1)
            if P[1000]==1:
                timeover=1
                break
        return timeover	

'''
return: objective value, flag(if predefined constaints are satisfied)
'''
def hy_Object(hyCase, hysolver, variable):
        error=0
        error_type = np.array([0,0,0])
        butane = 100 - (variable[4] + variable[5] + variable[6] + variable[7])
        # Pre-defined constraints, if not satisfied, a flag will be returned
        predefined_flag = True

        if variable[0]<=variable[1] and variable[1]<=variable[2] and variable[2]<=variable[3] and variable[4] >= 0:
            hyCase.Flowsheet.operations.Item('optimization').Cell('C2').cellvalue = variable[0]*100   #LP multiply 100 for making unit as bar
            hyCase.Flowsheet.operations.Item('optimization').Cell('C3').cellvalue = variable[1]*100   #MP1
            hyCase.Flowsheet.operations.Item('optimization').Cell('C4').cellvalue = variable[2]*100   #MP2
            hyCase.Flowsheet.operations.Item('optimization').Cell('C5').cellvalue = variable[3]*100   #HP
            hyCase.Flowsheet.operations.Item('optimization').Cell('C6').cellvalue = variable[4]   	  #Nitrogen
            hyCase.Flowsheet.operations.Item('optimization').Cell('C7').cellvalue = variable[5]       #Methane
            hyCase.Flowsheet.operations.Item('optimization').Cell('C8').cellvalue = variable[6]       #Ethane
            hyCase.Flowsheet.operations.Item('optimization').Cell('C9').cellvalue = variable[7]       #propane
            hyCase.Flowsheet.operations.Item('optimization').Cell('C10').cellvalue = butane    
            timeover=hy_distinguish(hysolver)
            # --there are infeasible signs?
            try:
                V1=hyCase.Flowsheet.operations.Item('optimization').Cell('I2').cellvalue
                V2=hyCase.Flowsheet.operations.Item('optimization').Cell('I3').cellvalue
                V3=hyCase.Flowsheet.operations.Item('optimization').Cell('I4').cellvalue
                V4=hyCase.Flowsheet.operations.Item('optimization').Cell('I5').cellvalue
                V5=hyCase.Flowsheet.operations.Item('optimization').Cell('I6').cellvalue
            except:
                hyCase.Flowsheet.operations.Item('optimization').Cell('C11').cellvalue = 3
                timeover=hy_distinguish(hysolver)
                error_type[1]=1
                if np.dot(error_type,error_type) !=0:
                    error=1
                    OBJ = np.random.random_sample()*10**30
                    return OBJ,predefined_flag
                else:
                    pass
            
            if abs(V4-V5)>0.1:
                error_type[1]=1
                error =1
                OBJ = np.random.random_sample()*10**30
                return OBJ,predefined_flag

            MTD_HX1 = hyCase.Flowsheet.operations.Item('optimization').Cell('C14').cellvalue
            MTD_HX2 = hyCase.Flowsheet.operations.Item('optimization').Cell('C16').cellvalue

            if MTD_HX1 <=2.85 or MTD_HX2<=2.85:
                hyCase.Flowsheet.operations.Item('optimization').Cell('C11').cellvalue = 3
                timeover=hy_distinguish(hysolver)
                if MTD_HX1 <=2.85 or MTD_HX2<=2.85:
                    error_type[0]=1
                    if np.dot(error_type,error_type) !=0:
                        error=1
                        OBJ = np.random.random_sample()*10**30
                        return OBJ,predefined_flag
            
            VF1 = hyCase.Flowsheet.operations.Item('optimization').Cell('I8').cellvalue
            VF2 = hyCase.Flowsheet.operations.Item('optimization').Cell('I9').cellvalue
            VF3 = hyCase.Flowsheet.operations.Item('optimization').Cell('I10').cellvalue
            VF4 = hyCase.Flowsheet.operations.Item('optimization').Cell('I11').cellvalue
            VF5 = hyCase.Flowsheet.operations.Item('optimization').Cell('I12').cellvalue

            if VF1!=1 or VF2!=1 or VF3!=1 or VF4!=1 or VF5 !=1:
                error_type[2]=1
                if np.dot(error_type,error_type) !=0:
                    error =1
                    OBJ = np.random.random_sample()*10**30
                    return OBJ,predefined_flag

            # -- Transfer output variables (HYSYS --> python)
            OBJ  = hyCase.Flowsheet.operations.Item('optimization').Cell('G10').cellvalue # E Total Energy Consumption / LNG Production(Ton per day)
            return OBJ,predefined_flag
        # Predefined constraints are not satisfied
        else:
            error=1
            OBJ = np.random.random_sample()*10**30

            predefined_flag = False
            return OBJ, predefined_flag



def HYSYS(cycles,sample_method,sample_ini):
    '''maximum number of function evaluation is 2200'''
    ''' Connecting to the Aspen Hysys App just one time during optimization'''      
    print(' # Connecting to the Aspen Hysys App ... ')
    hyapp    = win32.Dispatch('HYSYS.Application')			   # Connecting to the Application
    hyCase   = hyapp.ActiveDocument                          # Access to active document
    hysolver = hyCase.Solver

    print(' # Initialize the black box model...')
    box = blackBox(cycles=cycles,numOfVar=8)
    box.lowBound=np.array([0.3000,0.7500,1.8750,4.6750,0.8590,2.5970,2.5410,3.9110])
    box.upBound=np.array([5.7000,14.2500,35.6250,88.8250,16.3210,49.3430,48.2790,74.3090])
    box.iniLocation = (box.lowBound+box.upBound)/2
    box.actualLocation = box.iniLocation

    #Initialization
    box.numOfSample = [sample_ini] * box.numOfVar
    box.radius = [1] * box.numOfVar

    for cycle in range(box.cycles):
        print('The No.',cycle+1,'cycle...')

        for i in range(box.numOfVar):
            print('The No.',i+1,'variable...')

            while(True):
                # used to store available data points
                tempXdata = []
                tempydata = []

                while(True):
                    '''
                    Search from the starting point, left to right
                    '''
                    if type(box.radius) == int:
                        r = box.radius
                    elif type(box.radius) == list:
                        r = box.radius[i]

                    # search to both directions
                    if box.direction[i] == 'both':
                        if box.actualLocation[i] - r < box.lowBound[i]:
                            lb = box.lowBound[i]
                        else:
                            lb = box.actualLocation[i] - r
                        
                        if box.actualLocation[i] + r > box.upBound[i]:
                            ub = box.upBound[i]
                        else:
                            ub = box.actualLocation[i] + r
                    # search to only left direction
                    elif box.direction[i] == 'left':
                        if box.actualLocation[i] - r < box.lowBound[i]:
                            lb = box.lowBound[i]
                        else:
                            lb = box.actualLocation[i] - r
                        ub = box.actualLocation[i]
                    elif box.direction[i] == 'right':
                        if box.actualLocation[i] + r > box.upBound[i]:
                            ub = box.upBound[i]
                        else:
                            ub = box.actualLocation[i] + r
                        lb = box.actualLocation[i]

                    else:
                        raise Exception("Invalid direction token")

                    # Sampling along the single direction within [lb,ub]
                    if(sample_method=="halton"):
                        Xdata,_ = halton_sequence(lb,ub,box.numOfSample[i])
                    elif(sample_method=="hammersley"):
                        Xdata,_ = hammersley_sequence(lb,ub,box.numOfSample[i])
                    elif(sample_method=="vander"):
                        Xdata,_=van_der_corput(lb,ub,box.numOfSample[i],2)
                    elif(sample_method=="latin"):
                        Xdata,_ = latin_random_sequence(lb,ub,box.numOfSample[i],1,1)
                    elif(sample_method=="sobol"):
                        Xdata = sobol_sequence(lb,ub,1,box.numOfSample[i])

                    box.totalCall += box.numOfSample[i]

                    ydata = []
                    ydata_flag = []
                    for val in Xdata:
                        location = box.actualLocation[:]
                        location[i] = val
                        obj,flag = hy_Object(hyCase, hysolver, location)
                        ydata.append(obj)
                        ydata_flag.append(flag)

                    # reset the search direction
                    if len(ydata_flag) %2 == 0:
                        left_count = Counter(ydata_flag[:int(len(ydata_flag)/2)])[True]
                        right_count = Counter(ydata_flag[int(len(ydata_flag)/2):])[True]
                        if left_count > right_count:
                            box.direction[i] = 'left'
                        elif left_count < right_count:
                            box.direction[i] = 'right'
                        else:
                            pass
                    else:
                        left_count = Counter(flag[:int(len(flag)/2)])[True]
                        right_count = Counter(flag[int(len(flag)/2)+1:])[True]
                        if left_count > right_count:
                            box.direction[i] = 'left'
                        elif left_count < right_count:
                            box.direction[i] = 'right'
                        else:
                            pass
                    
                    # add to temp Xdata
                    for f in range(len(ydata_flag)):
                        if ydata_flag[f] == True and ydata[f] < 1e4:
                            tempXdata.append(Xdata[f])
                            tempydata.append(ydata[f])
                        else:
                            box.totalCall -= 1

                    # Exit condition
                    if len(tempXdata) >= box.numOfSample[i]:
                        break
                
                # build surrograte model
                labels,expr = callAlamopy(tempXdata,tempydata,lb,ub)
                tempPoint,tempOptimal = callBaron(box,labels,expr,lb,ub,i)
                print('Baron point:',tempPoint,'...')
                print('Baron value:',tempOptimal,'...')

                boxVal, flag= hy_Object(hyCase,hysolver,tempPoint)
                if flag == True:
                    print('Aspen return value:',boxVal,'...')
                else:
                    print('Aspen return value:',boxVal,'...')
                    print('Predefined condition not satisfied')
                    box.radius[i] *= 1.2
                    box.numOfSample[i] += 4
                    continue

                if boxVal == 0:
                    boxVal += 1e-5
                ratio = tempOptimal / boxVal

                if (ratio>0.5 and ratio<1.5):
                    box.actualLocation = tempPoint
                    if len(box.optimalValues)<1 or boxVal<box.optimalValues[-1]:
                        box.optimalValues.append(boxVal)
                        box.optimalPoints.append(tempPoint)
                        box.calls.append(box.totalCall)
                        print('================================================')
                        print('Current optimal values:',box.optimalValues,'...')
                    box.radius[i] *=2
                    break
                else:
                    box.radius[i] *= 0.8
                    box.numOfSample[i] += 4
                
                # Exit condition
                if box.totalCall > 5000:
                    return
                elif len(box.optimalValues)>1 and box.optimalValues[-2]-box.optimalValues[-1] < 1e-4:
                    return
            
'''
============================================================================
'''

def main():
    HYSYS(cycles,sample_method,sample_ini)
    return

if __name__ == "__main__":
    cycles = int(sys.argv[1])
    sample_method = sys.argv[2]
    sample_ini = int(sys.argv[3])

main()

