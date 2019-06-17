import os
import sys
import re
import matplotlib.pyplot as plt
import random
import numpy as np
from SALib.sample import saltelli
from SALib.util import read_param_file

# imported packages for scikit-learn
from pandas import DataFrame
from sklearn import linear_model,tree,svm,neighbors,ensemble
from sklearn.ensemble import BaggingRegressor,ExtraTreesRegressor
from sklearn.model_selection import train_test_split
# imported packages for alamopy
import alamopy
import math
import sympy
import examples
from scipy.stats import t, ttest_1samp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, lambdify
# compile .c file
def compileSource(file):
    os.system('gcc source_princetonlibgloballib/'+file+'.c -lm -o '+file)

# helper function to generate input value with sobel sequence
def val_generate(lb,ub,numofvar):
    bound_lst = []
    name_lst = []
    index = 1
    for l, u in zip(lb,ub):
        bound_lst.append([l,u])
        name_lst.append('x'+str(index))
        index = index+1
    problem = {
        'num_vars': numofvar,
        'names': name_lst,
        'bounds': bound_lst
    }
    param_values = saltelli.sample(problem,1000)
    # np.savetxt("param_values.txt", param_values)
    return param_values,problem


# generate input.in file according to requested number of input values
def create_input(file,values):
    infile = open(file, 'w')
    for val in values:
        infile.write(str(val)+'\n')
    infile.close()

# read returned value from file
def read_output(file,lst,compilefile):
    readfile = open(file, 'r')
    # outfile = open(file,'w')
    for line in readfile.readlines():
        lst.append(float(line.strip()))
        # outfile.write(line)
    readfile.close()
    # outfile.close()

def read_input(file,lst,compilefile):
    temp = []
    readfile = open(file,'r')
    for line in readfile.readlines():
        temp.append(float(line.strip()))
    readfile.close()
    lst.append(temp)

# call the executable file repeatedly
# input: input file name, number of variables, number of loops
# output: returned values list
def repeat_call(infile,compilefile,outfile,lb,ub,loop,numofvar):
    input_values,problem = val_generate(lb,ub,numofvar)
    outlst = []
    inlst = []
    for i in range(len(input_values)):
        create_input(infile,input_values[i])
        os.system('.\\'+compilefile)
        read_output(outfile,outlst,compilefile)
        read_input(infile,inlst,compilefile)
    return outlst,inlst

# read data file to get values of number of vars, boundaries nad starting points
# input: data file name
# output: attributes
def read_datafile(file):
    numOfVar = 0
    lowBound = []
    upBound = []
    startPoint = []

    infile = open(file,'r')
    lines = infile.readlines()
    # The first line
    for num in lines[0].split():
        numOfVar = int(num.strip())
    # The second line
    for i in lines[1].split():
        lowBound.append(float(i.strip()))
    # The third line
    for j in lines[2].split():
        upBound.append(float(j.strip()))
    # The fourth line
    for k in lines[3].split():
        startPoint.append(float(k.strip()))
    infile.close()
    return numOfVar,lowBound,upBound,startPoint

# transpose 2d array into 1d arrays
def transpose(matrix):
        new_matrix = []
        for i in range(len(matrix[0])):
            matrix1 = []
            for j in range(len(matrix)):
                matrix1.append(matrix[j][i])
            new_matrix.append(matrix1)
        return new_matrix

def generate_dataframe(input_values,problem,ydata):
    new_dic = {}
    trans_values = transpose(input_values)
    for name,value in zip(problem['names'],trans_values):
        new_dic[name] = value
    new_dic['Y'] = ydata
    return new_dic

# use models defined in scikit-learn package to do prediction
# an image(test value vs predicted value) can be generated
def try_different_method(model,x_train,y_train,x_test,y_test,pic_name):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    # plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    # plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.plot(result, y_test,'ko',label=pic_name)
    plt.xlabel('tabulated data')
    plt.ylabel('experimental data')
    plt.title('score: %f'%score)
    plt.legend()
    plt.savefig(pic_name+'-parityplot.png')

#***************************************************************
# The following is the main function
# command line: python callexe.py filename problemdata/filename.problem.data input.in output.out
#***************************************************************

if __name__ == '__main__':
    compileFile = sys.argv[1]
    dataFile = sys.argv[2]
    inputFile = sys.argv[3]
    outputFile = sys.argv[4]

def main():
    # os.system("mkdir outfiles/"+compileFile)
    compileSource(compileFile)
    numOfVar, lb, ub, sp = read_datafile(dataFile)
    ydata,in_values = repeat_call(inputFile, compileFile, outputFile, lb, ub, 10, numOfVar)

    # print(in_values)
    # test = generate_dataframe(in_values,problem,ydata)
    # name_lst = problem['names'].append('Y')
    # df = DataFrame(test,columns=name_lst)
    # plt.scatter(df['x2'], df['Y'], color='red')
    # plt.xlabel('x2', fontsize=14)
    # plt.ylabel('Y_value', fontsize=14)
    # plt.grid(True)
    # plt.savefig('test.png')

    X_train,X_test,y_train,y_test=train_test_split(in_values,ydata,test_size=0.25)
    # print(y_train)

    # TODO:decision tree model
    # model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
    # try_different_method(model_DecisionTreeRegressor,X_train,y_train,X_test,y_test,'Decision-tree')

    # TODO:linear regression
    # model_LinearRegression = linear_model.LinearRegression()
    # try_different_method(model_LinearRegression,X_train,y_train,X_test,y_test,'linear-regression')

    # TODO:SVM
    # model_SVR = svm.SVR(gamma='scale')
    # try_different_method(model_SVR,X_train,y_train,X_test,y_test,'SVM')

    # TODO:random forest (20 trees are utilized)
    # model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)
    # try_different_method(model_RandomForestRegressor,X_train,y_train,X_test,y_test,'random-forest')

    # TODO:Adaboost regression (50 trees are used)
    # model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)
    # try_different_method(model_AdaBoostRegressor,X_train,y_train,X_test,y_test,'Adaboost')

    # TODO: GBRT regression (100 trees are used)
    # model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)
    # try_different_method(model_GradientBoostingRegressor,X_train,y_train,X_test,y_test,'GBRT')

    # TODO:Bagging regression
    # model_BaggingRegressor = BaggingRegressor()
    # try_different_method(model_BaggingRegressor,X_train,y_train,X_test,y_test,'Bagging')

    # TODO:Extra tree regression
    # model_ExtraTreeRegressor = ExtraTreesRegressor(n_estimators=100)
    # try_different_method(model_ExtraTreeRegressor,X_train,y_train,X_test,y_test,'Extra-tree')

    # TODO:Use alamopy to do the regression
    sim = examples.sixcamel
    almsim = alamopy.wrapwriter(sim)
    res = alamopy.doalamo(X_train,y_train,almname='cam6',monomialpower=(1,2,3,4,5),multi2power=(1,2,3,4),simulator=almsim,expandoutput=True,maxiter=50)

    print("Model expression: ",res['model'],'\n')
    print("Rhe sum of squared residuals: ",res['ssr'],'\n')
    print("R squared: ",res['R2'],'\n')
    print("Root Mean Square Error: ",res['rmse'],'\n')

main()


