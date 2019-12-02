import json
import csv
import sys
import xlrd
import collections
import numpy as np
import math
import re

def solution(name):
    referenceFile = xlrd.open_workbook('ModelList.xlsx')
    sheet = referenceFile.sheet_by_name('models')
    # index, name, smoothness, convexity, variables, type, solution, library,
    for i in range(sheet.nrows):
        row = sheet.row_values(i)
        if name == row[1]:
            return (float(row[6]), row[2],row[3])
    return None

def mysolution(name):
    reportDataFile = open('reportData.csv','r')
    reportDataReader = csv.reader(reportDataFile)
    solDict = collections.defaultdict(list)
    for line in reportDataReader:
        if line[0] == name:
            solDict[name].append((json.loads(line[2]),json.loads(line[3])))
    reportDataFile.close()
    return solDict

if __name__ == "__main__":
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)
            
    '''
    Main Function
    '''
    modelListFile = open('selected.txt','r')
    numOfModels = 100
    resultDict = {} 
    for model in modelListFile:
        name = model.strip()
        # Precise solution
        referenceSolution, modelSmoo, modelConv= solution(name)
        # Returned results
        if re.search('convex',name):
            name = 'f'+ name
        if re.search('problem',name):
            name = name.replace('.','_')

        solDict = mysolution(name)
        
        minVal = math.inf
        for pair in solDict[name]:
            values, calls = pair
            if len(values)>0 and values[-1] < minVal:
                minVal = values[-1]
                resultDict[name] = (values[-1],calls[-1],referenceSolution,modelSmoo,modelConv)
            
    numOfCallList = []
    numOfSolvedList = []
    for numOfCall in range(1000):
        solvedCounter = 0
        for key in resultDict.keys():
            if resultDict[key][1] < numOfCall:
                solvedCounter+=1
        numOfCallList.append(numOfCall)
        numOfSolvedList.append(solvedCounter/numOfModels)
        
    modelListFile.close()
    print(numOfCallList,numOfSolvedList)
    print(len(numOfCallList),len(numOfSolvedList))