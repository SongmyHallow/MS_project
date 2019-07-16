import os
import re

pass_lst = ['arglina','arglinb','arglinc','box3','camel6','denschnd','denschne','esfl','ex4_1_7','explin','explin2']

ModelFile = open('DataFileName.txt','r')

for line in ModelFile.readlines():
    if(line.strip() =='arith.h' or line.strip() == 'config.h' or line.strip() == 'expquad'):
        continue
    else:
        os.system("python CoreAlgo.py "+line.strip()+" 60")
    
ModelFile.close()