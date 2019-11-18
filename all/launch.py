import os
import re

ModelFile = open('selected.txt','r')
methods = ['halton','vander','hammersley','latin','sobol']

for line in ModelFile.readlines():
    name = line.strip()
    for method in methods:
        os.system("python blackbox.py "+name+" 60 "+method+" 8")
    
ModelFile.close()