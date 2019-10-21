'''
This script is used to gather information from nik's webiste:
'''
import requests
from bs4 import BeautifulSoup
import xlsxwriter as wx
import sys
import xlrd, xlwt
from datetime import date,datetime
import pandas as pd

def web_info():
    response = requests.get('http://thales.cheme.cmu.edu/dfo/comparison/comp.html')
    soup = BeautifulSoup(response.content, 'lxml')

    title = []
    sol = []
    for index, x in enumerate(soup.find_all('tr')):
        text = x.get_text().split()
        if index == 0:
            title = text[:]
            title.insert(0,"index")
        else:
            sol.append(text)
    return title,sol

def set_style(name,height,bold=False):
	style = xlwt.XFStyle()
	font = xlwt.Font()
	font.name = name
	font.bold = bold
	font.color_index = 4
	font.height = height
	style.font = font
	return style

def write_excel(title,sol):
	f = xlwt.Workbook()
	sheet1 = f.add_sheet('models',cell_overwrite_ok=True)
	row0 = title
	#写第一行
	for i in range(0,len(row0)):
		sheet1.write(0,i,row0[i],set_style('Times New Roman',220,True))
	for index,row in enumerate(sol):
		for i in range(0,len(row)):
			sheet1.write(index+1,i,row[i])
	f.save('ModelList.xls')

def read_excel(smoothness,convexity,variables):
	file = 'ModelList.xlsx'
	sheet = pd.read_excel(file,sheet_name='models',header=0,index_col=1,usecols="A:H")
	filter_sol = sheet[(sheet['smoothness']==smoothness)&(sheet['convexity']==convexity)&(sheet['variables']<variables)]
	return filter_sol

if __name__ == "__main__":
    # title, sol = web_info()
    # write_excel(title,sol)
	sol = read_excel('smooth','convex',10)