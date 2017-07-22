#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 22:08:43 2017

@author: Baobao
"""

from openpyxl import load_workbook
#from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd

wb=load_workbook("example.xlsm")
ws=wb.get_sheet_by_name(name='Sheet1')

excel_file=pd.ExcelFile('example.xlsm')
#df=excel_file.parse('Sheet1')
df2=pd.read_excel(excel_file, sheetname='Sheet1',index_col=None, 
                  parse_cols = "I:O")
df1=pd.read_excel(excel_file, sheetname='Sheet1',index_col=None, 
                  parse_cols = "A:G")

