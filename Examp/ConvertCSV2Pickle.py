# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:35:53 2018

@author: wpgardner
"""

from pylab import *
import pandas as pd
import datetime as dt
import pickle as p
import pdb

df = pd.read_csv('MDOT1.csv',skiprows=11,header=0,parse_dates=[['Date','Time']])


f = open('MDOT1.pkl','wb')

p.dump(df,f)

f.close()
