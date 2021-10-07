from pylab import *
import pandas as pd
import datetime as dt
import pickle as p
import pdb

f = open('MDOT3.pkl','rb')

df = p.load(f)

f.close()

#pdb.set_trace()
plot(df.Date_Time,df.LEVEL,'ro-',mec='k')

show()


