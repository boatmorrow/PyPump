
# First load the libraries we need
import pickle as p
import numpy as N
from pylab import *
from slug_tests import *
import pdb
import datetime as dt
import pandas as pd


###########################################################
# input parameters  - everything needs to be in si - m/s
###########################################################

############################################################
#transducer data - assumes that the data is in water level.
f = open('MDOT1.pkl','rb')

############################################################
#Start and end points of the data to be matched.
test1_start=dt.datetime(2021,4,3,13,14,15)
test1_end=dt.datetime(2021,4,3,16,19,15)

############################################################
#some well parameters for horslev
L = 10./3.28  #length of screen (m)
R = .0254 # 2 in. pvc
D = 30./3.28
S = 0.

# type of well from table 12.1 in your text.  first type is F=1 second is F=2, etc.
F=3

#inverse parameters
K_i = 1.e-7
K_high = 5.e-4
K_low = 1.e-9
#datafile

###########################################################
# RUNTIME parameters
###########################################################
#show all inversion plots?
show_interim_results=False
show_interim_results=True


############################################################################
# RUN - don't change things below here.
#############################################################################
#import data
df = p.load(f)
f.close()

#add milliseconds
tdelta = pd.to_timedelta(df['ms'],unit="ms")
df['DateTime2']=df.Date_Time+tdelta
df1=df[(df['DateTime2']>test1_start) & (df['DateTime2']<test1_end)]

#horslev drawdown ratio
df1['drawdown']=(df1.LEVEL-df1.LEVEL.iloc[-1])/(df1.LEVEL.iloc[0]-df1.LEVEL.iloc[-1])
df1['elapsed']=(df1['DateTime2']-np.min(df1['DateTime2']))
df1['elapsed_seconds'] = df1.elapsed.dt.total_seconds()
df1['elapsed_seconds'] = df1['elapsed_seconds']-np.min(df1['elapsed_seconds'])
df1 = df1[df1.drawdown>1e-2]

t = df1.elapsed_seconds.values
d = df1['drawdown'].values
H_t = d+1e-30 #make it the drawdown ratio
out = estimate_horslev_params_scipy(K_i,K_high,K_low,H_t,t,D=D,R=R,F=F,L=L,show_interim_results=show_interim_results)
H_f = horslev(out.x[0],t,D=D,S=S,L=L,R=R,F=F)
d_mod_f = H_f*d[0]

#visualize the solution
plot(t,H_t,'ro',label='observed')
semilogy(t,H_f,label='modeled')
xlabel('time (s)')
ylabel('drawdown (m)')
legend(loc='best',numpoints=1)
show()
