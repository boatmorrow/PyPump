#Example Well Test Interpretation using PyPump
#Written W. Payton Gardner - University of Montana

import pandas as pd
from pylab import *
import datetime as dt
from pump_test import *

###########################################################################
#INPUTS
###########################################################################
#Start and end points of the data to be matched.
############################################################
test1_start=dt.datetime(2018,3,27,8,23,0)
test1_end=dt.datetime(2018,3,30,7,13,00)
#test1_end=dt.datetime(2018,3,27,8,30,00)

###########################################################
# pump test parameters  - everything needs to be in si - m/s
###########################################################
#some well parameters
Q = 54.  #gpm
Q = Q*0.00378541/60. #m3/s

#initial guesses for S and T
S_i = 1e-7 #initial guess at storativity
T_i = 1e-5 #intial guess at transmissivity

#bounds for the inversion
S_low = 1e-9 #minimum S
S_high = 1e-0 #maximum S
T_low = 1e-8 #max trans
T_high = 1e-2 #min trans


###########################################################
# RUNTIME parameters
###########################################################
#show all inversion plots?
show_interim_results=True

#match raw or modified drawdown
use_apparent_confined_drawdown = False

#############################################################################
# RUN - don't change things below here.
#############################################################################
#import data
df = pd.read_excel('LQ23 72_hour_Aq Test.xlsx','LQ23_290662_data',index_col=0)
df['drawdown'] = -1*(df.Baro_corrected_WL_ft[df.Baro_corrected_WL_ft.idxmin()]-df.Baro_corrected_WL_ft)
df.drawdown = df.drawdown/3.28
r = df['distance to pumping well_ft'][0]/3.28 #meters
b = 10./3.28 #estimate for now
df['confined_drawdown'] = df['drawdown']-(df['drawdown']**2)/(2.*b)
df['elapsed']=(df.index-df.index[0])
df['elapsed_seconds'] = df.elapsed.dt.total_seconds()
df1=df[(df.index>test1_start) & (df.index<test1_end)]
df1['elapsed_seconds'] = df1['elapsed_seconds']-(df1['elapsed_seconds'][0])
df1=df1.iloc[1::]

t = df1.elapsed_seconds.values
if use_apparent_confined_drawdown:
    s = df1.confined_drawdown.values
else:
    s = df1.drawdown.values

plot(df1.elapsed_seconds,df1.drawdown,'ro',mec='k')

out = estimate_theis_params_scipy(S_i,S_high,S_low,T_i,T_high,T_low,s,r,Q,t,error_perc=0.10,show_interim_results=show_interim_results)

s_mod = theis(out.x[0],out.x[1],r,Q,t)

#plot the data

plot(df1.elapsed_seconds,df1.confined_drawdown,'go',mec='k')
loglog(df1.elapsed_seconds,df1.drawdown,'ro',mec='k')
plot(t,s_mod,'b-')
xlabel('time (s)')
ylabel('Drawdown (m)')


show()
