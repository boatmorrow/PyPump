from slug_tests import *
from pylab import *
import pdb
import pickle as p
import datetime as dt


#well configuration
ls=3.0
lp=1.0
lr=4.0
rs=.0254
rp=.0254
rr=.0254
HA = 5. 
lw = 5.

#ls=tested screen section
#lp=upper packer access pipe section
#lr=riser pipe section
#rs=radius of tested screen section
#rp=radius access pipe
#rr=radius riser pipe
#lw=length of well

#datafile
f = open('SlugTestData.pkl','rb')


#load data
df = p.load(f)
f.close()


df.set_index('Date_Time',inplace=True)

#enter time start (t1) and stop (t2) for a slug test get this from your data file

# time stamp goes (Year,Month,Day,Hour,Minute,Secon)

t1 = dt.datetime(2018,4,11,15,30,24,100) 
t2 = dt.datetime(2018,4,11,15,30,34)


df_slug = df[t1:t2]

#time is elapsed time in sectons
t = ((df_slug.index-df_slug.index[0]).seconds+df_slug.ms/1000.).values
t=t[3::]
s = df_slug.Level-df_slug.Level[-1]
s=s[3::]
H_obs = s/(df_slug.Level[0]-df_slug.Level[-1])
H_obs = s/(s[0])

#Fast Bouer Analysis Below

td = CalcFastBouerTd(t,ls,lp,lr,rs,rp,rr)   
#H_tB = FastBouer(.8,td*1.4)



meow = estimate_FastBouer_params(td,H_obs,error_perc=0.05,show_interim_results=False)


###Get our parameters  
#meow=estimate_Bouer_params(0.5,0,8.0,Td_scalar_init,Td_scalar_low,Td_scalar_high,Td,w_obs)


##Input into Bouer function    
H_tB=FastBouer(meow.params['Cd'].value,td*meow.params['Td_scalar'].value)


###Plot 
plot(td,H_tB,label='modeled')
plot(td,H_obs,'ro',mec='k',label='obs.')
legend()
show()

K = get_FastBouer_K_unconfined(t,td*meow.params['Td_scalar'].value,meow.params['Cd'].value,ls,HA,lw,rs,rp)
print('estimated conductivity is ' + '%3.1e' %K +' m/s')
print('estimated conductivity is ' + '%3.1e' %(K*3.28*60*60*24) +'ft/d')
