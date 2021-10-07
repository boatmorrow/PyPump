# package of utilties for interpreting well tests
from pylab import *
import numpy as np
from math import factorial
from scipy.special import expi
import lmfit as lm
from scipy.optimize import least_squares
import pdb


#Thiem
def thiem(Q,r1,r2,h1,h2):
    '''calculate transmissivity for a confined aquifer for steady state conditions. Requires Q in (m3/s), r1 distance to closer well in m, r2 distance to farther well in m, h1 head at closer well and h2 head a farther well in m.  Returns T in m2/s.'''
    T = Q/(2*np.pi*(h2-h1))*np.log(r2/r1)
    return T

# Well function (exponential integral)
def well_func_p(u,res=1e-7):
    '''compute well function to within a given residual'''
    residual = 100
    w_u = -.5772156649 - np.log(u) + u
    i = 1 
    sign = 1

    while residual > res:
        sign = -1*sign
        i += 1
        w_u_new = w_u + sign*(u**i)/(i*factorial(i))
        residual = abs(w_u_new - w_u)
        w_u = w_u_new
#        print i, w_u, residual

    return w_u 

def well_func(u):
    '''calculate the well function using built scipy.special.expi exponential integral function.  Needs u = r^2S/4Tt. returns W(u).'''
    w_u = -1*expi(-1*u)
    return w_u

# Theis
def theis(S,T,r,Q,t):
    '''calculates the drawdown time series for a confined aquifer using the theis equation, at distance r (m) from well, given S the storativity (Ss*b) (m3/m3), T transmissivity in m2/s Q dischage in m3/s, and t time desired in seconds. Returns the drawdown at distance r  and time t in meters.'''
    u = (r**2*S)/(4*T*t)
    w_u = well_func(u)
    s = Q/(4*np.pi*T)*w_u
    return s

def theis_residual(params,s_obs,r,Q,t,error_perc=0.10,show_interim_results=False):
    '''calculate the chi squared residuals given params (S,T) at the location r for pump rate Q and times t.'''
    S = params['Storativity'].value
    T = params['Transmissivity'].value
    s_mod = theis(S,T,r,Q,t)
    residual = (s_obs-s_mod)/(s_obs*error_perc)
    if show_interim_results:
        print('S  = ' + str(S) + ' , T = '+ str(T), 'res = ' + str(np.linalg.norm(residual)))
        figure()
        loglog(t,s_obs,'ro',label='data')
        plot(t,s_mod,label='estimated')
        show()
    return residual

def theis_residual_scipy(x,s_obs,r,Q,t,error_perc=0.10,show_interim_results=False):
    '''calculate the chi squared residuals given params (S,T) at the location r for pump rate Q and times t.'''
    S = x[0]
    T = x[1]
    s_mod = theis(S,T,r,Q,t)
    residual = (s_obs-s_mod)/((s_obs+1e-5)*error_perc)
    if show_interim_results:
        print('S  = ' + str(S) + ' , T = '+ str(T), 'res = ' + str(np.linalg.norm(residual)))
        figure()
        loglog(t,s_obs,'ro',label='data')
        plot(t,s_mod,label='estimated')
        show()
    return residual

def estimate_theis_params(S,S_high,S_low,T,T_high,T_low,s_obs,r,Q,t_obs,error_perc=0.10,show_interim_results=False):
    '''mimimize the chi sqaured residual to estime S and T for theis solution.'''
    x = lm.Parameters()
    x.add('Storativity',value=S,min=S_low,max=S_high,vary=True)
    x.add('Transmissivity',value=T,min=T_low,max=T_high,vary=True)
    out = lm.minimize(theis_residual,x,args=(s_obs,r,Q,t_obs),kws={'error_perc':error_perc,'show_interim_results':show_interim_results})
    print(lm.fit_report(out))
    return out

def estimate_theis_params_scipy(S,S_high,S_low,T,T_high,T_low,s_obs,r,Q,t_obs,error_perc=0.10,show_interim_results=False):
    '''mimimize the chi sqaured residual to estime S and T for theis solution.'''
    x = np.array([S,T])
    bounds = (np.array([S_low,T_low]),np.array([S_high,T_high]))
    out = least_squares(theis_residual_scipy,x,bounds=bounds,args=(s_obs,r,Q,t_obs),kwargs={'error_perc':error_perc,'show_interim_results':show_interim_results})
    #pdb.set_trace()
    var_noise = 1./(len(s_obs)-len(out.x))*np.sum(out.fun**2)
    P_covar = var_noise*np.linalg.inv(np.dot(np.transpose(out.jac),out.jac))
    P_95 = 1.96*sqrt(np.diag(P_covar))  #1.96 should be looked up but for now...
    print('Storativity = %3.2g' %out.x[0], ' +/- %3.2g ' %P_95[0])
    print('Transmissivity = %3.2g' %out.x[1], ' +/- %3.2g ' %P_95[1])
    return out

#Cooper-Jacob
def cooper_jacob(S,T,r,Q,t):
    '''calculates the drawdown time series for a confined aquifer using the cooper-jacob equation, at distance r (m) from well, given S the storativity (Ss*b) (m3/m3), T transmissivity in m2/s Q dischage in m3/s, and t time desired in seconds. Returns the drawdown at distance r  and time t in meters.'''
    s = (2.3*Q)/(4*np.pi*T)*np.log10((2.25*T*t)/(r**2*S))
    u = (r**2*S)/(4*T*t)
#    if u >= 0.01:
#        print 'Cooper-Jacob conditions not satisfied, use theis'''
    return s

def cooper_jacob_residual(params,s_obs,r,Q,t,t_lin,error_perc=0.10,show_interim_results=False):
    '''calculate the chi squared residuals for the cooper jacob method given params (S,T) at the location r for pump rate Q and times t greater than t_lin - the start of the linear drawdown curve.'''
    S = params['Storativity'].value
    T = params['Transmissivity'].value
    s_obs2 = s_obs[np.where(t>t_lin)]
    s_mod = cooper_jacob(S,T,r,Q,t)
    residual = (s_obs2-s_mod[np.where(t>t_lin)])/(s_obs2*error_perc)
    if show_interim_results:
        print('S  = ' + str(S) + ' , T = '+ str(T), 'res = ' + str(np.linalg.norm(residual)))
        figure()
        semilogx(t,s_obs,'ro',label='data')
        plot(t,s_mod,label='estimated')
        show()
    return residual

def cooper_jacob_residual_scipy(x,s_obs,r,Q,t,t_lin,error_perc=0.10,show_interim_results=False):
    '''calculate the chi squared residuals for the cooper jacob method given params (S,T) at the location r for pump rate Q and times t greater than t_lin - the start of the linear drawdown curve.'''
    S = x[0]
    T = x[1]
    s_obs2 = s_obs[np.where(t>t_lin)]
    s_mod = cooper_jacob(S,T,r,Q,t)
    residual = (s_obs2-s_mod[np.where(t>t_lin)])/(s_obs2*error_perc)
    if show_interim_results:
        print('S  = ' + str(S) + ' , T = '+ str(T), 'res = ' + str(np.linalg.norm(residual)))
        figure()
        semilogx(t,s_obs,'ro',label='data')
        plot(t,s_mod,label='estimated')
        show()
    return residual

def estimate_cooper_jacob_params(S,S_high,S_low,T,T_high,T_low,s_obs,r,Q,t_obs,t_lin,error_perc=0.10,show_interim_results=False):
    '''mimimize the chi sqaured residual to estime S and T for theis solution.'''
    x = lm.Parameters()
    x.add('Storativity',value=S,min=S_low,max=S_high,vary=True)
    x.add('Transmissivity',value=T,min=T_low,max=T_high,vary=True)
    out = lm.minimize(cooper_jacob_residual,x,args=(s_obs,r,Q,t_obs,t_lin),kws={'error_perc':error_perc,'show_interim_results':show_interim_results})
    print(lm.fit_report(out))
    return out

def estimate_cooper_jacob_params_scipy(S,S_high,S_low,T,T_high,T_low,s_obs,r,Q,t_obs,t_lin,error_perc=0.10,show_interim_results=False):
    '''mimimize the chi sqaured residual to estime S and T for theis solution.'''
    x = np.array([S,T])
    bounds = (np.array([S_low,T_low]),np.array([S_high,T_high]))
    out = least_squares(cooper_jacob_residual_scipy,x,bounds=bounds,args=(s_obs,r,Q,t_obs,t_lin),kwargs={'error_perc':error_perc,'show_interim_results':show_interim_results})
    var_noise = 1./(len(s_obs)-len(out.x))*np.sum(out.fun**2)
    P_covar = var_noise*np.linalg.inv(np.dot(np.transpose(out.jac),out.jac))
    P_95 = 1.96*sqrt(np.diag(P_covar))  #1.96 should be looked up but for now...
    print('Storativity = %3.2g' %out.x[0], ' +/- %3.2g ' %P_95[0])
    print('Transmissivity = %3.2g' %out.x[1], ' +/- %3.2g ' %P_95[1])
    return out

