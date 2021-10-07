#series of utilites for intepreting slug tests
# W. Payton Gardner - University of Montana
import numpy as N
from pylab import *
import pdb
from scipy.special import jn, yn
from scipy.integrate import quad, quadrature, romberg
from scipy.interpolate import interp1d
import lmfit as lm
from scipy.optimize import least_squares

def horslev(K,t,D=1,S=1,L=1,R=1,F=2):
    '''return the drawdown ratio for times t for a Horslev slug analysis.  D = depth of well, S = saturated thickness or confined aquifer thickness, L = well screen length and R is the well radius are measurements of the well configuration F.  Default is fully cased open hole'''
    
    #first cross sectional area
    A = N.pi*R**2.
    
    #now get proper F
    if F==1:
        FF = 16*N.pi*D*S*R
    
    elif F==2:
        FF = 11.*R/2.
    
    elif F==3:
        FF = (2*N.pi*L)/(N.log(L/R))
#        A = 2*N.pi*R*L
    
    elif F==6:
        if L/S <= 0.2:
            Cs = (2*N.pi*(L/R))/(N.log((L/R)+1.36))
            FF = Cs*R
        elif L==S:
            FF = (2*N.pi*L)/N.log(200.) #R/R_o assumed
        else:
            FF = (2*N.pi*L)/N.log(L/R)
    
    else:
        print('need to add this F function')
        return

    if F == 1:
        print("F=1 broken, don't use it")
        return
        #H_t = 1.-t*K*(16*D*S/R)
    else:
        H_t = N.exp(-(K*FF/A)*t)

    return H_t

def horslev_residual(params,s_obs,t,D=1,S=1,L=1,R=1,F=2,error_perc=0.10,show_interim_results=False):
    '''calculate the chi squared residuals for a horslev slug test for params = K, at all times t. s_mod and t must be aligned.'''
    K = params['K'].value
    s_mod = horslev(K,t,D=D,S=S,L=L,R=R,F=F)
    residual = (s_obs-s_mod)/(s_obs*error_perc)
    if show_interim_results:
        print 'K = '+ str(K), 'res = ' + str(N.linalg.norm(residual))
        figure()
        loglog(t,s_obs,'ro',label='data')
        plot(t,s_mod,label='estimated')
        show()
    return residual

def horslev_residual_scipy(params,s_obs,t,D=1,S=1,L=1,R=1,F=2,error_perc=0.10,show_interim_results=False):
    '''calculate the chi squared residuals for a horslev slug test for params = K, at all times t. s_mod and t must be aligned.'''
    K = params[0]
    s_mod = horslev(K,t,D=D,S=S,L=L,R=R,F=F)
    residual = (s_obs-s_mod)/((s_obs+1.e-10)*error_perc)
    residual = (s_obs-s_mod)
    if show_interim_results:
        print('K = '+ str(K), 'res = ' + str(N.linalg.norm(residual)))
        figure()
        semilogy(t,s_obs,'ro',label='data')
        plot(t,s_mod,label='estimated')
        show()
    return residual


def estimate_horslev_params(K,K_high,K_low,s_obs,t_obs,D=1,S=1,L=1,R=1,F=2,error_perc=0.10,show_interim_results=False):
    '''mimimize the chi sqaured residual to estime S and T for theis solution.'''
    x = lm.Parameters()
    x.add('K',value=K,min=K_low,max=K_high,vary=True)
    out = lm.minimize(horslev_residual,x,args=(s_obs,t_obs),kws={'D':D,'S':S,'L':L,'R':R,'F':F,'error_perc':error_perc,'show_interim_results':show_interim_results})
    print(lm.fit_report(out))
    return out

def cbp_int_func(u,alpha,beta):
    '''the function to integrate for cbp slug test'''
    F = (exp(-beta*u**2/alpha))/(u*((u*jn(0,u)-2*alpha*jn(1,u))**2+(u*yn(0,u)-2*alpha*yn(1,u))**2))
    return F

def estimate_horslev_params_scipy(K,K_high,K_low,s_obs,t_obs,D=1,S=1,L=1,R=1,F=2,error_perc=0.10,show_interim_results=False):
    '''mimimize the chi sqaured residual to estime S and T for theis solution.'''
    x = N.array([K])
    bounds = ([K_low,K_high])
    out = least_squares(horslev_residual_scipy,x,bounds=bounds,args=(s_obs,t_obs),kwargs={'D':D,'S':S,'L':L,'R':R,'F':F,'error_perc':error_perc,'show_interim_results':show_interim_results})
    var_noise = 1./(len(s_obs)-len(out.x))*np.sum(out.fun**2)
    P_covar = var_noise*np.linalg.inv(np.dot(np.transpose(out.jac),out.jac))
    P_95 = 1.96*sqrt(np.diag(P_covar))  #1.96 should be looked up but for now...
    print('Conductivity = %3.2g' %out.x[0], ' +/- %3.2g ' %P_95[0])
    #print 'Transmissivity = %3.2g' %out.x[1], ' +/- %3.2g ' %P_95[1]
    return out

def cbp_int_func(u,alpha,beta):
    '''the function to integrate for cbp slug test'''
    F = (exp(-beta*u**2/alpha))/(u*((u*jn(0,u)-2*alpha*jn(1,u))**2+(u*yn(0,u)-2*alpha*yn(1,u))**2))
    return F

def cbp(K,Ss,t,b,r_c,r_s):
    '''return the drawdown ratio for all times t, for a confined aquifer with fully penetrating well with conductivity K, storavtivity Ss, thickness b, r_c well casing radius, and r_s the radius of screen (effective radius).'''
    #alpha
    T = K*b
    alpha = (Ss*r_s**2)/r_c**2
    if alpha < 1.e-5:
        print(" the numerical integrator can't handle this small of an alpha, need to try another technique")
        return

    #beta
    beta = (T*t)/r_c**2
    H_t = N.zeros(len(beta))

    for i in xrange(len(beta)):
        H_t_i = (8*alpha/pi**2)*quad(cbp_int_func,0,N.inf,args=(alpha,beta[i]),limit=100)[0]
        H_t[i] = H_t_i
    
    return H_t

def br_linterp(LeRw_i):
    '''conctains the info to calculate the A, B, C coefficients for ln(Le/r_w) using linear interpolation. Emperical data taken from USGS bouwer-rice slug test excel sheet, returns the tuple (A,B,C).'''
    LeRw = N.array([0.5, 0.689133333,0.891133333,0.9893,1.284933333,1.4578,1.6855,1.827366667,1.987033333,2.2708,2.458133333,2.675366667,2.9806,3.277233333])
    A = N.array([1.738,1.738,1.802,1.87,2.175,2.464,3.057,3.604,4.397,6.022,7.069,8.062,9.156,9.767])
    B = N.array([0.229,0.229,0.269,0.265,0.339,0.407,0.49,0.585,0.738,1.103,1.51,2.1275,2.8485,3.3175])
    C = N.array([0.835,0.835,1.09,1.192,1.696,2.023,2.698,3.283,4.183,6.732,8.675,10.58,12.32,13.126])
    fA = interp1d(LeRw,A)
    fB = interp1d(LeRw,B)
    fC = interp1d(LeRw,C)
    A_i = fA(LeRw_i)
    B_i = fB(LeRw_i)
    C_i = fC(LeRw_i)
    return (A_i,B_i,C_i)


def bouwer_rice(K,t,L_e,L_w,H,r_c,r_w):
    '''caculate the drawdown ratio for the bouwer rice unconfined aquifer solution.  uses coefficient A,B,C interpolation values taken from the USGS excel spread sheet. where K is the conductivity, L_e is the screened length, L_w is the total depth beneath water table of well, H is the aquifer thickness, r_c is the radius of the casing and r_w is the screened radius/effective radius'''

    #first get A,B,C
    LeRw = N.log10(L_e/r_w)
    (A,B,C) = br_linterp(LeRw)

    if L_w == H:
        LRr = (1.1/N.log(L_w/r_w)+C/(L_e/r_w))**-1
    else:
        LRr = (1.1/N.log(L_w/r_w) + (A+B*N.log((H-L_w)/r_w))/(L_e/r_w))**-1

    P = (2*K*L_e)/(r_c**2*LRr)

    H_t = N.exp(-P*t)

    return H_t

def get_Le(ls,lp,lr,rs,rp,rr):
    le=ls/2.*(rr**2)/(rs**2)+lp*(rr**2)/(rp**2)+lr
    return le

def CalcFastBouerTd(t,ls,lp,lr,rs,rp,rr):
    '''Calculate Le and initial dimensional time given well configuration:
            ls=tested screen section
            lp=upper packer access pipe section
            lr=riser pipe section
            rs=radius of tested screen section
            rp=radius access pipe
            rr=radius riser pipe
        Returns: dimensionless time for all times in t'''
    g = 9.81
    Le=ls/2.*(rr**2)/(rs**2)+lp*(rr**2)/(rp**2)+lr 
    Td=(g/Le)**(0.5)*t
    return Td
   
def FastBouer(Cd,t_d):
    '''returns dimensionless drawdown H_t for all dimensionless times td, and the given Cd.  From KGS online paper.'''    
    #pdb.set_trace()
    g=9.81
    omega_d=np.abs(1-(Cd/2.)**2)**0.5
    omega_d_plus=(-Cd/2.)+omega_d
    omega_d_minus=(-Cd/2.)-omega_d
    
    if Cd < 2:
        wd=np.exp(-Cd/2*t_d)*(np.cos(omega_d*t_d)+(Cd/(2.*omega_d)*np.sin(omega_d*t_d)))
        
    elif Cd > 2:
        wd=-(1/(omega_d_plus-omega_d_minus)*(omega_d_minus*np.exp(omega_d_plus*t_d)-omega_d_plus*np.exp(omega_d_minus*t_d)))        
        
    elif Cd == 2:
        wd=np.exp(-t_d)*(1+t_d)
    return wd
        
def FastBouer_residual(params,Td,H_obs,error_perc=0.05,show_interim_results=False):
    Cd=params['Cd'].value
    Td_scalar=params['Td_scalar'].value
    Td = Td*Td_scalar
    H_mod=FastBouer(Cd,Td)
    residual=(H_obs-H_mod)/(error_perc*np.mean(H_obs))
    if show_interim_results:
        print('C_d = '+ str(Cd), 'res = ' + str(N.linalg.norm(residual)))
        print('td_scal = ' +str(Td_scalar))
        figure()
        plot(Td,H_obs,'ro',mec='k',label='data')
        plot(Td,H_mod,label='estimated')
        show()
    return residual
    
def estimate_FastBouer_params(Td,H_obs,error_perc=0.05,show_interim_results=False):
    Td_scalar_init=1
    Td_scalar_low=0.001
    Td_scalar_high=15
    Cd = 0.5
    Cd_low = 0.
    Cd_high= 30.
    x=lm.Parameters()
    x.add('Cd',value=Cd,min=Cd_low,max=Cd_high,vary=True)
    x.add('Td_scalar',value=Td_scalar_init,min=Td_scalar_low,max=Td_scalar_high,vary=True) 
    out=lm.minimize(FastBouer_residual,x,args=(Td,H_obs),kws={'error_perc':error_perc,'show_interim_results':show_interim_results})  
    return out

def get_FastBouer_K_unconfined(t,td,Cd,ls,HA,lw,rw,rc):
    '''Caculate hydraulic conductiviy from a Fast Bouer curve match. http://www.kgs.ku.edu/Hydro/Publications/OFR00_40/
       Returns:
            K - hydraulic cond given scaled dimen Cd
       Inputs:
           t - sample time (vector len t)
           td - best fit scaled dimesionless time (vector len t)
           Cd - best fit FastBouer paramter
           ls - screen length
           HA - aquifer thickness
           lw - depth of well
           rw - effective radius of well- well + gravel pack
           rc - radius of casing'''
    #first get A,B,C
    LeRw = N.log10(ls/rw)
    (A,B,C) = br_linterp(LeRw)

    if lw == HA:
        LRr = (1.1/N.log(lw/rw)+C/(ls/rw))**-1
    else:
        LRr = (1.1/N.log(lw/rw) + (A+B*N.log((HA-lw)/rw))/(ls/rw))**-1
    
    t_ratio_index = int(np.floor(len(t)/2))
    t_ratio = td[t_ratio_index]/t[t_ratio_index]
    Kr=t_ratio*(rc**2*LRr)/(2*ls*Cd)
    return Kr
           
