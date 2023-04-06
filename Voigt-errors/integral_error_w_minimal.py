import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.optimize import minimize

def gG(x):
    return 2*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*x**2)

def gL(x):
    return 2/((np.pi) * (4*x**2 + 1))

dx = 0.001
da = 0.0005
x_max = 10.0
x = np.arange(-x_max/2,x_max/2,dx)
p = 1.2

def errfun(a,g,x,p,t):
    I    = g(x)
    
    Ia_0 = p** t   *g(x*p** t   )
    Ia_1 = p**(t-1)*g(x*p**(t-1))

    Ia   = (1-a)*Ia_0 + a*Ia_1
    e_2  = np.sqrt(np.sum((I - Ia)**2)*dx)
    return e_2

def taylor_error(g,x,p,t):

    dgdx   = np.ediff1d(g(x),to_end = 0)/dx
    d2gdx2 = np.ediff1d(dgdx,to_end = 0)/dx
    d2gdx2[-2] = 0.0
    err1 = 0.50*t*(1-t)*np.log(p)**2 * np.sqrt(np.sum((g(x)     + 3  *x*dgdx + x**2*d2gdx2)**2)*dx)
    err2 = 0.50*t*(1-t)*np.log(p)**2 * np.sqrt(np.sum((           2  *x*dgdx + x**2*d2gdx2)**2)*dx)
    err3 = 0.50*t*(1-t)*np.log(p)**2 * np.sqrt(np.sum((0.5*g(x) + 2.5*x*dgdx + x**2*d2gdx2)**2)*dx)
    return err1,err2,err3

t_tay  = np.arange(0,1.01,0.01)
a1_tay = t_tay
a2_tay = (1-p**-t_tay)/(1-p**-1)
a3_tay = 0.5*(a1_tay + a2_tay)

t_arr  = np.linspace(0.0,1.0,9)

g = gG
ls = '-'

err1,err2,err3 = taylor_error(g,x,p,t_tay)
plt.plot(a1_tay,err1,'--',c='salmon')
plt.plot(a2_tay,err2,'-.',c='cornflowerblue')
plt.plot(a3_tay,err3,'-' ,c='lightgray')

a1_list  = []
a1e_list = []

a2_list  = []
a2e_list = []

a3_list  = []
a3e_list = []

a_dict = {}
ae_dict = {}

weight_kinds = ['linear','ZEP','min-RMS']
for kind in weight_kinds:
    a_dict[kind] = []
    ae_dict[kind] = []

for ti in t_arr:

    a1 = ti
    a1_list.append(a1)
    a1e_list.append(errfun(a1,g,x,p,ti))

    a2 = (1-p**-ti)/(1-p**-1)
    a2_list.append(a2)
    a2e_list.append(errfun(a2,g,x,p,ti))

    a3 = 0.5*(a1+a2)
    a3_list.append(a3)
    a3e_list.append(errfun(a3,g,x,p,ti))

    if g == gL and ti > 0.0 and ti < 1.0:
        plt.text(a3_list[-1],a3e_list[-1],'\n$\\tau={:.2f}$'.format(ti),ha='center',va='top')


    a_arr = np.arange(ti-0.025,ti+0.05,da)
    e_arr = np.array([errfun(ai,g,x,p,ti) for ai in a_arr])
    plt.plot(a_arr,e_arr,ls)
    
plt.plot([],[])
plt.plot(a1_list,a1e_list,'+k',fillstyle='none',label='$a_1=t$')
plt.plot(a2_list,a2e_list,'xk',fillstyle='none',label='$a_2=(1-p^{-t})/(1-p^{-1})$')
plt.plot(a3_list,a3e_list,'.k',fillstyle='none',label='$a_3=(a_1+a_2)/2$')

plt.xlabel('$a_w$')
plt.ylabel('RMS-Error')
plt.xlim(0,1)
plt.ylim(0,0.0055)
#plt.legend()
plt.show()
