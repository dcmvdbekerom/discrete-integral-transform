import numpy as np
import matplotlib.pyplot as plt
import fss_py


v_max =  10.0 #cm-1
dv =     0.001 #cm-1
v = np.arange(0,v_max,dv) #cm-1

v0i = [0.0]
log_wGi = [0.23]
log_wLi = [0.57]
S0i = [1.0]

log_pwG = 0.01
log_pwL = 0.1

log_wG = np.arange(0.2,0.4,log_pwG)
log_wL = np.arange(0.5,0.7,log_pwL)

S_klm = fss_py.calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, 'linear','linear')
I0 = fss_py.apply_transform(v, log_wG, log_wL, S_klm)

log_pwG = 0.1
log_wG = np.arange(0.2,0.4,log_pwG)

S_klm = fss_py.calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, 'linear','linear')
I1 = fss_py.apply_transform(v, log_wG, log_wL, S_klm)

S_klm = fss_py.calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, 'opti','opti')
I2 = fss_py.apply_transform(v, log_wG, log_wL, S_klm)

##S_klm = fss_py.calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, 'min-RMS','min-RMS')
##I3 = fss_py.apply_transform(v, log_wG, log_wL, S_klm)

plt.axhline(0,c='k')
plt.plot(v,(I1-I0)**2,label='linear')
plt.plot(v,(I2-I0)**2,'--',label='opti')
##plt.plot(v,I3-I0,label='min-RMS')

print('linear: ',np.sum((I1-I0)**2))
print('opti:   ',np.sum((I2-I0)**2))
##print('min-RMS: ',np.sum((I3-I0)**2))

plt.legend()
plt.show()
