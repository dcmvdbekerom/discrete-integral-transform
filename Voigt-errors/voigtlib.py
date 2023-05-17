import numpy as np
from scipy.special import erfcx


def wGwL_to_r(wG,wL):
    return (wL/wG)*(2*np.log(2))**0.5
    
def wGwL_to_d(wG,wL):
    return (wL - wG)/(wL + wG)

def wGwL_to_wV(wG,wL):
    C1 = 1.0692
    C2 = 0.86639
    return 0.5*(C1*wL + (C2*wL**2 + 4*aG**2)**0.5)

def d_to_aGaL(d):
    C1 = 1.0692
    C2 = 0.86639
    Q  = C1*(1+d) + (C2*(1+d)**2 + 4*(1-d)**2)**0.4
    return 2*(1-d)/Q, 2*(1+d)/Q

def d_to_r(d):
    return (1+d)/(1-d)*(2*np.log(2))**0.5

def r_to_d(r):
    rA = r/(2*np.log(2))**0.5
    return (rA-1)/(rA+1)


class ErrorIntegrals:
    def __init__(self):
        self.names = ['L1L1','L1L2','L2L2','G1L1','G1L2','G2L1','G2L2','G1G1','G1G2','G2G2']
        self.c_arr = np.array([
                 [0,0,1/2,-1,1,0,0,0,0],
                 [0,0,1/2,-2,5/2,-1,1,0,0],
                 [0,0,1/2,-3,19/4,-9/2,5,-1,1],
                 [0,1,-3/2,1,-1,0,0,0,0],
                 [0,1,-9/4,7/2,-4,1,-1,0,0],
                 [0,0,3/4,-5/2,3,-1,1,0,0],
                 [0,0,9/8,-23/4,33/4,-6,13/2,-1,1],
                 [3/4,-5/2,3,-1,1,0,0,0,0],
                 [-3/8,13/4,-21/4,5,-11/2,1,-1,0,0],
                 [33/16,-95/8,39/2,-89/4,53/2,-19/2,10,-1,1]
                ])
        
    def calc(self,r):
        self.last_r = r
        self.F_arr = np.array([(1 if n%2 else np.pi**0.5*erfcx(r))*r**n for n in range(9)]) #integrals not normalized by 1/a
        self.S_arr = self.c_arr.dot(self.F_arr)
        for i in range(self.S_arr.shape[0]):
            setattr(self, self.names[i], self.S_arr[i])

    def calc_from_wGwL(self,wG,wL):
        self.calc(wGwL_to_r(wG,wL))

    def calc_from_d(self,d):
        self.calc(d_to_r(d))

class WeightOptimizer:
    def __init__(self):
        self.S = ErrorIntegrals()

    def calc(self,r):
        self.S.calc(r)
        self.last_r = r
        
        self.R_G = - self.S.G1G2 / self.S.G1G1
        self.R_L = - self.S.L1L2 / self.S.L1L1
        
        S_V  = self.S.G1G1*self.S.L1L1 - self.S.G1L1**2

        self.R_GG = (self.S.G1L1*self.S.G2L1 - self.S.G1G2*self.S.L1L1) / S_V
        self.R_GL = (self.S.G1L1*self.S.L1L2 - self.S.G1L2*self.S.L1L1) / S_V
        self.R_LL = (self.S.G1L1*self.S.G1L2 - self.S.G1G1*self.S.L1L2) / S_V # = -1
        self.R_LG = (self.S.G1G2*self.S.G1L1 - self.S.G1G1*self.S.G2L1) / S_V

    def calc_from_wGwL(self,wG,wL):
        self.calc(wGwL_to_r(wG,wL))

    def calc_from_log_wGwL(self,log_wG,log_wL):
        self.calc(wGwL_to_r(np.exp(log_wG),np.exp(log_wL)))

    def calc_from_d(self,d):
        self.calc(d_to_r(d))

    def weights(self,tauG,tauL,log_pG,log_pL):
        aG = tauG + (self.R_GG*tauG*(1-tauG)*log_pG**2 + self.R_GL*tauL*(1-tauL)*log_pL**2)/(2*log_pG)
        aL = tauL + (self.R_LL*tauL*(1-tauL)*log_pL**2 + self.R_LG*tauG*(1-tauG)*log_pG**2)/(2*log_pL)
        return aG,aL

    def RMS_error(aG,aL,tauG,tauL,log_pG,log_pL,r):
    
        cG1 = (aG - tauG)*log_pG
        cL1 = (aL - tauL)*log_pL
        cG2 = 0.5*(tauG**2 - 2*aG*tauG + aG)*log_pG**2
        cL2 = 0.5*(tauL**2 - 2*aL*tauL + aL)*log_pL**2

        self.S.calc(r)

        E_RMS2 = (    cL1*cL1 * self.S.L1L1 +
                  2 * cL1*cL2 * self.S.L1L2 +
                      cL2*cL2 * self.S.L2L2 +
                  2 * cG1*cL1 * self.S.G1L1 +
                  2 * cG1*cL2 * self.S.G1L2 +
                  2 * cG2*cL1 * self.S.G2L1 +
                  2 * cG2*cL2 * self.S.G2L2 +
                      cG1*cG1 * self.S.G1G1 +
                  2 * cG1*cG2 * self.S.G1G2 +
                      cG2*cG2 * self.S.G2G2 )

        return (np.sum(E_RMS2)/E_RMS2.size)**0.5

