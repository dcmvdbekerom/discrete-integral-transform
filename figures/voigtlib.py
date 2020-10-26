import numpy as np

pi = np.pi
ln2 = np.log(2)

C1_GG = ((6 * pi - 16) / (15 * pi - 32)) ** (1 / 1.50)
C2_GG = (2 * ln2 / 15) ** (1 / 1.50)
C1_LG = ((6 * pi - 16) / 3 * (ln2 / (2*pi)) ** 0.5) ** (1 / 2.25)
C2_LG = ((2 * ln2) ** 2 / 15) ** (1 / 2.25)

C1 = 1.06920
C2 = 0.86639

f_alphaG  = lambda alpha: 2/(C1*alpha + (C2*alpha**2 + 4)**0.5)
f_alphaL  = lambda alpha: alpha*f_alphaG(alpha)

GG_FT = lambda x: np.exp(-(pi*x)**2/(4*ln2))
GL_FT = lambda x: np.exp(- pi*np.abs(x))

GG = lambda x: 2*(ln2/pi)**0.5*np.exp(-4*ln2*x**2)
GL = lambda x: 2/(pi*(1+4*x**2))

def gL(v,v0,w):
    return (1/np.pi) * (w/2) / ((v-v0)**2 + (w/2)**2)

def gG(v,v0,w):
    return (2/w)*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*((v-v0)/w)**2)
 
def gV(v,v0,wL,wG):
    gamma = (wG**5 + 2.69269*wG**4*wL + 2.42843*wG**3*wL**2 + 4.47163*wG**2*wL**3 + 0.07842*wG*wL**4 + wL**5)**0.2
    eta   = 1.36603*wL/gamma - 0.47719*(wL/gamma)**2 + 0.11116*(wG/gamma)**3
    return (1-eta) * gG(v,v0,gamma) + eta * gL(v,v0,gamma)

def optimized_weights(tv,tG,tL,dxvG,dxG,dxL,alpha):

    R_Gv = 8*ln2 
    R_GG = 2 - 1/(C1_GG + C2_GG*alpha**(2/1.5))**1.5
    R_GL = -2*ln2*alpha**2

    R_LG = 1/(C1_LG*alpha**(1/2.25) + C2_LG*alpha**(4/2.25))**2.25
    R_LL = 1

    av = tv
    
    aG = tG + (R_Gv*tv*(tv-1)*dxvG**2 +
               R_GG*tG*(tG-1)*dxG**2 +
               R_GL*tL*(tL-1)*dxL**2) / (2*dxG)
                             
    aL = tL + (R_LG*tG*(tG-1)*dxG**2 +
               R_LL*tL*(tL-1)*dxL**2) / (2*dxL)

    return av,aG,aL

def lineshape_basis_FT(x_FT,tv,tG,tL,dxv,dxG,dxL,alpha):

    alphaG = f_alphaG(alpha)
    alphaL = f_alphaL(alpha)

    Iav0_FT = np.exp(-2j*pi* tv   *dxv*x_FT)
    Iav1_FT = np.exp(-2j*pi*(tv-1)*dxv*x_FT)
    IaG0_FT = GG_FT(alphaG*x_FT*np.exp(  -tG *dxG))
    IaG1_FT = GG_FT(alphaG*x_FT*np.exp((1-tG)*dxG))
    IaL0_FT = GL_FT(alphaL*x_FT*np.exp(  -tL *dxL))
    IaL1_FT = GL_FT(alphaL*x_FT*np.exp((1-tL)*dxL)) 

    return Iav0_FT,Iav1_FT,IaG0_FT,IaG1_FT,IaL0_FT,IaL1_FT


def approximate_lineshape_FT(av,aG,aL,basis):

        Iav0_FT,Iav1_FT,IaG0_FT,IaG1_FT,IaL0_FT,IaL1_FT = basis

        Ia000 = (1-av)*(1-aG)*(1-aL) * Iav0_FT * IaG0_FT * IaL0_FT
        Ia001 = (1-av)*(1-aG)*   aL  * Iav0_FT * IaG0_FT * IaL1_FT
        Ia010 = (1-av)*   aG *(1-aL) * Iav0_FT * IaG1_FT * IaL0_FT
        Ia011 = (1-av)*   aG *   aL  * Iav0_FT * IaG1_FT * IaL1_FT
        Ia100 =    av *(1-aG)*(1-aL) * Iav1_FT * IaG0_FT * IaL0_FT 
        Ia101 =    av *(1-aG)*   aL  * Iav1_FT * IaG0_FT * IaL1_FT
        Ia110 =    av *   aG *(1-aL) * Iav1_FT * IaG1_FT * IaL0_FT
        Ia111 =    av *   aG *   aL  * Iav1_FT * IaG1_FT * IaL1_FT

        Ia = Ia000 + Ia001 + Ia010 + Ia011 + Ia100 + Ia101 + Ia110 + Ia111

        return Ia


def exact_lineshape_FT(x_FT,alpha):
    alphaG = f_alphaG(alpha)
    alphaL = f_alphaL(alpha)
    Iex_FT = GG_FT(alphaG*x_FT)*GL_FT(alphaL*x_FT)
    return Iex_FT

def integrate(I,dx):
    
    odd = len(I)%2
    
    #Simpsons's rule:
    integral = (I[0] + 4*np.sum(I[1:-1:2]) + 2*np.sum(I[2:-2:2]) + I[odd - 2])*dx/3
    if odd:
        integral += I[-1]*dx
        
    return integral


def error_integrals(alpha, ksi_max = 10, dksi = 0.01):

    x_FT = np.arange(0,ksi_max,dksi)

    alpha_arr = (np.array([alpha]) if np.isscalar(alpha) else alpha)
    alphaG_arr = f_alphaG(alpha_arr)
    alphaL_arr = f_alphaL(alpha_arr)

    xv_FT = x_FT[:,np.newaxis] * np.ones(alpha_arr.size)[np.newaxis,:]
    xG_FT = x_FT[:,np.newaxis] * alphaG_arr[np.newaxis,:]
    xL_FT = x_FT[:,np.newaxis] * alphaL_arr[np.newaxis,:]

    hv1_FT =   2*pi*xv_FT
    hv2_FT = -(2*pi*xv_FT)**2

    hG1_FT = -(pi*xG_FT)**2/(2*ln2) 
    hG2_FT =  (pi*xG_FT)**2/(2*ln2)*((pi*xG_FT)**2/(2*ln2)-2)

    hL1_FT =  -pi*np.abs(xL_FT)
    hL2_FT =   pi*np.abs(xL_FT)*(pi*np.abs(xL_FT)-1)

    GV2_FT = (GG_FT(xG_FT)*GL_FT(xL_FT))**2

    S_G1G1 = 2*np.sum(hG1_FT * hG1_FT * GV2_FT,0)*dksi
    S_G1G2 = 2*np.sum(hG1_FT * hG2_FT * GV2_FT,0)*dksi
    S_G2G2 = 2*np.sum(hG2_FT * hG2_FT * GV2_FT,0)*dksi

    S_G1L1 = 2*np.sum(hG1_FT * hL1_FT * GV2_FT,0)*dksi
    S_G1L2 = 2*np.sum(hG1_FT * hL2_FT * GV2_FT,0)*dksi
    S_G2L1 = 2*np.sum(hG2_FT * hL1_FT * GV2_FT,0)*dksi
    S_G2L2 = 2*np.sum(hG2_FT * hL2_FT * GV2_FT,0)*dksi

    S_L1L1 = 2*np.sum(hL1_FT * hL1_FT * GV2_FT,0)*dksi
    S_L1L2 = 2*np.sum(hL1_FT * hL2_FT * GV2_FT,0)*dksi
    S_L2L2 = 2*np.sum(hL2_FT * hL2_FT * GV2_FT,0)*dksi

##    # Calculate these from the existing integrals to save time
##    S_v1v1 =-2*np.sum(hv1_FT * hv1_FT * GV2_FT,0)*dksi
##    S_v2v2 = 2*np.sum(hv2_FT * hv2_FT * GV2_FT,0)*dksi
##
##    S_v2G1 = 2*np.sum(hv2_FT * hG1_FT * GV2_FT,0)*dksi
##    S_v2G2 = 2*np.sum(hv2_FT * hG2_FT * GV2_FT,0)*dksi
##    S_v2L1 = 2*np.sum(hv2_FT * hL1_FT * GV2_FT,0)*dksi
##    S_v2L2 = 2*np.sum(hv2_FT * hL2_FT * GV2_FT,0)*dksi

    S_v1v1 =        -4/alphaL_arr**2 * S_L1L1
    S_v2v2 = 64*ln2**2/alphaG_arr**4 * S_G1G1

    S_v2G1 =     8*ln2/alphaG_arr**2 * S_G1G1
    S_v2G2 =     8*ln2/alphaG_arr**2 * S_G1G2
    S_v2L1 =     8*ln2/alphaG_arr**2 * S_G1L1
    S_v2L2 =     8*ln2/alphaG_arr**2 * S_G1L2

    if alpha_arr.size == 1:
        return (S_v1v1[0], S_v2v2[0],
                S_v2G1[0], S_v2G2[0], S_v2L1[0], S_v2L2[0],
                S_G1G1[0], S_G1G2[0], S_G2G2[0],
                S_G1L1[0], S_G1L2[0], S_G2L1[0], S_G2L2[0],
                S_L1L1[0], S_L1L2[0], S_L2L2[0])

    else:
        return (S_v1v1, S_v2v2,
                S_v2G1, S_v2G2, S_v2L1, S_v2L2,
                S_G1G1, S_G1G2, S_G2G2,
                S_G1L1, S_G1L2, S_G2L1, S_G2L2,
                S_L1L1, S_L1L2, S_L2L2)

def approximate_difference_FT(x_FT,av,aG,aL,tv,tG,tL,dxv,dxG,dxL,alpha):
    alphaG = f_alphaG(alpha)
    alphaL = f_alphaL(alpha)

    cv1  =     (av - tv)*dxv
    cG1  =     (aG - tG)*dxG
    cL1  =     (aL - tL)*dxL
    cv2  = 0.5*(tv**2 - 2*av*tv + av)*dxv**2
    cG2  = 0.5*(tG**2 - 2*aG*tG + aG)*dxG**2
    cL2  = 0.5*(tL**2 - 2*aL*tL + aL)*dxL**2

    xv_FT = x_FT
    xG_FT = x_FT*alphaG
    xL_FT = x_FT*alphaL

    hv1_FT =   2j*pi*xv_FT
    hv2_FT = -(2 *pi*xv_FT)**2
    hG1_FT = -(pi*xG_FT)**2/(2*ln2) 
    hG2_FT =  (pi*xG_FT)**2/(2*ln2)*((pi*xG_FT)**2/(2*ln2)-2)
    hL1_FT =  -pi*np.abs(xL_FT)
    hL2_FT =   pi*np.abs(xL_FT)*(pi*np.abs(xL_FT)-1)

    IGv2 = GG_FT(2**0.5*xG_FT) * GL_FT(2*xL_FT)

    ev1 = cv1 * hv1_FT
    eG1 = cG1 * hG1_FT 
    eL1 = cL1 * hL1_FT
    
    ev2 = cv2 * hv2_FT
    eG2 = cG2 * hG2_FT 
    eL2 = cL2 * hL2_FT

##    Ierr = (ev + eG + eL) * IGv
##    Ierr = ((1 + ev) * (1 + eG) * (1 + eL) - 1) * IGv
##    Ierr = (ev + eG + ev*eG + eL + ev*eL + eG*eL + ev*eG*eL) * IGv
    Ierr2 = np.abs(ev1 + ev2 + eG1 + eG2 + eL1 + eL2 + eG1*eL1 + eG1*eL2 + eG2*eL1)**2 * IGv2

    return Ierr2

def approximate_error(av,aG,aL,tv,tG,tL,dxv,dxG,dxL,error_integrals):

    cv1  =     (av - tv)*dxv
    cG1  =     (aG - tG)*dxG
    cL1  =     (aL - tL)*dxL
    cv2  = 0.5*(tv**2 - 2*av*tv + av)*dxv**2
    cG2  = 0.5*(tG**2 - 2*aG*tG + aG)*dxG**2
    cL2  = 0.5*(tL**2 - 2*aL*tL + aL)*dxL**2

    (S_v1v1, S_v2v2,
     S_v2G1, S_v2G2, S_v2L1, S_v2L2,
     S_G1G1, S_G1G2, S_G2G2,
     S_G1L1, S_G1L2, S_G2L1, S_G2L2,
     S_L1L1, S_L1L2, S_L2L2) = error_integrals

    E_RMS2 = (
              cv1**2        * S_v1v1 + 
              cv2**2        * S_v2v2 + 

              2 * cv2 * cG1 * S_v2G1 +
              2 * cv2 * cG2 * S_v2G2 +
              2 * cv2 * cL1 * S_v2L1 + 
              2 * cv2 * cL2 * S_v2L2 + 

              cG1**2        * S_G1G1 +
              2 * cG1 * cG2 * S_G1G2 +
              cG2**2        * S_G2G2 +

              2 * cG1 * cL1 * S_G1L1 +
              2 * cG1 * cL2 * S_G1L2 +
              2 * cG2 * cL1 * S_G2L1 +
              2 * cG2 * cL2 * S_G2L2 +

              cL1**2        * S_L1L1 + 
              2 * cL1 * cL2 * S_L1L2 + 
              cL2**2        * S_L2L2 + 
  
              0)

    return E_RMS2**0.5

    
def average_error_simple(dxv,dxG,dxL,alpha,ksi_max = 10):

    (S_v1v1, S_v2v2,
     S_v2G1, S_v2G2, S_v2L1, S_v2L2,
     S_G1G1, S_G1G2, S_G2G2,
     S_G1L1, S_G1L2, S_G2L1, S_G2L2,
     S_L1L1, S_L1L2, S_L2L2) = error_integrals(alpha,ksi_max = ksi_max)

    E_RMS2 = (    
              S_v2v2*dxv**4/120 +
              S_v2G2*dxG**2*dxv**2/72 +
              S_v2L2*dxL**2*dxv**2/72 +
              S_G2G2*dxG**4/120 +
              S_G2L2*dxG**2*dxL**2/72 +
              S_L2L2*dxL**4/120
              )

    return E_RMS2**0.5


def average_error_optimized(dxv,dxG,dxL,alpha,ksi_max = 10):

    dxvG = dxv / f_alphaG(alpha)

    R_Gv = 8*ln2 
    R_GG = 2 - 1/(C1_GG + C2_GG*alpha**(2/1.5))**1.5
    R_GL = -2*ln2*alpha**2

    R_LG = 1/(C1_LG*alpha**(1/2.25) + C2_LG*alpha**(4/2.25))**2.25
    R_LL = 1

    (S_v1v1, S_v2v2,
     S_v2G1, S_v2G2, S_v2L1, S_v2L2,
     S_G1G1, S_G1G2, S_G2G2,
     S_G1L1, S_G1L2, S_G2L1, S_G2L2,
     S_L1L1, S_L1L2, S_L2L2) = error_integrals(alpha,ksi_max = ksi_max)

    E_RMS2 = (0
                + S_v2v2*dxv**4/120 +

                - R_GG*S_v2G1*dxG**2*dxv**2/72
                - R_GL*S_v2G1*dxL**2*dxv**2/72
                - R_Gv*S_v2G1*dxv**2*dxvG**2/60
                
                + S_v2G2*dxG**2*dxv**2/72

                - R_LG*S_v2L1*dxG**2*dxv**2/72
                - R_LL*S_v2L1*dxL**2*dxv**2/72 

                + S_v2L2*dxL**2*dxv**2/72 

                + R_GG**2*S_G1G1*dxG**4/120
                + R_GG*R_GL*S_G1G1*dxG**2*dxL**2/72
                + R_GG*R_Gv*S_G1G1*dxG**2*dxvG**2/72
                + R_GL**2*S_G1G1*dxL**4/120
                + R_GL*R_Gv*S_G1G1*dxL**2*dxvG**2/72
                + R_Gv**2*S_G1G1*dxvG**4/120

                - R_GG*S_G1G2*dxG**4/60
                - R_GL*S_G1G2*dxG**2*dxL**2/72
                - R_Gv*S_G1G2*dxG**2*dxvG**2/72

                + R_GG**2*S_G2G2*dxG**6/3360
                + R_GG*R_GL*S_G2G2*dxG**4*dxL**2/1440
                + R_GG*R_Gv*S_G2G2*dxG**4*dxvG**2/1440
                + R_GL**2*S_G2G2*dxG**2*dxL**4/1440
                + R_GL*R_Gv*S_G2G2*dxG**2*dxL**2*dxvG**2/864
                + R_Gv**2*S_G2G2*dxG**2*dxvG**4/1440
                + S_G2G2*dxG**4/120

                + R_Gv*R_LG*S_G1L1*dxG**2*dxvG**2/72
                + R_Gv*R_LL*S_G1L1*dxL**2*dxvG**2/72
                + R_GG*R_LG*S_G1L1*dxG**4/60
                + R_GG*R_LL*S_G1L1*dxG**2*dxL**2/72
                + R_GL*R_LG*S_G1L1*dxG**2*dxL**2/72
                + R_GL*R_LL*S_G1L1*dxL**4/60

                - R_GG*S_G1L2*dxG**2*dxL**2/72
                - R_GL*S_G1L2*dxL**4/60
                - R_Gv*S_G1L2*dxL**2*dxvG**2/72

                - R_LG*S_G2L1*dxG**4/60
                - R_LL*S_G2L1*dxG**2*dxL**2/72

                + S_G2L2*dxG**2*dxL**2/72

                + R_LG**2*S_L1L1*dxG**4/120
                + R_LG*R_LL*S_L1L1*dxG**2*dxL**2/72
                + R_LL**2*S_L1L1*dxL**4/120

                - R_LG*S_L1L2*dxG**2*dxL**2/72
                - R_LL*S_L1L2*dxL**4/60

                + R_LG**2*S_L2L2*dxG**4*dxL**2/1440
                + R_LG*R_LL*S_L2L2*dxG**2*dxL**4/1440
                + R_LL**2*S_L2L2*dxL**6/3360
                + S_L2L2*dxL**4/120
              )

    return E_RMS2**0.5







