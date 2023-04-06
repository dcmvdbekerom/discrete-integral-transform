from sympy import *

tv,tG,tL,dxv,dxG,dxL = symbols('tv tG tL dxv dxG dxL',positive=True,real=True)
av,aG,aL = symbols('av aG aL',real=True)

(SL1L1, SL1L2, SL2L2, SG1L1, SG1L2, SG2L1, SG2L2, SG1G1, SG1G2, SG2G2, Sv1v1, Sv2v2, SG1v2, SG2v2, SL1v2, SL2v2) = \
 symbols('SL1L1 SL1L2 SL2L2 SG1L1 SG1L2 SG2L1 SG2L2 SG1G1 SG1G2 SG2G2 Sv1v1 Sv2v2 SG1v2 SG2v2 SL1v2 SL2v2',real=True)

init_printing(use_unicode=True)

cv1 = (av - tv)*dxv
cG1 = (aG - tG)*dxG
cL1 = (aL - tL)*dxL

cv2 = (tv**2 - 2*av*tv + av)*dxv**2/2
cG2 = (tG**2 - 2*aG*tG + aG)*dxG**2/2
cL2 = (tL**2 - 2*aL*tL + aL)*dxL**2/2

EV2 = (
##        cG1**2 *SG1G1 +
##      2*cG1*cG2*SG1G2 +
##        cG2**2 *SG2G2 +
##
##      2*cG1*cL1*SG1L1 +
##      2*cG1*cL2*SG1L2 +
##      2*cG2*cL1*SG2L1 +
##      2*cG2*cL2*SG2L2 +

        cL1**2 *SL1L1 +
      2*cL1*cL2*SL1L2 +
        cL2**2 *SL2L2 +

##        cv1**2 *Sv1v1 +
##        cv2**2 *Sv2v2 +

##      2*cG1*cv2*SG1v2 +
##      2*cG2*cv2*SG2v2 +
##      2*cL1*cv2*SL1v2 +
##      2*cL2*cv2*SL2v2 +
        0)

##dEV2daG = diff(EV2,aG).factor()
dEV2daL = diff(EV2,aL).factor()
##dEV2dav = diff(EV2,av).factor()


# Solving for three at a time takes too long, so solve for 2/1 and do the rest manually:
res1 = solve([dEV2daL],(aL))
##res2 = solve([dEV2dav],(av))

##aG = res1[aG].factor()
aL = res1[aL].factor()
##av = res2[av].factor()
