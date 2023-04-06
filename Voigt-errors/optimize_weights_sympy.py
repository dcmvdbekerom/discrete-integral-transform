from sympy import *

tauG,tauL,log_pG,log_pL = symbols('τG τL log_pG log_pL',positive=True,real=True)
aG,aL = symbols('aG aL',real=True) #are technically allowed to go negative

(S_L1L1,
 S_L1L2,
 S_L2L2,
 S_G1L1,
 S_G1L2,
 S_G2L1,
 S_G2L2,
 S_G1G1,
 S_G1G2,
 S_G2G2) = symbols('SL1L1 SL1L2 SL2L2 SG1L1 SG1L2 SG2L1 SG2L2 SG1G1 SG1G2 SG2G2',real=True)
#ɢ₁ʟ₂
init_printing(use_unicode=True)

cG1 = (aG - tauG)*log_pG
cL1 = (aL - tauL)*log_pL
cG2 = (tauG**2 - 2*aG*tauG + aG)*log_pG**2/2
cL2 = (tauL**2 - 2*aL*tauL + aL)*log_pL**2/2


##EG = cG1**2*S_G1G1 + 2*cG1*cG2*S_G1G2 + cG2**2*S_G2G2 
##dEGdaG = diff(EG,aG)
##res = solve([dEGdaG],[aG])
##pureG = res[aG].factor()
##
##EL = cL1**2*S_L1L1 + 2*cL1*cL2*S_L1L2 + cL2**2*S_L2L2 
##dELdaL = diff(EL,aL)
##res = solve([dELdaL],[aL])
##pureL = res[aL].factor()

EV = (cG1**2*S_G1G1 + 2*cG1*cG2*S_G1G2 + cG2**2*S_G2G2 +
      2*cG1*cL1*S_G1L1 + 2*cG1*cL2*S_G1L2 + 2*cG2*cL1*S_G2L1 + 2*cG2*cL2*S_G2L2 +
      cL1**2*S_L1L1 + 2*cL1*cL2*S_L1L2 + cL2**2*S_L2L2)

dEVdaG = diff(EV,aG).factor()/log_pG
dEVdaL = diff(EV,aL).factor()/log_pL

res = solve([dEVdaG,dEVdaL],(aG,aL))

aG = res[aG].factor()
##aG = aG.collect(pureG)

aL = res[aL].factor()
##aL = aL.collect(pureL)
