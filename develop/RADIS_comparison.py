import matplotlib.pyplot as plt
from radis import calc_spectrum
from discrete_integral_transform import synthesize_spectrum
import numpy as np

conditions = {
    "molecule":"CO2",
    "isotope":1,
    "wavenum_min":667.5,
    "wavenum_max":667.7,
    "pressure":0.00646122,
    "Tgas":234.25,
    "path_length":100000,
    "broadening_max_width":3,
    "warnings":{'MissingSelfBroadeningWarning':'ignore'},
}

plt.ion()
plt.figure()

wstep = 0.003 #cm-1

s1 = calc_spectrum(**conditions,
                   wstep=wstep,
                   optimization='min-RMS',      # if None, no error !
                   name=f'min-RMS'
                   )
A = s1.get('abscoeff')[1]
A/=A.max()

v = np.arange(666.0,669.201,wstep)

v0i = s1.lines.shiftwav.values
wGi = s1.lines.hwhm_gauss.values * 2
wLi = s1.lines.hwhm_lorentz.values * 2
S0i = s1.lines.S.values

dxG = 0.1375350788016573
dxL = 0.20180288881201608

I1,S_klm = synthesize_spectrum(v,v0i,np.log(wGi),np.log(wLi),S0i,dxG=dxG,dxL=dxL,optimized=False)
I1/=I1.max()

I2,S_klm = synthesize_spectrum(v,v0i,np.log(wGi),np.log(wLi),S0i,dxG=dxG,dxL=dxL,optimized=True)
I2/=I2.max()

plt.plot(s1.get_wavenumber(),A)
plt.plot(v,I2,'--')
plt.show()
