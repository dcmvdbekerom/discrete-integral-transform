import astropy.units as u
from radis import SpectrumFactory,plot_diff
import matplotlib.pyplot as plt
import numpy as np

sf = SpectrumFactory(wavelength_min=4165 * u.nm,
                     wavelength_max=4500 * u.nm,
                     path_length=1.0 * u.cm,
                     pressure=100 * u.mbar,
                     molecule='CO2',
                     isotope='1,2,3',
                     cutoff=1e-25,              # cm/molecule
                     broadening_max_width=10,   # cm-1
                     wstep=0.001,
                     chunksize='DLM',
                     )

##sf.load_databank('CDSD-HITEMP')
sf.fetch_databank()

#The base truth spectrum
sf.misc.awL_kind = 'ZEP2'
sf.params.dlm_res_L = 0.001
sf.params.dlm_res_G = 0.001
s0 = sf.eq_spectrum(Tgas=1000 * u.K)
w,A0 = s0.get('absorbance')

#Deliberately worsen width resolution
sf.params.dlm_res_G = 0.01
plt.axhline(0,c='k')

sf.misc.awG_kind = 'linear'
s1 = sf.eq_spectrum(Tgas=1000 * u.K)
A1 = s1.get('absorbance')[1]
e1 = A0-A1
plt.plot(w,e1,label=sf.misc.awG_kind)


sf.misc.awG_kind = 'linear2'
s2 = sf.eq_spectrum(Tgas=1000 * u.K)
A2 = s2.get('absorbance')[1]
e2 = A0-A2
plt.plot(w,e2,label=sf.misc.awG_kind)

##sf.misc.awL_kind = 'min-RMS'
##s3 = sf.eq_spectrum(Tgas=1000 * u.K)
##A3 = s3.get('absorbance')[1]
##e3 = A0-A3
##plt.plot(w,e3,label=sf.misc.awL_kind)





plt.legend()

print('min-RMS  RMS-error:  ', np.sum(e1**2)**0.5)
print('min-RMS2 RMS-error:  ', np.sum(e2**2)**0.5)
##print('min-RMS RMS-error: ', np.sum(e3**2)**0.5)

plt.show()
