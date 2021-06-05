from radis import SpectrumFactory
import numpy as np

sf = SpectrumFactory(2000, 2400, # cm-1
                  molecule='CO',
                  isotope='1,2,3',
                  wstep=0.002,
                  )

sf.fetch_databank('hitemp',
                  )

s = sf.eq_spectrum(Tgas=300, #K 
                   pressure=10, #bar 
                   mole_fraction=0.1,
                   path_length= 2.0, #cm
                   )

thresh = 1e-20
idx = sf.df1.S.values > thresh
print(np.sum(idx))
S0i = sf.df1.S.values[idx]
v0i = sf.df1.shiftwav.values[idx] 
wLi = sf.df1.hwhm_lorentz.values[idx] * 2
wGi = sf.df1.hwhm_gauss.values[idx] * 2

np.save('CO_db.npy',np.array([S0i,v0i,wLi,wGi]))
