from radis import SpectrumFactory

sf = SpectrumFactory(2000, 2400, # cm-1
                  molecule='H2O',
                  isotope='1,2,3',
                  wstep=0.002,
                  )

sf.fetch_databank('hitemp',
                  )

s = sf.eq_spectrum(Tgas=1200, #K 
                       pressure=0.2, #bar 
                       mole_fraction=0.1,
                       path_length= 2.0, #cm
                       )
s.plot('abscoeff')
