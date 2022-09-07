import astropy.units as u
from radis import SpectrumFactory, config, plot_diff
import matplotlib.pyplot as plt
plt.ion()
print("With Sparse Range Optimization")
print("=================================")
print("\n"*2)

# NO: 20.5s
# NO2: 6.0s
# CO2: 17.6s
# H2O:


config["SPARSE_WAVERANGE"] = True
sf = SpectrumFactory(1, 25000,         # cm-1
                  #chunksize = 2**30,
                  molecule='NO',
                  isotope='1,2,3',
                  pressure=760 * u.torr,
                  verbose=False,
                  wstep=0.01,
                  mole_fraction=1)

sf.fetch_databank('hitemp')
s = sf.eq_spectrum(700)
s.print_perf_profile()
print(sf.params['sparse_ldm'])


##print("With Sparse Range Optimization")
##print("==============================")
##print("\n"*2)
##
##config["SPARSE_WAVERANGE"] = True
##sf.params["sparse_ldm"] = True
##sf._sparse_ldm = True
##s2 = sf.eq_spectrum(700)
##print(sf.params['sparse_ldm'])
##s2.print_perf_profile()

# Plot
s.name = f'0.10.3 ({s.conditions["calculation_time"]:.0f}s)'
s.plot('absorbance')

##s2.name = f'Sparse optim ({s2.conditions["calculation_time"]:.0f}s)'
##_, [ax0, ax1] = plot_diff(s, s2)

plt.gca().set_yscale('log')
