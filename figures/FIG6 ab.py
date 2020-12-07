# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:43:37 2019

@author: vandenbekerom.1
"""
##from matplotlib import use
##use('PDF')
import matplotlib.pyplot as plt
from radis import plot_diff, SpectrumFactory, get_residual
from radis import load_spec
from publib import set_style, fix_style
from socket import gethostname
import pickle

#%% Machine-dependant params

SAVE=True
if gethostname() == 'LAPTOP-UC145I0C':
    prefix=r'C:/'
else:
    prefix=r"D:/GitHub/radis-local/CO2_test2/"

database=[prefix+'HITEMP/02_2000-2125_HITEMP2010.par',
          prefix+'HITEMP/02_2125-2250_HITEMP2010.par',
          prefix+'HITEMP/02_2250-2500_HITEMP2010.par']

#%% Computation parameters
wmin = 2000
wmax = 2400
dv = 0.002
T = 3000.0 #K
p = 0.1 #bar
broadening_max_width=5    # lineshape broadening width, ext

spectra_default = {}
spectra_DLM = {}
spectra_DLM_opt = {}

Nlines = [101, 1001, 10017, 100725, 1813040, 1813040]
plt.ion()

for N in Nlines:
    spectra_default[N] = load_spec('../spec/spectra_default_[{0}].spec'.format(N), binary=True)
    spectra_DLM_opt[N] = load_spec('../spec/spectra_DLM_opt[{0}].spec'.format(N), binary=True)

# %% Plot last spectra (opt)

fig, [ax0, ax1] = plot_diff(spectra_default[N], spectra_DLM_opt[N], 'abscoeff')

ax0.set_ylim((0, 0.5))
ax1.set_ylim((-0.005, 0.005))

axins = ax0.inset_axes([0.06, 0.3, 0.3, 0.4])
axins.ticklabel_format(useOffset=False, axis='x')
axins.tick_params(which='major', labelsize=14)
axins.get_yaxis().set_visible(False)
axins.plot(*spectra_default[N].get('abscoeff', wunit='cm-1'), color='k', lw=2)
axins.plot(*spectra_DLM_opt[N].get('abscoeff', wunit='cm-1'), color='r', lw=1)
axins.set_xlim((2200, 2205))
axins.set_ylim((0.0125, 0.32))
ax0.indicate_inset_zoom(axins)

fig.savefig('output/Fig6a.pdf')

fig, [ax0, ax1] = plot_diff(spectra_default[N], spectra_DLM_opt[N], 'abscoeff')

ax0.set_ylim((0, 0.5))
ax1.set_ylim((-0.002, 0.002))
ax0.set_xlim((2200, 2205))
ax1.set_xlim((2200, 2205))

axins = ax0.inset_axes([0.12, 0.53, 0.3, 0.4])
axins.ticklabel_format(useOffset=False, axis='x')
axins.tick_params(which='major', labelsize=14)
axins.get_yaxis().set_visible(False)
axins.plot(*spectra_default[N].get('abscoeff', wunit='cm-1'), color='k', lw=2)
axins.plot(*spectra_DLM_opt[N].get('abscoeff', wunit='cm-1'), color='r', lw=1)
axins.set_xlim((2203.15, 2203.37))
axins.set_ylim((0.02, 0.32))
ax0.indicate_inset_zoom(axins)
ax0.get_legend().remove()

fig.savefig('output/Fig6b.pdf')

plt.show()
