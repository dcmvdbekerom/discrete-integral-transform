# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:43:37 2019

@author: vandenbekerom.1
"""

import numpy as np
import matplotlib.pyplot as plt
from radis.misc import ProgressBar
from radis import Spectrum, plot_diff, calc_spectrum, SpectrumFactory, get_residual
from radis import load_spec
from publib import set_style, fix_style
from socket import gethostname
import pickle

#%% Machine-dependant params

SAVE=False
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
p = 1 #bar
broadening_max_width=5    # lineshape broadening width, ext


#%% Calculate Reference 

#%%
spectra_default = {}      # calculated with Whiting's Voigt approximatino
spectra_convolve = {}     # calculated  with no approximatino

##broadening_max_widths = [0.05, 0.1, 1, 5, 10, 15, 20, 25, 50]
broadening_max_widths = [0.1,50]
Nlines = 1e4

#%% Calculate normal

for width in broadening_max_widths:
        
    spectra_default[width] = load_spec('../spec_fig78/spectra_default[{0}].spec'.format(width), binary=True)
    spectra_convolve[width] = load_spec('../spec_fig78/spectra_convolve[{0}].spec'.format(width), binary=True)

spectra_DLM_opt = load_spec('../spec_fig78/spectra_DLM_opt.spec')


plt.ion()

##fig, [ax0, ax1] = plot_diff(spectra_default[broadening_max_widths[1]], spectra_DLM_opt, 'abscoeff', figsize=(10,6))
fig, [ax0, ax1] = plot_diff(spectra_default[broadening_max_widths[1]], spectra_DLM_opt, 'abscoeff', figsize=(10,6))
                           
ax0.set_ylim((0, 0.9))
ax1.set_ylim((-0.0025, 0.0025))

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

axins = ax0.inset_axes([0.115, 0.25, 0.35, 0.45])
axins.ticklabel_format(useOffset=False, axis='x')
axins.tick_params(which='major', labelsize=14)
axins.get_yaxis().set_visible(False)
axins.plot(*spectra_default[broadening_max_widths[1]].get('abscoeff', wunit='cm-1'), color='k', lw=2)
axins.plot(*spectra_DLM_opt.get('abscoeff', wunit='cm-1'), color='r', lw=1)

axins.set_xlim((2235, 2238.5))


axins.set_ylim((0, 0.25))
ax0.indicate_inset_zoom(axins)

axins2 = axins.inset_axes([-0.20, 0.5, 0.8, 1])
axins2.ticklabel_format(useOffset=False, axis='x')
axins2.get_xaxis().set_visible(False)
axins2.get_yaxis().set_visible(False)
axins2.plot(*spectra_default[broadening_max_widths[1]].get('abscoeff', wunit='cm-1'), color='k', lw=2)
axins2.plot(*spectra_DLM_opt.get('abscoeff', wunit='cm-1'), color='r', lw=1)
axins2.set_xlim((2237.55, 2237.75))
axins2.set_ylim((0, 0.2))
axins.indicate_inset_zoom(axins2)


handles, labels = ax0.get_legend_handles_labels()
leg = ax0.legend(handles[:-1], labels[:-1],frameon=True,fontsize=19)
leg.set_bbox_to_anchor((2190,0.55, 200, 0.1), transform=ax0.transData )


fig.savefig('output/Fig8.pdf')
