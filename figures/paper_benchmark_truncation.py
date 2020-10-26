# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:43:37 2019

@author: vandenbekerom.1
"""

import numpy as np
import matplotlib.pyplot as plt
from radis.misc import ProgressBar
from radis import Spectrum, plot_diff, calc_spectrum, SpectrumFactory, get_residual
from publib import set_style, fix_style
from socket import gethostname

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
p = 1 #bar
broadening_max_width=5    # lineshape broadening width, ext


#%% Calculate Reference 

#%%
spectra_default = {}      # calculated with Whiting's Voigt approximatino
spectra_convolve = {}     # calculated  with no approximatino

broadening_max_widths = [0.05, 0.1, 1, 5, 10, 15, 20, 25, 50]
Nlines = 1e4

#%% Calculate normal

for width in broadening_max_widths:
        
    sf = SpectrumFactory(wavenum_min=wmin, wavenum_max=wmax, 
                      pressure=p,
                      isotope='1,2,3', 
                      wstep=dv,
                      broadening_max_width=width, 
                      cutoff=0, # 1e-27,
                      chunksize=1e7,  # no DLM
#                      verbose=4,
                      )
    sf.params['optimization'] = None
    sf.params['broadening_method'] = 'voigt'
    
    # Load HITEMP database for this species/range (433984 rows)
    sf.load_databank(path=database,
                     format='hitran',
                     parfuncfmt='hapi',
                     include_neighbouring_lines=False)
    # Reduce number of lines to approximately Nlines:
    sf.df0 = sf.df0[::max(1, int(len(sf.df0)//Nlines))]

    s_none = sf.eq_spectrum(T)
    s_none.name = 'Truncation {0:.1f}'.format(width)+'cm$^{-1}$'+' ({0:.1f}s)'.format(
                                    s_none.conditions['calculation_time'])
    spectra_default[width] = s_none
    
    
    sf.params['broadening_method'] = 'convolve'
    s_convolve = sf.eq_spectrum(T)
    s_convolve.name = 'Truncation {0:.1f}'.format(width)+'cm$^{-1}$'+' ({0:.1f}s)'.format(
                                    s_convolve.conditions['calculation_time'])
    spectra_convolve[width] = s_convolve

# %% Calculate DLM  (simple)

sf.params['optimization'] = 'simple'
sf.params['broadening_method'] = 'fft'
s_dlm = sf.eq_spectrum(T)
s_dlm.name = 'New method [simple] ({0:.1f}s)'.format(s_dlm.conditions['calculation_time'])
spectra_DLM = s_dlm

    
# %% Calculate DLM  (min-RMS)

sf.params['optimization'] = 'min-RMS'
sf.params['broadening_method'] = 'fft'
s_dlm_opt = sf.eq_spectrum(T)
s_dlm_opt.name = 'New method ({0:.1f}s)'.format(s_dlm_opt.conditions['calculation_time'])
spectra_DLM_opt = s_dlm_opt


    
# %% Plot last spectra  Compare simple vs weight

# fig, [ax0, ax1] = plot_diff(spectra_default[broadening_max_widths[1]], spectra_DLM, 'abscoeff')
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# axins = inset_axes(ax0, width=1.8, height=1.3, loc=6)
# axins.ticklabel_format(useOffset=False, axis='x')
# axins.get_yaxis().set_visible(False)
# axins.plot(*spectra_default[broadening_max_widths[1]].copy().crop(2314.9, 2315.24).get('abscoeff'), color='k', lw=2)
# axins.plot(*spectra_DLM.copy().crop(2314.9, 2315.24).get('abscoeff'), color='r', lw=1)
# if SAVE: plt.savefig('out/DLMpaper_benchmark_truncation_2spectra_minwidth.png') 
# if SAVE: plt.savefig('out/DLMpaper_benchmark_truncation_2spectra_minwidth.pdf')



fig, [ax0, ax1] = plot_diff(spectra_default[broadening_max_widths[-1]], spectra_DLM_opt, 'abscoeff')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax0, width=1.8, height=1.3, loc=6)
axins.ticklabel_format(useOffset=False, axis='x')
axins.get_yaxis().set_visible(False)
axins.plot(*spectra_default[broadening_max_widths[-1]].copy().crop(2314.9, 2315.24).get('abscoeff'), color='k', lw=2)
axins.plot(*spectra_DLM_opt.copy().crop(2314.9, 2315.24).get('abscoeff'), color='r', lw=1)

if SAVE: plt.savefig('out/fig11_DLMpaper_benchmark_truncation_optimized.png') 
if SAVE: plt.savefig('out/fig11_DLMpaper_benchmark_truncation_optimized.pdf') 

fig, [ax0, ax1] = plot_diff(spectra_default[broadening_max_widths[-1]], spectra_DLM, 'abscoeff')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax0, width=1.8, height=1.3, loc=6)
axins.ticklabel_format(useOffset=False, axis='x')
axins.get_yaxis().set_visible(False)
axins.plot(*spectra_default[broadening_max_widths[-1]].copy().crop(2314.9, 2315.24).get('abscoeff'), color='k', lw=2)
axins.plot(*spectra_DLM.copy().crop(2314.9, 2315.24).get('abscoeff'), color='r', lw=1)

if SAVE: plt.savefig('out/fig11a_DLMpaper_benchmark_truncation_simple.png') 
if SAVE: plt.savefig('out/fig11a_DLMpaper_benchmark_truncation_simple.pdf') 

# %% Compare DIT method vs different truncation 

fig, [ax0, ax1] = plot_diff(spectra_default[broadening_max_widths[1]], spectra_DLM_opt, 'abscoeff')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax0, width=1.8, height=1.3, loc=6)
axins.ticklabel_format(useOffset=False, axis='x')
axins.get_yaxis().set_visible(False)
axins.plot(*spectra_default[broadening_max_widths[1]].copy().crop(2314.9, 2315.24).get('abscoeff'), color='k', lw=2)
axins.plot(*spectra_DLM_opt.copy().crop(2314.9, 2315.24).get('abscoeff'), color='r', lw=1)
if SAVE: plt.savefig('out/fig7_DLMpaper_benchmark_truncation_2spectra_minwidth_opt.png') 
if SAVE: plt.savefig('out/fig7_DLMpaper_benchmark_truncation_2spectra_minwidth_opt.pdf')

fig, [ax0, ax1] = plot_diff(spectra_default[broadening_max_widths[-1]], spectra_DLM_opt, 'abscoeff')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax0, width=1.8, height=1.3, loc=6)
axins.ticklabel_format(useOffset=False, axis='x')
axins.get_yaxis().set_visible(False)
axins.plot(*spectra_default[broadening_max_widths[-1]].copy().crop(2314.9, 2315.24).get('abscoeff'), color='k', lw=2)
axins.plot(*spectra_DLM_opt.copy().crop(2314.9, 2315.24).get('abscoeff'), color='r', lw=1)

if SAVE: plt.savefig('out/fig8_DLMpaper_benchmark_truncation_2spectra_maxwidth_opt.png') 
if SAVE: plt.savefig('out/fig8_DLMpaper_benchmark_truncation_2spectra_maxwidth_opt.pdf') 



#%% Compare residual
set_style('origin')
fig, [ax0, ax1] = plt.subplots(2, 1, sharex=True)
ax0.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
ax0.plot(broadening_max_widths, [get_residual(spectra_default[N], spectra_convolve[broadening_max_widths[-1]], 'abscoeff')
                  for N in broadening_max_widths], '-ok')
ax0.plot(broadening_max_widths, [get_residual(spectra_convolve[N], spectra_convolve[broadening_max_widths[-1]], 'abscoeff')
                  for N in broadening_max_widths], '--ok')
#ax0.xlabel('Lineshape Truncation (cm-1)')
ax0.set_xscale('log')
ax0.set_yscale('log')
(xmin, xmax) = ax0.get_xlim()
# ax0.plot(xmax, get_residual(spectra_default[broadening_max_widths[-1]], spectra_DLM, 'abscoeff'), 'or') # Note: no truncation : show a constant line
ax0.plot(xmax, get_residual(spectra_convolve[broadening_max_widths[-1]], spectra_DLM, 'abscoeff'), 'or') # Note: no truncation : show a constant line
#plt.yscale('log')
ax0.set_ylabel('Residual')
ax0.text(6, 5.3e-7, 'Whiting Voigt approx.', color='k')
ax0.text(25, 1e-8, 'New method', color='r')
plt.legend()
fix_style('origin', ax0)
#if SAVE: plt.savefig('out/DLMpaper_benchmark_truncation_residual.png')
#if SAVE: plt.savefig('out/DLMpaper_benchmark_truncation_residual.pdf')


#Compare performance
#set_style('origin')
#plt.figure()
ax1.plot(broadening_max_widths, [spectra_default[N].conditions['calculation_time'] for N in broadening_max_widths], '-ok')
ax1.plot(broadening_max_widths, [spectra_convolve[N].conditions['calculation_time'] for N in broadening_max_widths], '--ok')
# ax1.plot((broadening_max_widths[0], broadening_max_widths[-1]), 
#          2*[spectra_DLM.conditions['calculation_time']], '-r') # Note: no truncation : show a constant line
ax1.plot(xmax, spectra_DLM.conditions['calculation_time'], 'or') # Note: no truncation : show a constant line
ax1.set_xlabel('Lineshape Truncation (cm-1)')
ax1.set_xlim((xmin, xmax))
ax1.set_xscale('log')
ax1.set_yscale('log')
#plt.yscale('log')
ax1.set_ylabel('Calculation time (s)')
ax1.text(25, 2, 'New method', color='r')
ax1.text(43, 11, 'Whiting ', color='k')
plt.legend()
fix_style('origin', ax1)
if SAVE: plt.savefig('out/fig9_DLMpaper_benchmark_truncation.png')
if SAVE: plt.savefig('out/fig9_DLMpaper_benchmark_truncation.pdf')

#%% Data for article

print('\n'*3+'>>>>>> for Article')
for N in broadening_max_widths:
    print('width {2} cm-1: {0:.1f}s \t --> {1:.1e} lines.points/s'.format(
            spectra_default[N].conditions['calculation_time'],
            (len(spectra_default[N].get_wavenumber())*spectra_default[N].conditions['lines_calculated']/
                    spectra_default[N].conditions['calculation_time']),
                    spectra_default[N].conditions['broadening_max_width'],
            ))
print('DLM (full) : {0:.1f}s \t --> {1:.1e} lines.points/s'.format(
        spectra_DLM.conditions['calculation_time'],
        (len(spectra_DLM.get_wavenumber())*spectra_DLM.conditions['lines_calculated']/
                spectra_DLM.conditions['calculation_time']),
        ))

