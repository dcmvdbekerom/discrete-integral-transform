# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:43:37 2019

@author: vandenbekerom.1
"""

import matplotlib.pyplot as plt
from radis import plot_diff, SpectrumFactory, get_residual
from publib import set_style, fix_style

#%% Machine-dependant params

SAVE=False  # save figs
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


#%% Calculate Reference 

sf = SpectrumFactory(wavenum_min=wmin, wavenum_max=wmax, 
                  pressure=p,
                  isotope='1,2,3',
                  wstep=dv,
                  broadening_max_width=broadening_max_width, 
                  cutoff=0, # 1e-27,
                  verbose=2,
                  )

#%%
spectra_default = {}
spectra_DLM = {}
spectra_DLM_opt = {}

Nlines_target = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]  # note: 1e7 will be less (not enough lines in database)
Nlines = []

for Ntarget in Nlines_target:
    
    # Load HITEMP database for this species/range (433984 rows)
    sf.load_databank(path=database,
                     format='hitran',
                     parfuncfmt='hapi',
                     include_neighbouring_lines=False)
    # Reduce number of lines to approximately Nlines:
    sf.df0 = sf.df0[::max(1,int(len(sf.df0)//Ntarget))]
    N = len(sf.df0)
    Nlines.append(N)
    
    #%% Calculate normal

    sf.params['optimization'] = None
    sf.params['broadening_method'] = 'voigt'
    sf.misc['chunksize'] = 1e7
    s_none = sf.eq_spectrum(T)
    s_none.name = 'Default ({0:.1f}s)'.format(s_none.conditions['calculation_time'])
    spectra_default[N] = s_none

    # %% Calculate DLM
    
    sf.params['optimization'] = "simple"
    sf.params['broadening_method'] = 'fft'
    s_dlm = sf.eq_spectrum(T)
    s_dlm.name = 'DLM ({0:.1f}s)'.format(s_dlm.conditions['calculation_time'])
    spectra_DLM[N] = s_dlm

    # %% Calculate DLM (optimized)
    
    sf.params['optimization'] = "min-RMS"
    sf.params['broadening_method'] = 'fft'
    s_dlm_opt = sf.eq_spectrum(T)
    s_dlm_opt.name = 'DLM ({0:.1f}s)'.format(s_dlm_opt.conditions['calculation_time'])
    spectra_DLM_opt[N] = s_dlm_opt

# %% Plot last spectra
plot_diff(spectra_default[N], spectra_DLM[N], 'abscoeff')
if SAVE: plt.savefig('out/fig6b_DLMpaper_benchmark_Nlines_2spectra.png') 
if SAVE: plt.savefig('out/fig6b_DLMpaper_benchmark_Nlines_2spectra.pdf') 

# %% Plot last spectra (opt)
plot_diff(spectra_default[N], spectra_DLM_opt[N], 'abscoeff')
if SAVE: plt.savefig('out/fig6_DLMpaper_benchmark_Nlines_2spectra_minRMS.png') 
if SAVE: plt.savefig('out/fig6_DLMpaper_benchmark_Nlines_2spectra_minRMS.pdf') 

# %% Plot last spectra
plot_diff(spectra_DLM[N], spectra_DLM_opt[N], 'abscoeff')
if SAVE: plt.savefig('out/DLMpaper_benchmark_Nlines_2spectra_simple_vs_minRMS.png') 
if SAVE: plt.savefig('out/DLMpaper_benchmark_Nlines_2spectra_simple_vs_minRMS.pdf') 

#%% Compare residual
set_style('origin')
plt.figure()
plt.plot(Nlines, [get_residual(spectra_default[N], spectra_DLM_opt[N], 'abscoeff')
                  for N in Nlines], '--ok')
plt.xlabel('Number of lines (#)')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Residual (accuracy)')
plt.legend()
fix_style('origin')
if SAVE: plt.savefig('out/DLMpaper_benchmark_Nlines_residual.png')
if SAVE: plt.savefig('out/DLMpaper_benchmark_Nlines_residual.pdf')


#%% Compare performance (simple)
set_style('origin')
plt.figure()
plt.plot(Nlines, [spectra_default[N].conditions['calculation_time'] for N in Nlines], '--ok')
plt.plot(Nlines, [spectra_DLM[N].conditions['calculation_time'] for N in Nlines], '--or')
plt.xlabel('Number of lines (#)')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Calculation time (s)')
plt.legend()
fix_style('origin')
if SAVE: plt.savefig('out/DLMpaper_benchmark_Nlines_time.png')
if SAVE: plt.savefig('out/DLMpaper_benchmark_Nlines_time.pdf')


#%% Compare performance (opt)
set_style('origin')
plt.figure()
plt.plot(Nlines, [spectra_default[N].conditions['calculation_time'] for N in Nlines], '--ok')
plt.plot(Nlines, [spectra_DLM_opt[N].conditions['calculation_time'] for N in Nlines], '--or')
plt.xlabel('Number of lines (#)')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Calculation time (s)')
plt.legend()
fix_style('origin')
if SAVE: plt.savefig('out/fig5_DLMpaper_benchmark_Nlines_time_minRMS.png')
if SAVE: plt.savefig('out/fig5_DLMpaper_benchmark_Nlines_time_minRMS.pdf')

#%% Data for article

print('\n'*3+'>>>>>> for Article')
for N in Nlines:
    print('\n{0} lines'.format(spectra_default[N].conditions['lines_calculated']))
    print('noDLM : {0:.1f}s \t --> {1:.1e} lines.points/s'.format(
            spectra_default[N].conditions['calculation_time'],
            (len(spectra_default[N].get_wavenumber())*spectra_default[N].conditions['lines_calculated']/
                    spectra_default[N].conditions['calculation_time']),
            ))
    print('DLM : {0:.1f}s \t --> {1:.1e} lines.points/s'.format(
            spectra_DLM_opt[N].conditions['calculation_time'],
            (len(spectra_DLM_opt[N].get_wavenumber())*spectra_DLM[N].conditions['lines_calculated']/
                    spectra_DLM_opt[N].conditions['calculation_time']),
            ))

