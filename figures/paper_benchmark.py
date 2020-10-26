# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:43:37 2019

@author: vandenbekerom.1
"""

import numpy as np
import matplotlib.pyplot as plt
from radis.misc import ProgressBar
from radis import Spectrum, plot_diff, calc_spectrum

#%% Machine-dependant params

#HT_folder = './'   # '../../HITEMP/'
#database='CO2.par'
HT_folder = 'HITEMP/'   # '../../HITEMP/'
database='HITEMP/02_2250-2500_HITEMP2010.par'
database = r'D:\GitHub\radis-local\CO2_test2\HITEMP\02_2125-2500_HITEMP2010.par'

#%% Computation parameters
wmin = 2250
wmax = 2500
dv = 0.002
iso = 3     # isotope max (delete cached database CO2_hitemp.npy if changing)
T = 3000.0 #K
p = 0.1 #bar
broadening_max_width=5    # lineshape broadening width, ext


#%% Calculate Reference 

s_none = calc_spectrum(wavenum_min=wmin, wavenum_max=wmax, 
                  pressure=p,
                  Tgas=T,
                  isotope=list(range(1,iso+1)), 
                  molecule='CO2',
                  wstep=dv,
                  broadening_max_width=broadening_max_width, 
                  cutoff=0, # 1e-27,
                  verbose=4,
                  optimization=None,
                  databank=database,
                  )
s_none.name = 'Default ({0:1f}s)'.format(s_none.conditions['calculation_time'])

# %% Calculate DLM (simple)

s_dlm = calc_spectrum(wavenum_min=wmin, wavenum_max=wmax, 
                  pressure=p,
                  Tgas=T,
                  isotope=list(range(1,iso+1)), 
                  molecule='CO2',
                  wstep=dv,
                  broadening_max_width=broadening_max_width, 
                  cutoff=0, # 1e-27,
                  verbose=4,
                  optimization='simple',
                  databank=database,
                  )
s_dlm.name = 'DLM ({0:1f}s)'.format(s_dlm.conditions['calculation_time'])

# # %% CAlculate DLM (optimized)
# s_dlm_opt = calc_spectrum(wavenum_min=wmin, wavenum_max=wmax, 
#                   pressure=p,
#                   Tgas=T,
#                   isotope=list(range(1,iso+1)), 
#                   molecule='CO2',
#                   wstep=dv,
#                   broadening_max_width=broadening_max_width, 
#                   cutoff=0, # 1e-27,
#                   verbose=4,
#                   optimization='min-RMS',
#                   databank=database,
#                   )
# s_dlm_opt.name = 'DLM ({0:1f}s)'.format(s_dlm.conditions['calculation_time'])


#%% Compare

plot_diff(s_none, s_dlm, 'abscoeff', show_residual=True)
plot_diff(s_dlm, s_dlm_opt, 'abscoeff', show_residual=True)
