# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:43:29 2018

@author: erwan
"""

from radis import load_spec, plot_diff
import matplotlib.pyplot as plt

SAVE = False

case = 'h2o_2200K_1000-2000cm-1_2200T'
#case = 'h2o_2200K_2000-3000cm-1_2200T'
#case = 'h2o_2200K_3000-4000cm-1_2200T'
#case = 'h2o_2200K_4000-5000cm-1_2200T'
#case = 'h2o_2200K_5000-6000cm-1_2200T'
#case = 'h2o_2200K_6000-7000cm-1_2200T'
#case = 'h2o_2200K_7000-8000cm-1_2200T'
#case = 'h2o_2200K_8000-9000cm-1_2200T'
#case = 'h2o_2200K_9000-10000cm-1_2200T'

#%%
import numpy as np
def round_to_n(x, n):
    " Round x to n significant figures "
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)

def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'$\mathregular{{{m:s}\cdot 10^{{{e:d}}} }}$'.format(m=m, e=int(e))

def str_fmt(x, n=0):
    " Format x into nice Latex rounding to n"
#    power = int(np.log10(round_to_n(x, 0)))
#    f_SF = round_to_n(x, n) * pow(10, -power)
#    return as_si(f_SF, power)
    return as_si(x, n)

#%%

s_ref = load_spec('../spec/'+case+'.spec', binary=True)
s_opt = load_spec('../spec/'+case+'_optim.spec', binary=True)
s_dlm = load_spec('../spec/h2o_2200K_1000-2000cm-1_2200K_DLM.spec', binary=True)

s_ref.update()
s_opt.update()

# compare times
t_ref = s_ref.conditions['calculation_time']
t_opt = s_opt.conditions['calculation_time']
t_dlm = s_dlm.conditions['calculation_time']
#%%
plt.ion()

#%%

_, [ax0, ax1] = plot_diff(s_ref, s_opt, 'transmittance_noslit', method='ratio',
          label1='Reference {1} lines ({0:.0f}s)'.format(t_ref, str_fmt(s_ref.conditions['lines_calculated'])),
          label2='Continuum + {1} lines ({0:.0f}s)'.format(t_opt, str_fmt(s_opt.conditions['lines_calculated'])),
          )#legendargs={'loc':'lower left', 'handlelength':1, 'fontsize':16.5}
ax0.set_ylim((0.87, 1))
ax1.set_ylim((0.999, 1.001))

handles, labels = ax0.get_legend_handles_labels()
leg = ax0.legend(frameon=False, fontsize=16,loc=3)
##leg.set_bbox_to_anchor((2190,0.55, 200, 0.1), transform=ax0.transData )


axins = ax0.inset_axes([0.60, 0.125, 0.35, 0.45])
axins.ticklabel_format(useOffset=False, axis='x')
axins.tick_params(which='major', labelsize=12)
axins.get_yaxis().set_visible(False)
axins.plot(*s_ref.get('transmittance_noslit', wunit='cm-1'), color='k', lw=2)
axins.plot(*s_opt.get('transmittance_noslit', wunit='cm-1'), color='r', lw=1)
##axins.set_xlim((1508.00, 1512))
axins.set_xlim((1473, 1483))

axins.set_ylim((0.935, 1.0))
ax0.indicate_inset_zoom(axins)

plt.savefig('output/Fig10a.pdf')






_, [ax0, ax1] = plot_diff(s_ref, s_opt, 'transmittance_noslit', method='ratio',
          label1='Reference {1} lines ({0:.0f}s)'.format(t_ref, str_fmt(s_ref.conditions['lines_calculated'])),
          label2='Continuum + {1} lines ({0:.0f}s)'.format(t_opt, str_fmt(s_opt.conditions['lines_calculated'])),
          )#legendargs={'loc':'lower left', 'handlelength':1, 'fontsize':16.5}
ax0.set_ylim((0.915, 1))
ax0.set_xlim((1473, 1483))


ax1.set_ylim((0.9994, 1.0006))

handles, labels = ax0.get_legend_handles_labels()
leg = ax0.get_legend()
leg.remove()
##leg.set_bbox_to_anchor((2190,0.55, 200, 0.1), transform=ax0.transData )


axins = ax0.inset_axes([0.12, 0.125, 0.5, 0.45])
axins.ticklabel_format(useOffset=False, axis='x')
axins.tick_params(which='major', labelsize=12)
axins.get_yaxis().set_visible(False)
axins.plot(*s_ref.get('transmittance_noslit', wunit='cm-1'), color='k', lw=2)
axins.plot(*s_opt.get('transmittance_noslit', wunit='cm-1'), color='r', lw=1)
##axins.set_xlim((1508.00, 1512))
axins.set_xlim((1480.5, 1480.9))

axins.set_ylim((0.935, 1.0))
ax0.indicate_inset_zoom(axins)

plt.savefig('output/Fig10b.pdf')




##
##_, [ax0, ax1] = plot_diff(s_ref, s_dlm, 'transmittance_noslit', method='ratio',
##          label1='Reference {1} lines ({0:.0f}s)'.format(t_ref, str_fmt(s_ref.conditions['lines_calculated'])),
##          label2='New method {1} lines ({0:.0f}s)'.format(t_dlm, str_fmt(s_dlm.conditions['lines_calculated'])),
##          )#legendargs={'loc':'lower left', 'handlelength':1, 'fontsize':16.5}
##ax0.set_ylim((0.881, 1))
##ax1.set_ylim((0.999, 1.001))
##
##handles, labels = ax0.get_legend_handles_labels()
##leg = ax0.legend(frameon=True, fontsize=16,loc=3)
##
##axins = ax0.inset_axes([0.53, -0.02, 0.45, 0.45])
##axins.ticklabel_format(useOffset=False, axis='x')
##axins.tick_params(which='major', labelsize=14)
##axins.get_yaxis().set_visible(False)
##axins.plot(*s_ref.get('transmittance_noslit', wunit='cm-1'), color='k', lw=2)
##axins.plot(*s_opt.get('transmittance_noslit', wunit='cm-1'), color='r', lw=1)
##axins.set_xlim((1508.00, 1512))
##ax0.indicate_inset_zoom(axins)
##
##if SAVE: plt.savefig('out/fig10_compare_H2O_ref_vs_optim_DLM.png')
##if SAVE: plt.savefig('out/fig10_compare_H2O_ref_vs_optim_DLM.pdf')


