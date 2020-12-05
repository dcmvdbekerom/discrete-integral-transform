import numpy as np
import matplotlib.pyplot as plt
from numpy import log


size = 14
vmin = 0.5
k   = np.arange(5)
ki  = 2
t   = 0.7
a   = t

Si = np.zeros(len(k))
Si[ki]   = (1-a)
Si[ki+1] =    a
plt.bar(k,Si,width=1.0,fc='cornsilk',ec='k')
ht  = 0.05
hdv = 0.19


plt.plot([-1,ki+0.5],[  a,  a],'k--')
plt.plot([-1,ki-0.5],[1-a,1-a],'k--')
plt.plot([-1,ki+t],[1,1],'k--')
plt.plot([ki+t,ki+t],[0,1],'r-')
plt.plot([ki+t,ki+t],[1.0,0.0],'ro')

plt.text(ki+0.5,hdv+0.02,'$\\Delta \\nu$',fontsize = size,ha='center',backgroundcolor='cornsilk')
plt.plot([ki+1,ki+1],[  0,  hdv],'k--')
plt.annotate(s='', xy=(ki,hdv), xytext=(ki+1,hdv), arrowprops=dict(arrowstyle='<->'))

plt.text(ki+0.5*t,ht+0.02,'$t_\\nu^i \\Delta \\nu$',fontsize = size,ha='center',backgroundcolor='cornsilk')
plt.plot([ki,ki],[  0,  max(ht,hdv)],'k--')
plt.annotate(s='', xy=(ki,ht), xytext=(ki+t,ht), arrowprops=dict(arrowstyle='<->'))



plt.plot([vmin,vmin,vmin,ki,ki+1],[1.0,1-a,a,0,0],'ko')
plt.xticks( [0,1,ki,ki+t,ki+1,4],
            ['','','$\\nu[k^i]$','$\\nu_0^i$','$\\nu[k^i+1]$',''],
            fontsize = size,
            ha='left')
plt.yticks( [0,(1-a),a,1.0],['$0$','$(1-a_\\nu^i)$','$a_\\nu^i$','$1$'],fontsize = size)

plt.xlim(vmin,4.5)
plt.tight_layout()
plt.savefig('output/Fig2.png')
plt.savefig('output/Fig2.pdf')

plt.show()
