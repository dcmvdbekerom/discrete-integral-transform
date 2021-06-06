import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy import log
from matplotlib.widgets import Slider
from PIL import Image

size = 14
vmin = 0.5
k   = np.arange(5)
kt = 2.2
ki = int(kt)
t = kt-ki
a = t

Si = np.zeros(len(k))
Si[ki]   = (1-a)
Si[ki+1] =    a
ht  = 0.05
hdv = 0.19



p0  = plt.bar(k,Si,width=1.0,fc='cornsilk',ec='k')
p1, = plt.plot([ki+t,ki+t],[0,1],'r-')
p2, = plt.plot([ki+t,ki+t],[1.0,0.0],'ro')
p3, = plt.plot([-1,ki+0.5],[  a,  a],'k--')
p4, = plt.plot([-1,ki-0.5],[1-a,1-a],'k--')
p5, = plt.plot([-1,ki+t],[1,1],'k--')
p6, = plt.plot([vmin,vmin,vmin,ki,ki+1],[1.0,1-a,a,0,0],'ko')

p7, = plt.plot([ki+1,ki+1],[  0,  hdv],'k--')
p8, = plt.plot([ki,ki],[  0,  max(ht,hdv)],'k--')

t1 = plt.text(ki+0.5,hdv+0.02,'$\\Delta \\nu$',fontsize = size,ha='center',backgroundcolor='cornsilk')
a1 = plt.annotate(text='', xy=(ki,hdv), xytext=(ki+1,hdv), arrowprops=dict(arrowstyle='<->'))

t2 = plt.text(ki+0.5*t,ht+0.02,'$t\\Delta \\nu$',fontsize = size,ha='center',backgroundcolor='cornsilk')
a2 = plt.annotate(text='', xy=(ki,ht), xytext=(ki+t,ht), arrowprops=dict(arrowstyle='<->'))


plt.yticks( [0,(1-a),a,1.0],['$0$','$(1-t)$','$t$','$1$'],fontsize = size)

plt.xticks( [0,1,ki,ki+t,ki+1,4],
            ['','','$\\nu[k]$','\n$\\nu_0$','$\\nu[k+1]$',''],
            fontsize = size,
            ha='left')
ax=  plt.gca()
plt.xlim(vmin,4.5)
##plt.subplots_adjust(left=0.15,right=0.9,top=0.9,bottom=0.3)
##
##axslider = plt.axes([0.15, 0.1, 0.70, 0.03], facecolor='cornsilk')
##t_slider = Slider(
##    ax=axslider,
##    label='$\\nu_0$',
##    valmin=1.01,
##    valmax=3.99,
##    valinit=0.5,
##)
plt.sca(ax)

def update(val):
    global a1,a2
##    kt = t_slider.val
    kt = val
    ki = int(kt)
    t = kt-ki
    a = t

    for kk in k:
        p0[kk].set_height(0)
        
    p0[ki].set_height(1-a)
    p0[ki+1].set_height(a)
    p1.set_xdata([ki+t,ki+t])
    p2.set_xdata([ki+t,ki+t])
    p3.set_xdata([-1,ki+0.5])
    p3.set_ydata([a,a])
    p4.set_xdata([-1,ki-0.5])
    p4.set_ydata([1-a,1-a])
    p5.set_xdata([0,ki+t])
    p6.set_xdata([vmin,vmin,vmin,ki,ki+1])
    p6.set_ydata([1.0,1-a,a,0,0])
    p7.set_xdata([ki+1,ki+1])
    p8.set_xdata([ki,ki])
    t1.set_x(ki+0.5)
    t2.set_x(ki+0.5*t)

    a1.remove()
    a1 = plt.annotate(text='', xy=(ki,hdv), xytext=(ki+1,hdv), arrowprops=dict(arrowstyle='<->'))
    a2.remove()
    a2 = plt.annotate(text='', xy=(ki,ht), xytext=(ki+t,ht), arrowprops=dict(arrowstyle='<->'))

    i_list = [i for i in range(5)]
    x_list =  i_list[:ki+1]+[ki+t]+i_list[ki+1:]
    s_list = ki*['']+['$\\nu[k]$','\n$\\nu_0$','$\\nu[k+1]$']+(3-ki)*['']

    plt.xticks(x_list,s_list,
            fontsize = size,
            ha='left')
    plt.yticks( [0,(1-a),a,1.0],['$0$','$(1-t)$','$t$','$1$'],fontsize = size)

    plt.xlim(vmin,4.5)
    plt.gcf().canvas.draw_idle()
    
##t_slider.on_changed(update)

gif_images = []
for kt in np.linspace(1.01,3.99,200):
    print(kt)
    update(kt)
    fname = 'output/pos_weight_{:.3f}.png'.format(kt)
    plt.savefig(fname,dpi=100.0)
    pImage=Image.open(fname)
    gif_images.append(pImage.convert('RGB').convert('P', palette=Image.ADAPTIVE))

gif_images += gif_images[::-1]
gif_images[0].save('pos_weight.gif',
                    save_all=True,
                    duration = 100,
                    loop = 0,
                    append_images=gif_images[1:])
