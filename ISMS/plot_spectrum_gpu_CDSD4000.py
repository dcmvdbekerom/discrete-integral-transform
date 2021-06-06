from radis import calc_spectrum
import radis.lbl.gpu as gpu
from radis.lbl.gpu import gpu_init, gpu_iterate
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import cupy as cp
from PIL import Image
import py_cuffs

dv = 0.002 #cm-1
v_arr = np.arange(1800.0,2400.0,dv)
NwG = 4
NwL = 8
N = 0
path = 'C:/CDSD4000/npy/'
print('Loading...')
t0 = perf_counter()
Q_arr = np.ones(4,dtype=np.float32)
iso = np.load(path+'iso.npy', mmap_mode = 'r')[-N:].astype(np.uint8)
v0 = np.load(path+'v0.npy', mmap_mode = 'r')[-N:]
da = np.load(path+'da.npy', mmap_mode = 'r')[-N:]
El = np.load(path+'El.npy', mmap_mode = 'r')[-N:]
na = np.load(path+'na.npy', mmap_mode = 'r')[-N:]
S0 = np.load(path+'S0.npy', mmap_mode = 'r')[-N:]
log_2gs = np.load(path+'log_2gs.npy', mmap_mode = 'r')[-N:]
log_2vMm = np.load(path+'log_2vMm.npy', mmap_mode = 'r')[-N:]
t0 = perf_counter() - t0
n_lines = len(v0)*1e-6

print('Done!')
print('Loaded {:.1f}M lines in {:.2f} s'.format(n_lines,t0))
print('Initializing...')

##n_lines = int(2.3E8)
##py_cuffs.set_path('C:/CDSD4000/npy/')
##py_cuffs.set_N_lines(n_lines)
##py_cuffs.init(v_arr,NwG,NwL)




gpu_init(
    v_arr,
    NwG,
    NwL,
    iso,
    v0,
    da,
    log_2gs,
    na,
    log_2vMm,
    S0,
    El,
    Q_arr,
    verbose_gpu=False,
    gpu=True,
)
print('Done!')
p = 1.0
T = 1100.0
x = 1.0

abs_coeff, transmittance = gpu_iterate(p,T,x,verbose_gpu=False,gpu=True)
##abs_coeff = py_cuffs.iterate(p, T)
abs_coeff /= np.max(abs_coeff)
fig = plt.figure(figsize=(12,3))
p1, = plt.plot(v_arr,abs_coeff,lw=1)
ax = plt.gca()
plt.xlim(2400,1800)
plt.axhline(0,c='k',alpha=0.5)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.23)
plt.xlabel('$\\nu$ (cm$^{-1}$)',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
axins = plt.gca().inset_axes([0.53, 0.35, 0.46, 0.58])
##axins.tick_params(axis='x', labelsize=10)
axins.get_xaxis().set_visible(False)
axins.get_yaxis().set_visible(False)
axins.set_xlim(2330, 2320)
axins.set_ylim(0,1)
ax.indicate_inset_zoom(axins)
p2,=axins.plot(v_arr,abs_coeff,lw=1)

def update(val):
    T = val
    t0 = perf_counter()
    abs_coeff, transmittance = gpu_iterate(p,T,x,verbose_gpu=False,gpu=True)
##    abs_coeff = py_cuffs.iterate(p, T)
    t0 = perf_counter() - t0
    abs_coeff /= np.max(abs_coeff)
    p1.set_ydata(abs_coeff)
    p2.set_ydata(abs_coeff)
    ax.set_title('T = {:.0f} K ({:.1f} M lines, {:.0f} ms)'.format(T,n_lines,1e3*t0),fontsize=16)
    plt.gcf().canvas.draw_idle()

T_arr = np.linspace(300,4000,100)

gif_images = []
for T in T_arr:
    print(T)
##    update(T)
    fname = 'output/CDSD_{:.1f}K.png'.format(T)
##    plt.savefig(fname,dpi=100.0)
    pImage=Image.open(fname)
    gif_images.append(pImage.convert('RGB').convert('P', palette=Image.ADAPTIVE))


gif_images += gif_images[::-1]
gif_images[0].save('CDSD4000.gif',
                    save_all=True,
                    duration = 50,
                    loop = 0,
                    append_images=gif_images[1:])


cp._default_memory_pool.free_all_blocks()

