import numpy as np
import matplotlib.pyplot as plt
import sys
import voigtlib as vl

x_max = 100.0
dx = 0.01
dt = 0.1

t_arr = np.arange(0,1,dt)
x    = np.arange(0,x_max,dx)
x_FT = np.arange(x.size + 1) / (2 * x.size * dx)
dksi = 1 / (2 * x_max)

d_arr = np.array([-0.99] + list(np.arange(-0.95,1.0,0.025)) + [0.99])
alpha_arr = (1+d_arr)/(1-d_arr)

dxv = 0.1
dxG = 0.14
dxL = 0.2

Erms_sim = []
Erms_opt = []
Erms_w1  = []
Erms_w2  = []
Erms_liu = []

IG = vl.GG(x)
IL = vl.GL(x)

Iav0_FT, Iav1_FT, IaG0_FT, IaG1_FT, IaL0_FT, IaL1_FT = {}, {}, {}, {}, {}, {} 

for alpha in alpha_arr:
    
    d = (alpha - 1)/(alpha + 1)
    print(d)
    
    alphaG = vl.f_alphaG(alpha)
    alphaL = vl.f_alphaL(alpha)

    for tv in t_arr:
        Iav0_FT[tv] = np.exp(-2j*np.pi* tv   *dxv*x_FT)
        Iav1_FT[tv] = np.exp(-2j*np.pi*(tv-1)*dxv*x_FT)

    for tG in t_arr:    
        IaG0_FT[tG] = vl.GG_FT(alphaG*x_FT*np.exp(  -tG *dxG))
        IaG1_FT[tG] = vl.GG_FT(alphaG*x_FT*np.exp((1-tG)*dxG))

    for tL in t_arr:
        IaL0_FT[tL] = vl.GL_FT(alphaL*x_FT*np.exp(  -tL *dxL))
        IaL1_FT[tL] = vl.GL_FT(alphaL*x_FT*np.exp((1-tL)*dxL)) 
    
    # Calc exact lineshape:
    Iex_FT = vl.exact_lineshape_FT(x_FT,alpha)
    Iex = np.fft.irfft(Iex_FT)[:x.size]/dx

    E_sim = 0
    E_opt = 0
    
    # Iterate over alignments:
    for tv in t_arr:
        for tG in t_arr:
            for tL in t_arr:

                basis = Iav0_FT[tv], Iav1_FT[tv], IaG0_FT[tG], IaG1_FT[tG], IaL0_FT[tL], IaL1_FT[tL]    

                # Simple weights:
                av,aG,aL = tv,tG,tL
                Ia_sim_FT = vl.approximate_lineshape_FT(av,aG,aL,basis)
                E_sim += 2*np.sum(np.abs(Ia_sim_FT - Iex_FT)**2)*dksi*dt**3

                # Optimized weights:
                dxvG = dxv/alphaG
                av,aG,aL = vl.optimized_weights(tv,tG,tL,dxvG,dxG,dxL,alpha)
                Ia_opt_FT = vl.approximate_lineshape_FT(av,aG,aL,basis)
                E_opt += 2*np.sum(np.abs(Ia_opt_FT - Iex_FT)**2)*dksi*dt**3
                
    Erms_sim.append(E_sim**0.5)
    Erms_opt.append(E_opt**0.5)

    # Literature lineshapes

    Iw0 = 1/(1.065+0.447*alphaL + 0.058*alphaL**2)
    Iw1 = ((1 - alphaL) * np.exp(-2.772 * x**2) + alphaL * 1 / (1 + 4 * x**2)) * Iw0
    Iw2 = Iw1 + Iw0*0.016*(1-alphaL)*alphaL*(np.exp(-0.4*x**2.25)-10/(10+x**2.25))

    E_w1 = 2*np.sum((Iw1 - Iex)**2)*dx
    E_w2 = 2*np.sum((Iw2 - Iex)**2)*dx

    Erms_w1.append(E_w1**0.5)
    Erms_w2.append(E_w2**0.5)

    c_liu_G = 0.32460 - 0.61825*d + 0.17681*d**2 + 0.12109*d**3
    c_liu_L = 0.68188 + 0.61293*d - 0.18384*d**2 - 0.11568*d**3        
    Iliu = c_liu_G * IG + c_liu_L * IL
    E_liu = 2*np.sum((Iliu - Iex)**2)*dx
    Erms_liu.append(E_liu**0.5)



plt.plot(d_arr,Erms_w1,label = 'Whiting 1')
plt.plot(d_arr,Erms_w2,label = 'Whiting 2')
plt.plot(d_arr,Erms_liu,label = 'Liu et al.')
plt.plot(d_arr,Erms_sim,label = 'Simple\n({:.1f},{:.2f},{:.1f})'.format(dxv,dxG,dxL))
plt.plot(d_arr,Erms_opt,label = 'Optimized\n({:.1f},{:.2f},{:.1f}) '.format(dxv,dxG,dxL))


plt.grid(True)
plt.axhline(0,c='k')
plt.axvline(-1,ls='--',c='gray')
plt.axvline(1,ls='--',c='gray')
plt.xlim(-1.1,1.1)

plt.ylabel('RMS-error',fontsize = 12)
plt.xlabel('$Gaussian\\quad\\quad\\quad\\quad \\leftarrow \\quad\\quad\\quad\\quad\\quad d=\\frac{w_L-w_G}{w_L+w_G} \\quad\\quad\\quad\\quad\\quad \\rightarrow \\quad\\quad\\quad\\quad Lorentzian$',fontsize = 12)


## Plot approximated errors:

E_est_sim = vl.average_error_simple(dxv,dxG,dxL,alpha_arr,ksi_max = 100)
plt.plot(d_arr,E_est_sim,'k--')

E_est_opt = vl.average_error_optimized(dxv,dxG,dxL,alpha_arr,ksi_max = 100)
plt.plot(d_arr,E_est_opt,'k--')
plt.subplots_adjust(left=0.15,bottom = 0.15)
plt.legend(fontsize=12)
plt.savefig('output/Fig4.pdf')
plt.savefig('output/Fig4.png')
plt.show()

sys.exit()






