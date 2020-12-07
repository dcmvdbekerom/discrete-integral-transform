import numpy as np
import matplotlib.pyplot as plt
import voigtlib as vl

markersize = 7
NN = 50
Nlevels = 10
fontsize = 14

dx = 0.01
x_max = 10.0
x = np.arange(0,x_max,dx)

dxv  = 0.05
dxG  = 0.14
dxL  = 0.2

tv = 0.5

av = tv
tG_arr = np.array([0.1,0.5,0.9])
tL_arr = np.array([0.1,0.5,0.9])

alpha_arr = np.array([1/25,1/2,2,25])
labels = ['1/25','1/2','2','25']
for alpha,label in zip(alpha_arr,labels):
    print(alpha)
    dxvG = dxv/vl.f_alphaG(alpha)

    error_integrals = vl.error_integrals(alpha)
    Ie_FT = vl.exact_lineshape_FT(x,alpha)

    fig,axes = plt.subplots(3,3,figsize=(9,8))
    plt.subplots_adjust(wspace = 0.35)
    axes[0,1].set_title('$\\alpha={:s}$\n'.format(label),fontsize=16)

    for tG,ax in zip(tG_arr,axes[-1,:]):
        ax.set_xlabel('$a_G$\n$t_G={:.1f}$'.format(tG),fontsize=14)
        
    for tL,ax in zip(tL_arr,axes[::-1,0]):
        ax.set_ylabel('$t_L={:.1f}$\n$a_L$'.format(tL),fontsize=14)
  
    for tL,axh in zip(tL_arr,axes[::-1]):
        for tG,axhv in zip(tG_arr,axh):
            print(tG,tL)
            basis = vl.lineshape_basis_FT(x,tv,tG,tL,dxv,dxG,dxL,alpha)

            av,aG,aL = vl.optimized_weights(tv,tG,tL,dxvG,dxG,dxL,alpha)

            axhv.plot([tG],[tL],'k+',fillstyle='full',markersize=markersize)
            axhv.plot([tG],[tL],'ks',fillstyle='none',markersize=markersize)
            axhv.plot([aG],[aL],'k.',fillstyle='full',markersize=markersize)
            axhv.plot([aG],[aL],'ko',fillstyle='none',markersize=markersize)

            # Generate grid:
            G_width = 0.5 * np.abs(aG - tG)
            L_width = 0.5 * np.abs(aL - tL)
            
            awG_arr = np.linspace(min(aG,tG) - G_width, max(aG,tG) + G_width, NN)
            awL_arr = np.linspace(min(aL,tL) - L_width, max(aL,tL) + L_width, NN)
            X, Y = np.meshgrid(awG_arr,awL_arr)

            # Numerical integration:
            Z = np.zeros((awL_arr.size,awG_arr.size))
            for i in range(awG_arr.size):
                for j in range(awL_arr.size):
                     Ia_FT = vl.approximate_lineshape_FT(av,awG_arr[i],awL_arr[j],basis)
                     Z[j,i] = np.sqrt(2*np.sum(np.abs(Ie_FT - Ia_FT)**2)*dx)

            CS = axhv.contour(X,Y,Z,levels = Nlevels)

            # Analytic approximation:
            Z2 = vl.approximate_error(av,X,Y,tv,tG,tL,dxv,dxG,dxL,error_integrals)
            CS2 = axhv.contour(X,Y,Z2,levels = Nlevels,colors='k',linestyles='dashed')

    plt.savefig('output/Fig3_alpha={:.3f}.pdf'.format(alpha,tv,alpha))
    plt.savefig('output/Fig3_alpha={:.3f}.png'.format(alpha,tv,alpha))
    plt.show()
##    plt.clf()
