import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Benchmark results saved separately, this script only handles the plotting part.
df = pd.read_excel('benchmark_output.xlsx')

N = df['N'].values
t_init = df['t_init (s)'].values
t_iter = df['t_iter (s)'].values

plt.plot(N,t_init,'ko--',label='First spectrum')
plt.plot(N,t_iter,'ro--',label='Subsequent spectra')

plt.xlabel('Number of lines',fontsize=12)
plt.ylabel('Calculation Time (s)',fontsize=12)

plt.xscale('log')
plt.yscale('log')

plt.grid(True)
plt.legend(loc=2,fontsize=12)

plt.savefig('output/Fig11.png')
plt.savefig('output/Fig11.pdf')

plt.show()
