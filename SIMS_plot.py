import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

bad_sample = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/TOF-SIMS/bad sample-depth profile-500eV.txt'
good_sample = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/TOF-SIMS/good sample-depth profile 500ev.txt'

# depthvalspd = pd.read_csv(bad_sample)
# print(depthvalspd.__dataframe__, columns = ['O+'])

depthvals = np.loadtxt(bad_sample, delimiter='\t', skiprows=0, usecols=0, dtype=float)
O_vals = np.loadtxt(bad_sample, delimiter='\t', skiprows=0, usecols=1, dtype=float)
Si_vals = np.loadtxt(bad_sample, delimiter='\t', skiprows=0, usecols=3, dtype=float)
AlO_vals = np.loadtxt(bad_sample, delimiter='\t', skiprows=0, usecols=5, dtype=float)
Ti_vals = np.loadtxt(bad_sample, delimiter='\t', skiprows=0, usecols=7, dtype=float)
Pt_vals = np.loadtxt(bad_sample, delimiter='\t', skiprows=0, usecols=9, dtype=float)

depthvals_g = np.loadtxt(good_sample, delimiter='\t', skiprows=0, usecols=0, dtype=float)
O_vals_g = np.loadtxt(good_sample, delimiter='\t', skiprows=0, usecols=1, dtype=float)
Si_vals_g = np.loadtxt(good_sample, delimiter='\t', skiprows=0, usecols=3, dtype=float)
AlO_vals_g = np.loadtxt(good_sample, delimiter='\t', skiprows=0, usecols=5, dtype=float)
Ti_vals_g = np.loadtxt(good_sample, delimiter='\t', skiprows=0, usecols=7, dtype=float)
Pt_vals_g = np.loadtxt(good_sample, delimiter='\t', skiprows=0, usecols=9, dtype=float)

plt.semilogy(depthvals, O_vals,linewidth = 0.5, label = 'O D1-R2-B')
# plt.semilogy(depthvals, Si_vals, linewidth = 0.5, label = 'Si D1-R2-B')
# plt.semilogy(depthvals, AlO_vals, linewidth = 0.5, label = 'AlO+ D1-R2-B')
plt.semilogy(depthvals, Ti_vals, linewidth = 0.5, label = 'Ti D1-R2-B')
# plt.semilogy(depthvals, Pt_vals, linewidth = 0.5, label = 'Pt D1-R2-B')

plt.semilogy(depthvals_g, O_vals_g, linewidth = 0.5, label = 'O FIB3')
# plt.semilogy(depthvals_g, Si_vals_g, linewidth = 0.5, label = 'Si FIB3')
# plt.semilogy(depthvals_g, AlO_vals_g, linewidth = 0.5, label = 'AlO+ FIB3')
plt.semilogy(depthvals_g, Ti_vals_g, linewidth = 0.5, label = 'Ti FIB3')
# plt.semilogy(depthvals_g, Pt_vals_g, linewidth = 0.5, label = 'Pt FIB3')

plt.title('TOF-SIMS results: TiO2 Comparison')
plt.ylabel('Intensity Counts (Background Corrected)')
plt.xlabel('Depth (nm)')
plt.xlim(0,250)
plt.legend(loc = 'upper right')
plt.show()
