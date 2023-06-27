import numpy as np
import matplotlib.pyplot as plt
from fourier_stuff import StandardFourier, HanningApply

file1 = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-12 noise/FIB3_M7_3_hrs_on_0_low_drift_LP6dB6dBHz_Integ0.2_retentiondata.csv'
file2 = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-12 noise/FIB3_M7_3_lrs_on_1_low_drift_LP6dB6dBHz_Integ0.2_retentiondata.csv'
file3 = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-12 noise/FIB3_M7_3_gnd_2_low_drift_LP6dB6dBHz_Integ0.2_retentiondata.csv'
file4 = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-12 noise/FIB3_M7_3_probe_up_3_low_drift_LP6dB6dBHz_Integ0.2_retentiondata.csv'
file4 = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-12 noise/FIB3_M7_3_probe_up_3_low_drift_LP6dB6dBHz_Integ0.2_retentiondata.csv'
# file4 = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-12 noise/FIB3_M7_3_gnd_4_low_drift_LP6dB6dBHz_Integ0.2_retentiondata.csv'



curvals = np.loadtxt(file1, delimiter=',', skiprows=1, usecols=1, dtype=float)
#tval = np.loadtxt(tg_dir, delimiter=',', skiprows=1, usecols=2, dtype=float)[3] - np.loadtxt(tg_dir, delimiter=',', skiprows=1, usecols=2, dtype=float)[2]
tval = 0.005 #seconds
# cvals = 1/rvals
fr = 1/tval

curvals2 = np.loadtxt(file2, delimiter=',', skiprows=1, usecols=1, dtype=float)
curvals3 = np.loadtxt(file3, delimiter=',', skiprows=1, usecols=1, dtype=float)
curvals4 = np.loadtxt(file4, delimiter=',', skiprows=1, usecols=1, dtype=float)
# curvals5 = np.loadtxt(file2, delimiter=',', skiprows=1, usecols=1, dtype=float)


data = StandardFourier(curvals,fr)
data2 = StandardFourier(curvals2,fr)
data3 = StandardFourier(curvals3,fr)
data4 = StandardFourier(curvals4,fr)
data = StandardFourier(HanningApply(curvals),fr)
data2 = StandardFourier(HanningApply(curvals2),fr)
data3 = StandardFourier(HanningApply(curvals3),fr)
data4 = StandardFourier(HanningApply(curvals4),fr)


# data_new = StandardFourier(HanningApply(curvals), fr)

# mean_cur = np.mean(curvals)
# print(mean_cur)
# curvals_meansub = np.subtract(curvals, mean_cur)
# data_new_new = StandardFourier(HanningApply(curvals_meansub), fr)

# plt.loglog(data[0], data[1], alpha = 0.8, linewidth = 0.75, label = 'hrs')
# plt.loglog(data_new[0], data_new[1], alpha = 0.8, linewidth = 0.75, label = 'hrs, hanning window')
# plt.loglog(data_new_new[0], data_new_new[1], alpha = 0.8, linestyle = 'dashed', linewidth = 0.5, label = 'hrs, hanning window, mean subtracted', color = 'red')

# plt.loglog(data[0], data[1], alpha = 0.8, linewidth = 0.75, label = 'hrs')
# plt.loglog(data2[0], data2[1], alpha = 0.8, linewidth = 0.75, label = 'hrs')
# plt.loglog(data3[0], data3[1], alpha = 0.8, linewidth = 0.75, label = 'gnd', color = 'green')
# plt.loglog(data4[0], data4[1], alpha = 0.8, linewidth = 0.75, label = 'probe up', color = 'red')

plt.loglog(data[0], data[1], alpha = 0.8, linewidth = 0.75, label = 'lrs')
plt.loglog(data2[0], data2[1], alpha = 0.8, linewidth = 0.75, label = 'hrs')
plt.loglog(data3[0], data3[1], alpha = 0.8, linewidth = 0.75, label = 'gnd', color = 'green')
plt.loglog(data4[0], data4[1], alpha = 0.8, linewidth = 0.75, label = 'probe up', color = 'red')


# plt.title('FIB3_W8_3: Noise Spectra HRS -0.1V')
# plt.title('FIB3_W8_3: Noise Spectra HRS 0V')
plt.title('FIB3_W8_3: Co-plotted lrs, hrs, gnd, probe up noise')
plt.ylabel('')
plt.xlabel('Frequency (Hz)')
#plt.xlim(-0.1,1)
plt.legend()
plt.show()

print('success')


#need to remember that this is not the spectral density, but the FT of the data. need to convert to spectral density to compare the two, or PSD. 