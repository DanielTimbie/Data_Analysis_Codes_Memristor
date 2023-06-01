import numpy as np
import matplotlib.pyplot as plt
from fourier_stuff import StandardFourier

tg_dir = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-05-24/W8_3_run4/FIB3_W8_3_noGNDprobe_5_low_drift_LP6dB6dBHz_Integ0.2_retentiondata.csv'

curvals = np.loadtxt(tg_dir, delimiter=',', skiprows=1, usecols=1, dtype=float)
#tval = np.loadtxt(tg_dir, delimiter=',', skiprows=1, usecols=2, dtype=float)[3] - np.loadtxt(tg_dir, delimiter=',', skiprows=1, usecols=2, dtype=float)[2]
tval = 0.005 #seconds
# cvals = 1/rvals
fr = 1/tval

data = StandardFourier(curvals,fr)

plt.loglog(data[0], data[1])
# plt.title('FIB3_W8_3: Noise Spectra HRS -0.1V')
# plt.title('FIB3_W8_3: Noise Spectra HRS 0V')
plt.title('FIB3_W8_3: Noise Spectra HRS No GND')
plt.ylabel('')
plt.xlabel('Frequency (Hz)')
#plt.xlim(-0.1,1)
plt.show()

print('success')
