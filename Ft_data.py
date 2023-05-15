import numpy as np
import matplotlib.pyplot as plt
from fourier_stuff import StandardFourier

tg_dir = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/Measurements 050123/FIB3_U8_3_noise_7_low_drift_LP6dB6dBHz_Integ1_retentiondata.csv'

rvals = np.loadtxt(tg_dir, delimiter=',', skiprows=1, usecols=0, dtype=float)
#tval = np.loadtxt(tg_dir, delimiter=',', skiprows=1, usecols=2, dtype=float)[3] - np.loadtxt(tg_dir, delimiter=',', skiprows=1, usecols=2, dtype=float)[2]
tval = 0.025 #seconds
cvals = 1/rvals
fr = 1/tval

data = StandardFourier(cvals,fr)

plt.loglog(data[0], data[1])
plt.title('FIB3_U8_3: Noise Spectra')
plt.ylabel('')
plt.xlabel('Frequency (Hz)')
#plt.xlim(-0.1,1)
plt.show()



