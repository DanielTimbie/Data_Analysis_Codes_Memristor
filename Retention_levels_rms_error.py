import matplotlib.pyplot as plt
import numpy as np
import os


# set target directory for retention plot - directory should contain CSV files for one device
tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes/data/FIB3_S8_1_50_low_drift_LP6dB6dBHz_Integ1.0_retentiondata'
#tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes/Measurements 050123'

#colorlist = ('blue', 'red', 'orange', 'yellow', 'green', 'brown', 'indigo', 'black', 'violet', 'purple')
colorlist = ('orange', 'yellow', 'green', 'brown', 'blue', 'indigo', 'red', 'black', 'violet', 'purple')
reslist = (10200000.0, 12200000.0, 16100000.0, 23800000.0, 45300000.0, 477000000.0)
reslistm = (10400000.0, 12500000.0, 16700000.0, 25000000, 49800000.0, 10000000000)
chigh = 1/((reslist[0]+reslistm[0])/2)
clow = 1/((reslist[5]+reslistm[5])/2)

def color_plotter(input,resvals):
    len_vals = len(resvals)
    k = 0
    while k < len_vals:
        if input == resvals[k]:
            return colorlist[k]
        k += 1

# import data files - format should be CSV with info about resistance value, time, and target values
def import_data():
    path = tg_dir
    dir_list = os.listdir(path)
    plt.figure(dpi=120)
    data_master = [[],[],[],[],[],[]]
    for i in dir_list:
        if i[-5:] == 'a.csv':
            for k in reslist:
                #print('Loading file: ' + i)
                rvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=0, dtype=float)
                cvals = 1/rvals
                tvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=1, dtype=float)
                rmin = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=2, dtype=float)[1]
                rmax = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=3, dtype=float)[1]
                if rmin == k:
                    data_master[reslist.index(k)] = data_master[reslist.index(k)] + [cvals]

    kk = 0
    for k in data_master:
        target = 1/reslist[kk]  #(1/reslist[kk] + 1/reslistm[kk])/2
        print(target)
        rmsre_array = []
        j = 0
        while j < len(k[0]): # looping through j measurements for each time step
            temp_array = []
            ms_array = []
            for i in k: #looping through i time steps
                temp_array = temp_array + [i[j]]
                ms_array = ms_array + [(i[j] - target)**2]
            j += 1
            #rmsre_array = rmsre_array + [(100/len(ms_array))*np.sqrt(np.sum(ms_array))/(len(ms_array)*target)]
            rmsre_array = rmsre_array + [100/7*np.sqrt(np.sum(ms_array)) / (chigh-clow)]


        plt.plot(tvals, rmsre_array, label=str(round(target, 11)))
        plt.plot(tvals, rmsre_array, '.', color='black')
        plt.title('FIB3_U8_3: Analog Weight Spread', weight='bold', fontsize = 14)
        plt.ylabel('Normalized RMS Error (%)', weight='bold', fontsize = 14)
        plt.xlabel('Time (s)', weight='bold', fontsize = 14)
        plt.legend()
        kk += 1
    plt.show()


        # pd.read_csv(i, header=None).T[1][1:][::-1]

import_data()


