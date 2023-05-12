import matplotlib.pyplot as plt
import numpy as np
import os

# set target directory for retention plot - directory should contain CSV files for one device
tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes/data/FIB3_S8_1_50_low_drift_LP6dB6dBHz_Integ1.0_retentiondata'
#tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes/Measurements 050123'

#colorlist = ('blue', 'red', 'orange', 'yellow', 'green', 'brown', 'indigo', 'black', 'violet', 'purple')
colorlist = ('orange', 'yellow', 'green', 'brown', 'blue', 'indigo', 'red', 'black', 'violet', 'purple')
reslist = (10200000.0, 12200000.0, 16100000.0, 23800000.0, 45300000.0, 477000000.0)
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
    plt.figure(dpi=1200)
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

    k = 0
    for k in data_master:
        avg_array = [] #placeholder for average of all measurements of a particular time step
        stdev_array = [] #stdev of above, for plotting the distribution
        j = 0
        while j < len(k[0]): # looping through j measurements for each time step
            temp_array = []
            for i in k: #looping through i time steps
                temp_array = temp_array + [i[j]]
            j += 1
            avg_array = avg_array + [np.median(temp_array)]
            stdev_array = stdev_array + [np.std(temp_array)]
        plt.semilogy(tvals, avg_array, color='black')
        plt.semilogy(tvals, avg_array, '.', color='black')
        plt.fill_between(tvals, np.subtract(avg_array,stdev_array), np.add(avg_array,stdev_array), alpha = 0.5)
        plt.title('FIB3_U8_3: Analog Weight Stability', weight='bold', fontsize = 14)
        plt.ylabel('Conductance value (S)', weight='bold', fontsize = 14)
        plt.xlabel('Time (s)', weight='bold', fontsize = 14)
    plt.show()

import_data()


