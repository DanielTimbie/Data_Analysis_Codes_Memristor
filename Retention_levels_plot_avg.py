import matplotlib.pyplot as plt
import numpy as np
import os

# set target directory for retention plot - directory should contain CSV files for one device
#tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes/data/FIB3_S8_1_50_low_drift_LP6dB6dBHz_Integ1.0_retentiondata'
tg_dir = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-06 6 state/six_state_standard' 
# tg_dir = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-19 8 state proportional measurements' 
#tg_dir = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/FIB3_S8_1_50_low_drift_LP6dB6dBHz_Integ1.0_retentiondata'

#colorlist = ('blue', 'red', 'orange', 'yellow', 'green', 'brown', 'indigo', 'black', 'violet', 'purple')
colorlist = ('orange', 'yellow', 'green', 'brown', 'blue', 'indigo', 'red', 'black', 'violet', 'purple')
colorlist = ('red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'black', 'purple')
colorlist = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf')

# reslist = (10200000.0, 12200000.0, 16100000.0, 23800000.0, 45300000.0, 477000000.0)
reslist = (24400000.0, 910000000.0)
reslist = (24400000.0, 30200000.0, 39900000.0, 58500000.0, 910000000.0, 110000000)
# reslist = (2.375e7, 2.713e7, 3.164e7, 3.794e7, 4.738e7, 6.307e7, 9.429e7, 1.867e8, 9.5e9)


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
    #plt.figure(dpi=1200)
    #data_master = [[],[],[],[],[],[]] #one for each state
    data_master = [[],[],[],[],[],[]]
    for i in dir_list:
        if i[-5:] == 'a.csv':
            for k in reslist:
                #print(k)
                #print('Loading file: ' + i)
                rvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=0, dtype=float)
                cvals = 1/rvals
                tvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=2, dtype=float)
                rmin = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=3, dtype=float)[1]
                rmax = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=4, dtype=float)[1]
                if rmin == k:
                    data_master[reslist.index(k)] = data_master[reslist.index(k)] + [cvals]

    for k in data_master:
        avg_array = [] #placeholder for average of all measurements of a particular time step
        stdev_array = [] #stdev of above, for plotting the distribution
        j = 0
        while j < len(k[0]): # looping through j measurements for each time step
            print(j)
            temp_array = []
            for i in k: #looping through i time steps
                temp_array = temp_array + [i[j]]
            j += 1
            avg_array = avg_array + [np.median(temp_array)]
            stdev_array = stdev_array + [np.std(temp_array)]
        plt.semilogy(tvals, avg_array, color='black')
        #plt.semilogy(tvals, avg_array, '.', color='black')
        plt.fill_between(tvals, np.subtract(avg_array,stdev_array), np.add(avg_array,stdev_array), alpha = 0.5)
        #plt.title('FIB3_M7_1: Analog Weight Stability - Denoise', weight='bold', fontsize = 14)
        plt.title('FIB3_I7_3 Analog Weight Spread', weight='bold', fontsize = 14)

        plt.ylabel('Conductance value (S)', weight='bold', fontsize = 14)
        plt.xlabel('Time (s)', weight='bold', fontsize = 14)
        plt.ylim(7e-10, 1e-7)
    plt.show()

import_data()


