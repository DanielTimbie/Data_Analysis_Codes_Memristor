import matplotlib.pyplot as plt
import numpy as np
import os


# set target directory for retention plot - directory should contain CSV files for one device
#tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes/data/FIB3_S8_1_50_low_drift_LP6dB6dBHz_Integ1.0_retentiondata'
# tg_dir = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-06 6 state/six_state_denoise_2' 
tg_dir = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-19 8 state proportional measurements' 

#colorlist = ('blue', 'red', 'orange', 'yellow', 'green', 'brown', 'indigo', 'black', 'violet', 'purple')
colorlist = ('orange', 'yellow', 'green', 'brown', 'blue', 'indigo', 'red', 'black', 'violet', 'purple')
# reslist = (10200000.0, 12200000.0, 16100000.0, 23800000.0, 45300000.0, 477000000.0)
reslist = (24400000.0, 910000000.0)
reslist = (24400000.0, 30200000.0, 39900000.0, 58500000.0, 910000000.0, 110000000)
reslistm = (25000000.0, 31200000.0, 41500000.0, 61200000.0, 123000000.0, 1000000000.0)
reslist = (2.375e7, 2.713e7, 3.164e7, 3.794e7, 4.738e7, 6.307e7, 9.429e7, 1.867e8, 9.5e9)
reslistm = (2.625e7, 3.000e7, 3.497e7, 4.193, 5.236e7, 6.790e7, 1.042e8, 2.063e8, 1.05e10)


chigh = 1/((reslist[0]+reslistm[0])/2)
clow = 1/((reslist[1]+reslistm[1])/2)

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
    #data_master = [[],[],[],[],[],[]]
    data_master = [[],[],[],[],[],[],[],[],[]]
    for i in dir_list:
        if i[-5:] == 'a.csv':
            for k in reslist:
                #print('Loading file: ' + i)
                rvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=0, dtype=float)
                cvals = 1/rvals
                tvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=2, dtype=float)
                rmin = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=3, dtype=float)[1]
                rmax = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=4, dtype=float)[1]
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
        #plt.plot(tvals, rmsre_array, '.', color='black')
        plt.title('FIB3_I7-3: Analog Weight Spread RMSE', weight='bold', fontsize = 14)
        plt.ylabel('Normalized RMS Error (%)', weight='bold', fontsize = 14)
        plt.xlabel('Time (s)', weight='bold', fontsize = 14)
        plt.legend()
        #plt.ylim(0,105)
        kk += 1
    plt.show()


        # pd.read_csv(i, header=None).T[1][1:][::-1]

import_data()


