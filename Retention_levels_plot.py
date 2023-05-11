import matplotlib.pyplot as plt
import numpy as np
import os

# set target directory for retention plot - directory should contain CSV files for one device
tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes//FIB3_S8_1_50_low_drift_LP6dB6dBHz_Integ1.0_retentiondata'
#tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes/Measurements 050123'

#colorlist = ('blue', 'red', 'orange', 'yellow', 'green', 'brown', 'indigo', 'black', 'violet', 'purple')
colorlist = ('orange', 'yellow', 'green', 'brown', 'blue', 'indigo', 'red', 'black', 'violet', 'purple')
reslist = (12200000.0, 45300000.0, 16100000.0, 10200000.0, 23800000.0, 477000000.0, 500000000)

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
    for i in dir_list:
        if i[-5:] == 'a.csv':
            print('Loading file: ' + i)
            rvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=0, dtype=float)
            cvals = 1/rvals
            tvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=1, dtype=float)
            rmin = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=2, dtype=float)[1]
            rmax = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=3, dtype=float)[1]
            str_minmax = str("{:e}".format(rmin)) + ' Ohms to ' + str("{:e}".format(rmax)) + ' Ohms'
            # plt.plot(tvals,cvals,linestyle = 'dashed', label = str_minmax)
            #plt.plot(tvals, cvals, linestyle='dashed', color = color_plotter(rmin, reslist), label=str_minmax)
            #plt.plot(tvals, np.log10(cvals), label=str_minmax)
            plt.plot(tvals, np.log10(cvals), linestyle='dashed', color = color_plotter(rmin, reslist), label=str_minmax)
            plt.plot(tvals, np.log10(cvals), '.', color='black')
            #plt.plot(tvals, cvals, '.', color='black')
            #plt.axhspan(1/rmin, 1/rmax, color = 'mistyrose')
            plt.axhspan(np.log10(1/rmin), np.log10(1/rmax), color = 'mistyrose')
            #plt.title('FIB3_S8_1: time drift of ' + str(1/rmin)[:5] + str(1/rmin)[-4:] + ' to ' + str(1/rmax)[:5]+ str(1/rmax)[-4:]+ 'S') #
            #plt.ylim(1/rmin + 10*abs(1/rmin - 1/rmax), 1/rmax - 10*abs(1/rmin - 1/rmax))

    #plt.ylabel('Resistance value (Ohms)')
    #plt.title('FIB3_S8_1: time drift of six states between 1e-7 and 1e-10')
    plt.title('FIB3_U8_3: time drift of six states', weight='bold', fontsize = 14)
    plt.ylabel('Log Conductance value (S)', weight='bold', fontsize = 14)
    plt.xlabel('Time (s)', weight='bold', fontsize = 14)
    #plt.xlim(0,800)
    #plt.legend()
    plt.ylim(-10.1,-7)
    plt.show()



        # pd.read_csv(i, header=None).T[1][1:][::-1]

import_data()


