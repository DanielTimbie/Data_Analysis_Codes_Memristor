import matplotlib.pyplot as plt
import numpy as np
import os

# set target directory for retention plot - directory should contain CSV files for one device
tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes/data/FIB3_S8_1_50_low_drift_LP6dB6dBHz_Integ1.0_retentiondata'
#tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes/Measurements 050123'

# import data files - format should be CSV with info about resistance value, time, and target values
def import_data():
    path = tg_dir
    dir_list = os.listdir(path)
    files = []
    for i in dir_list:
        if i[-5:] == '0.csv':
            #print('Loading file: ' + i)
            if i[11] == '_':
                files = files + [int(i[10:11])]
            else:
                files = files + [int(i[10:12])]
    files.sort()
    Ivals_total = []

    plt.figure(dpi=1200)
    for i in files:

        Ivals = np.loadtxt(tg_dir + '/FIB3_S8_1_' + str(i) + '_low_drift_LP6dB6dBHz_Integ1.0.csv', delimiter=',', skiprows=1, usecols=7, dtype=float)
        Vtargmin = -np.loadtxt(tg_dir + '/FIB3_S8_1_' + str(i) + '_low_drift_LP6dB6dBHz_Integ1.0_retentiondata.csv', delimiter=',', skiprows=1, usecols=2, dtype=float)[0]
        Vtargmax = -np.loadtxt(tg_dir + '/FIB3_S8_1_' + str(i) + '_low_drift_LP6dB6dBHz_Integ1.0_retentiondata.csv', delimiter=',', skiprows=1, usecols=3, dtype=float)[0]
        meas_num = len(Ivals_total)
        cvals = 1e9*Ivals/1
        Ivals_total = np.concatenate((Ivals_total,Ivals),axis=0)
        if i%2 ==1:
            plt.plot(np.arange(meas_num, meas_num + len(Ivals)),cvals, color = 'tab:blue')
        else:
            plt.plot(np.arange(meas_num, meas_num + len(Ivals)), cvals, color='tab:red')
        plt.axhspan(1e8/Vtargmin,1e8/Vtargmax, color = 'mistyrose')
        # plt.plot(np.arange(meas_num, meas_num + len(Ivals)), np.full(len(Ivals),1e9*0.1/ Vtargmin), color='tab:orange')
        # plt.plot(np.arange(meas_num, meas_num + len(Ivals)), np.full(len(Ivals), 1e9*0.1/ Vtargmax), color='tab:orange')

    plt.title('FIB3_U8_3: Switching Between Six States', weight='bold', fontsize = 14)
    plt.ylabel('Current through device (nA)', weight='bold', fontsize = 14)
    plt.xlabel('Measurement Number', weight='bold', fontsize = 14)
    #plt.legend()
    plt.show()
    #plt.savefig('output.png', dpi = 300)

import_data()



