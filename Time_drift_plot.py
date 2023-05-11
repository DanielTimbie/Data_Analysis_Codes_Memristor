import matplotlib.pyplot as plt
import numpy as np
import os

# set target directory for retention plot - directory should contain CSV files for one device
tg_dir = '/Users/GuestUser/Documents/Labwork/Memristor/Data Analysis Codes/Measurements 042823'


# import data files - format should be CSV with info about resistance value, time, and target values
def import_data():
    path = tg_dir
    dir_list = os.listdir(path)
    counter = 0
    data_arr = []
    for i in dir_list:
        if i[-4:] == '.csv':
            counter += 1
            print('Loading file: ' + i)
            rvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=0, dtype=float)[0]
            cvals = 1/rvals
            tvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=2, dtype=float)[0]
            data_arr = data_arr + [cvals]
            tval = [tvals]

            print(data_arr)

    #plt.ylabel('Resistance value (Ohms)')

    #plt.axhspan(1/rmin, 1/rmax, color = 'mistyrose')

    plt.hist(data_arr, bins = 100)
    plt.title('D1-R2-U8-3: time drift of two levels at time ' + str(tval[0]))
    plt.ylabel('Number of devices in state')
    plt.xlabel('Conductance (S)')
    #plt.legend()
    #plt.ylim(0,1.5e-7)
    plt.show()



        # pd.read_csv(i, header=None).T[1][1:][::-1]

import_data()


