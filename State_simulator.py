import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats


# set target directory for retention plot - directory should contain CSV files for one device
# tg_dir = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-06 6 state/six_state_standard'
tg_dir = '/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-19 8 state proportional measurements' 


colorlist = ('orange', 'yellow', 'green', 'brown', 'blue', 'indigo', 'red', 'black', 'violet', 'purple')
colorlist = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf')
# reslist = (24400000.0, 30200000.0, 39900000.0, 58500000.0, 910000000.0, 110000000)
reslist = (2.375e7, 2.713e7, 3.164e7, 3.794e7, 4.738e7, 6.307e7, 9.429e7, 1.867e8, 9.5e9)



def color_plotter(input,resvals):
    len_vals = len(resvals)
    k = 0
    while k < len_vals:
        if input == resvals[k]:
            return colorlist[k]
        k += 1

def plot_selector(input,resvals):
    len_vals = len(resvals)
    k = 0
    while k < len_vals:
        if input == resvals[k]:
            return k
        k += 1


# import data files - format should be CSV with info about resistance value, time, and target values
def import_data():
    path = tg_dir
    dir_list = os.listdir(path)
    counter = 0
    data_arr = []
    data_levels = [[],[],[],[],[],[],[],[],[]]
    plot1 = plt.subplot2grid((3, 10), (0, 0), rowspan=3, colspan=10)
    # plot1 = plt.semilogx()
    for i in dir_list:
        if i[-5:] == 'a.csv':
            counter += 1
            print('Loading file: ' + i)
            rvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=0, dtype=float)[-1]
            cvals = 1/rvals
            print(cvals)
            tvals = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=2, dtype=float)[-1]
            rmin = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=3, dtype=float)[1]
            rmax = np.loadtxt(tg_dir + '/' + i, delimiter=',', skiprows=1, usecols=4, dtype=float)[1]
            data_levels[plot_selector(rmin,reslist)] = data_levels[plot_selector(rmin,reslist)] + [cvals]
            data_arr = data_arr + [cvals]
            tval = [tvals]
            # plot1.axvspan(1/rmin, 1/rmax, color = 'mistyrose')
    #plt.ylabel('Resistance value (Ohms)')

    #plt.axhspan(1/rmin, 1/rmax, color = 'mistyrose')
    ctr = 0
    for i in data_levels:
        (mu, sigma) = stats.norm.fit(i)
        print(i)
        n, bins, patches = plot1.hist(i, bins = 10, color = colorlist[ctr], alpha = 0.4)
        fitline = stats.norm.pdf(bins, mu, sigma)
        print(fitline)
        musimp = str(mu)[:5] + str(mu)[-4:]
        sigsimp = str(sigma)[:5] + str(sigma)[-4:]
        l = plot1.plot(bins, fitline/2e8, linestyle = 'dashed', linewidth=1, color = colorlist[ctr], label = '%s' % ('mu = ' + musimp + ', sigma = ' + sigsimp))
        ctr +=1
    plt.title('Fib3-I7-3: time drift of eight levels at time ' + str(tval[0]))
    plt.ylabel('Number of devices in state')
    plt.xlabel('Conductance (S)')
    #plt.xscale('log')
    plt.legend()
    #plt.ylim(0,1.5e-7)
    plt.show()



        # pd.read_csv(i, header=None).T[1][1:][::-1]

import_data()


