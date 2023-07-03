import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import seaborn as sns
from scipy import stats
from scipy.optimize import leastsq
from pylab import *
from decimal import Decimal
# from scipy.optimize import curve_fit


# set target directory for retention plot - directory should contain CSV files for one device

files_to_plot = []
ctr = 0 
while ctr < 18:
    number = ctr*6 + 1
    files_to_plot.append('/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-06 6 state/six_state_standard/FIB3_M7_3_six_state_%s_low_drift_LP6dB6dBHz_Integ0.2_retentiondata.csv' % str(number))
    ctr += 1
print(files_to_plot)

colorlist = ('orange', 'yellow', 'green', 'brown', 'blue', 'indigo', 'red', 'black', 'violet', 'purple')
colorlist_2 = ('firebrick', 'salmon', 'chocolate', 'darkorange', 'gold', 'darkkhaki', 'olive', 'darkolivegreen', 'forestgreen', 'yellowgreen', 'seagreen', 'lightseagreen', 'darkcyan', 'cadetblue', 'steelblue', 'royalblue', 'midnightblue', 'slateblue', 'darkslateblue', 'rebeccapurple', 'darkviolet', 'darkmagenta', 'mediumvioletred')

reslist = (24400000.0, 30200000.0, 39900000.0, 58500000.0, 910000000.0, 110000000)


def color_plotter(input,resvals):
    len_vals = len(resvals)
    k = 0
    while k < len_vals:
        if input == resvals[k]:
            return colorlist[k]
        k += 1

# root = tk.Tk()
# root.withdraw()

# file_path = filedialog.askopenfilename()

# import data files - format should be CSV with info about resistance value, time, and target values

mulist = []
sigmalist = []

def import_data():
    global mulist,sigmalist
    plt.figure(dpi=120)
    plot1 = plt.subplot2grid((3, 10), (0, 0), rowspan=3, colspan=5)
    plot2 = plt.subplot2grid((3, 10), (0, 5), rowspan=3, colspan=5)
      
    ctr = 0
    for i in files_to_plot:
        if i[-5:] == 'a.csv':
            file_path = i 
            rvals = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=0, dtype=float)
            cvals = 1/rvals
            tvals = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=2, dtype=float)
            rmin = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=3, dtype=float)[1]
            rmax = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=4, dtype=float)[1]
            str_minmax = str("{:e}".format(rmin)) + ' Ohms to ' + str("{:e}".format(rmax)) + ' Ohms'

            #gaussian fit:
            (mu, sigma) = stats.norm.fit(cvals)
            mulist = mulist + [mu]
            sigmalist = sigmalist + [sigma]

            plot1.plot(tvals, cvals, label=str_minmax, color = colorlist_2[ctr], alpha = 0.6, linewidth = 0.5) #color = color_plotter(rmin, reslist), label=str_minmax, linewidth = 0.5)
            n, bins, patches = plot2.hist(cvals, 100, histtype='stepfilled', orientation='horizontal', density=True, ec="k", color = colorlist_2[ctr], alpha = 0.3, linewidth = 0.5) #label=str_minmax,
            
            fitline = stats.norm.pdf(bins, mu, sigma)
            musimp = str(mu)[:5] + str(mu)[-4:]
            sigsimp = str(sigma)[:5] + str(sigma)[-4:]
            l = plot2.plot(fitline, bins, color = colorlist_2[ctr], linestyle = 'dashed', linewidth=1, label = '%s' % ('mu = ' + musimp + ', sigma = ' + sigsimp))

            ctr +=1

    plot1.set_title('FIB3_M7_3: Gaussian fit to various set states', weight='bold', fontsize = 8)
    plot1.set_ylabel('Conductance value (S)', weight='bold', fontsize = 8)
    plot1.set_xlabel('Time (s)', weight='bold', fontsize = 8) 
    plot2.set_xlabel('Counts at value', weight='bold', fontsize = 8) 


    plot2.axhspan(1/rmin, 1/rmax, color = color_plotter(rmin, reslist), alpha = 0.2)
    plot1.axhspan(1/rmin, 1/rmax, color = color_plotter(rmin, reslist), alpha = 0.2)
    plot1.set_ylim(0,1.5/rmin)
    plot2.set_ylim(0,1.5/rmin)
    # plot2.set_xlim(0,1)
    # plot2.xaxis.set_ticklabels([])
    plot2.tick_params(left = False)
    plot2.set_yticklabels([])
    # plot2.legend()
    plt.show()

import_data()

print('Average mu value:')
print(np.average(mulist))
print('Average sigma value:')
print(np.average(sigmalist))