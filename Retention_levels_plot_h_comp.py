import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

# set target directory for retention plot - directory should contain CSV files for one device

files_to_plot = []
ctr = 0 
while ctr < 20:
    number = ctr*6 + 4
    files_to_plot.append('/Users/daniel/Documents/Northwestern 2020-2023/Labwork/Memristor/Data_Analysis_Codes_Memristor/data/2023-06-06 6 state/six_state_standard/FIB3_M7_3_six_state_%s_low_drift_LP6dB6dBHz_Integ0.2_retentiondata.csv' % str(number))
    ctr += 1
print(files_to_plot)

#colorlist = ('blue', 'red', 'orange', 'yellow', 'green', 'brown', 'indigo', 'black', 'violet', 'purple')
colorlist = ('orange', 'yellow', 'green', 'brown', 'blue', 'indigo', 'red', 'black', 'violet', 'purple')
reslist = (12200000.0, 45300000.0, 16100000.0, 10200000.0, 23800000.0, 477000000.0, 500000000)
reslist = (24400000.0, 30200000.0, 39900000.0, 58500000.0, 910000000.0, 110000000)

colorlist_2 = ('firebrick', 'salmon', 'chocolate', 'darkorange', 'gold', 'darkkhaki', 'olive', 'darkolivegreen', 'forestgreen', 'yellowgreen', 'seagreen', 'lightseagreen', 'darkcyan', 'cadetblue', 'steelblue', 'royalblue', 'midnightblue', 'slateblue', 'darkslateblue', 'rebeccapurple', 'darkviolet', 'darkmagenta', 'mediumvioletred')

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
def import_data():
    plt.figure(dpi=120)
    plot1 = plt.subplot2grid((3, 10), (0, 0), rowspan=3, colspan=5)
    plot2 = plt.subplot2grid((3, 10), (0, 5), rowspan=3, colspan=1)  
    plot3 = plt.subplot2grid((3, 10), (0, 6), rowspan=3, colspan=1)  
    plot4 = plt.subplot2grid((3, 10), (0, 7), rowspan=3, colspan=1)  
    plot5 = plt.subplot2grid((3, 10), (0, 8), rowspan=3, colspan=1)  
    plot6 = plt.subplot2grid((3, 10), (0, 9), rowspan=3, colspan=1)  

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

            plot1.plot(tvals, cvals, label=str_minmax, color = colorlist_2[ctr], alpha = 0.6, linewidth = 0.5) #color = color_plotter(rmin, reslist), label=str_minmax, linewidth = 0.5)

            graph_var = ctr%5 + 1
            if graph_var == 1:
                plot2.hist(cvals, bins=100, histtype='stepfilled', orientation='horizontal', density=True, ec="k", color = colorlist_2[ctr], alpha = 0.3, label=str_minmax, linewidth = 0.5) #linestyle='dashed', color = color_plotter(rmin, reslist), label=str_minmax)
            if graph_var == 2:
                plot3.hist(cvals, bins=100, histtype='stepfilled', orientation='horizontal', density=True, ec="k", color = colorlist_2[ctr], alpha = 0.3, label=str_minmax, linewidth = 0.5) #linestyle='dashed', color = color_plotter(rmin, reslist), label=str_minmax)
            if graph_var == 3:
                plot4.hist(cvals, bins=100, histtype='stepfilled', orientation='horizontal', density=True, ec="k", color = colorlist_2[ctr], alpha = 0.3, label=str_minmax, linewidth = 0.5) #linestyle='dashed', color = color_plotter(rmin, reslist), label=str_minmax)
            if graph_var == 4:
                plot5.hist(cvals, bins=100, histtype='stepfilled', orientation='horizontal', density=True, ec="k", color = colorlist_2[ctr], alpha = 0.3, label=str_minmax, linewidth = 0.5) #linestyle='dashed', color = color_plotter(rmin, reslist), label=str_minmax)
            if graph_var == 5:
                plot6.hist(cvals, bins=100, histtype='stepfilled', orientation='horizontal', density=True, ec="k", color = colorlist_2[ctr], alpha = 0.3, label=str_minmax, linewidth = 0.5) #linestyle='dashed', color = color_plotter(rmin, reslist), label=str_minmax)

            ctr += 1         

    plot1.set_title('FIB3_M7_3: Time Drift of One State - Standard', weight='bold', fontsize = 8)
    plot1.set_ylabel('Conductance value (S)', weight='bold', fontsize = 8)
    plot1.set_xlabel('Time (s)', weight='bold', fontsize = 8) 
    plot3.set_xlabel('Counts at value', weight='bold', fontsize = 8) 


    plot2.axhspan(1/rmin, 1/rmax, color = color_plotter(rmin, reslist), alpha = 0.2)
    plot3.axhspan(1/rmin, 1/rmax, color = color_plotter(rmin, reslist), alpha = 0.2)
    plot4.axhspan(1/rmin, 1/rmax, color = color_plotter(rmin, reslist), alpha = 0.2)
    plot5.axhspan(1/rmin, 1/rmax, color = color_plotter(rmin, reslist), alpha = 0.2)
    plot6.axhspan(1/rmin, 1/rmax, color = color_plotter(rmin, reslist), alpha = 0.2)
    plot1.axhspan(1/rmin, 1/rmax, color = color_plotter(rmin, reslist), alpha = 0.2)
    plot1.set_ylim(0,1.5/rmin)
    plot2.set_ylim(0,1.5/rmin)
    plot3.set_ylim(0,1.5/rmin)
    plot4.set_ylim(0,1.5/rmin)
    plot5.set_ylim(0,1.5/rmin)
    plot6.set_ylim(0,1.5/rmin)
    xlim = 20e8
    plot2.set_xlim(0,xlim)
    plot3.set_xlim(0,xlim)
    plot4.set_xlim(0,xlim)
    plot5.set_xlim(0,xlim)
    plot6.set_xlim(0,xlim)
    plot2.xaxis.set_ticklabels([])
    plot3.xaxis.set_ticklabels(['.' ])
    plot4.xaxis.set_ticklabels([])
    plot5.xaxis.set_ticklabels([])
    #plot6.xaxis.set_ticklabels([])
    plot2.tick_params(left = False)
    plot3.tick_params(left = False)
    plot4.tick_params(left = False)
    plot5.tick_params(left = False)
    plot6.tick_params(right = True)
    plot6.tick_params(left = False)
    plot6.yaxis.tick_right()
    plot2.set_yticklabels([])
    plot3.set_yticklabels([])
    plot4.set_yticklabels([])
    plot5.set_yticklabels([])
    plt.show()

import_data()


