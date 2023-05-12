# Data_Analysis_Codes_Memristor

Welcome to the BISOL memristor data analysis software suite. This project is constantly changing, but this should allow collaborators to easily access codes, data, and submit their own contributions. 

# Data output from measurement software:

Outputs now come in four different flavors: we have:\
• 2 png files from each target sweep, one detailing the level switching and the other detailing applied voltage \
• One csv file labeled 'filename'.csv. This contains the measured currents, the applied voltages, the number of pulses, and the pulse width\
• One csv file labeled 'filename_retentiondata.csv'. This file contains information about the resistance values, the timing, and the target resistance values in a retention sweep. Later iterations of the file also include information about the LNA sensitivity and the integration time.

# Analysis codes:

The important analysis codes are:\
replot.py - replots data from a folder to reconstruct the original PNG files output by the measurement software/
retention_levels_plot.py  - plots the time drift of a given set of data\
retention_levels_plot_avg.py - plots the average time drift of a set of measurements, along with the standard deviation at each point
retention_levels_plot_rms_error.py - accompanies retention_levels_plot_avg.py and gives a plot of the drift of the error.
