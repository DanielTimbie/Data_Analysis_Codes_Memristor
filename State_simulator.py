import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import os
from scipy import stats

colorlist = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf')
# reslist = (24400000.0, 30200000.0, 39900000.0, 58500000.0, 910000000.0, 110000000)
reslist = (2.375e7, 2.713e7, 3.164e7, 3.794e7, 4.738e7, 6.307e7, 9.429e7, 1.867e8, 9.5e9)

slope = 0.103
yint = 3.623e-10
clow = 8.611e-10
chigh = 4.128e-8
multiplier = 1.5

print('Target resistance values:')
cplot_ct = 0
counter = clow
plot1 = plt.subplot2grid((3, 10), (0, 0), rowspan=3, colspan=10)
while counter <= chigh:
    val = counter
    sigma = slope*val + yint
    dist = multiplier*sigma
    plot1.axvspan(val-dist,val+dist, color = colorlist[cplot_ct], alpha = 0.3,label\
                   = '%s' % (str(val - dist)[:5]+str(val - dist)[-4:] + ' to ' + str(val + dist)[:5]+str(val + dist)[-4:] ))
    
    print('range -')
    pm = val*0.0650353 + 2.06565e-10
    print(1/(val - pm))
    print(1/(val + pm))

    nlevel = False
    valprev = val
    distprev = multiplier*sigma
    while not nlevel:
        val = val + dist/100
        sigma = slope*val + yint
        dist = multiplier*sigma
        if val - dist > valprev + distprev:
            nlevel = True

    counter = val
    cplot_ct += 1

plt.title('Simulated conductance ranges between %s and %s' % (clow,chigh))
plt.xlabel('Conductance (S)')
plt.legend()
plt.show()