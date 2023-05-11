import matplotlib.pyplot as plt
import numpy as np

eps_tio2 = 11
eps_al203 = 7.5
A = 2.82e-9 #meters: 30um radius contact
d1 = 40e-9 #meters
d2 = 10e-9 #meters

#shunt resistor value in ohms, approx initial resistance state
R_shunt = 15000
R_init = 1e10


C = (eps_al203*eps_tio2*A)/(eps_tio2*d2 + eps_al203*d2)
print(C)

vrange = np.arange(0,15,0.5)
E_tio2 = C*vrange/(eps_tio2*A)
E_al203 = C*vrange/(eps_al203*A)

#want Mv/cm, so multiply E by 1e-6/100 = 1e-8

plt.plot(vrange, 1e-8*E_tio2, label='E_tio2')
plt.plot(vrange, 1e-8*E_al203, label='E_al203')
plt.title('Electric field in Dielectric Layers at different V values')
plt.legend()
plt.ylabel('Electric Field (MV/cm)')
plt.xlabel('V_0 (V)')
plt.show()

