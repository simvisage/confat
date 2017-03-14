'''
Created on 19.09.2016

@author: Yingxiong
'''
from analytical import get_bond_slip
import numpy as np
import matplotlib.pyplot as plt


s_levels = np.linspace(0, 1000e-3, 2)
s_levels[0] = 0
s_levels.reshape(-1, 2)[:, 0] *= -1.
s_history = s_levels.flatten()

# slip array as input
s_arr = np.hstack([np.linspace(s_history[i], s_history[i + 1], 100)
                   for i in range(len(s_levels) - 1)])

# plt.plot(np.arange(len(s_history)), s_history)
# plt.show()

s_arr, tau_arr, tau_pi_arr, w_arr, xs_pi_arr = get_bond_slip(
    s_arr, tau_pi_bar=40)
ax1 = plt.subplot(111)
ax1.plot(s_arr, w_arr, label='tau_bar=40', alpha=0.5)
plt.xlabel('slip')
plt.ylabel('stress')


s_arr, tau_arr, tau_pi_arr, w_arr, xs_pi_arr = get_bond_slip(
    s_arr, tau_pi_bar=20)
ax1.plot(s_arr, w_arr, label='tau_bar=20', alpha=0.5)

s_arr, tau_arr, tau_pi_arr, w_arr, xs_pi_arr = get_bond_slip(
    s_arr, tau_pi_bar=1)
ax1.plot(s_arr, w_arr, label='tau_bar=1', alpha=0.5)

ax1.set_xlim(-0.15, 0.15)
ax1.legend(loc='best')
# ax1.set_title('Ad = 0.5')


plt.figure()

s_arr, tau_arr, tau_pi_arr, w_arr, xs_pi_arr = get_bond_slip(
    s_arr, tau_pi_bar=40, Ad=0.05, s0=5e-3, G=6000)
ax1 = plt.subplot(111)
ax1.plot(s_arr, tau_arr, label='tau_bar=40', alpha=0.5)
plt.xlabel('slip')
plt.ylabel('stress')

s_arr, tau_arr, tau_pi_arr, w_arr, xs_pi_arr = get_bond_slip(
    s_arr, tau_pi_bar=20, Ad=0.05, s0=5e-3, G=6000)
ax1.plot(s_arr, tau_arr, label='tau_bar=20', alpha=0.5)

s_arr, tau_arr, tau_pi_arr, w_arr, xs_pi_arr = get_bond_slip(
    s_arr, tau_pi_bar=1, Ad=0.05,  s0=5e-3, G=6000)
ax1.plot(s_arr, tau_arr, label='tau_bar=1', alpha=0.5)
ax1.legend(loc='best')
ax1.set_xlim(-0.15, 0.15)
# ax1.set_title('Ad = 0.007')
plt.show()
