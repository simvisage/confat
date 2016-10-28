'''
Created on 27.08.2016

@author: Yingxiong, Rosoba
'''
import math

import matplotlib.pyplot as plt
import numpy as np


def get_bond_slip():
    '''for plotting the bond slip relationship
    '''
    s_levels = np.linspace(0, 100e-3, 100)
#    s_levels = np.linspace(10e-3, 10e-3, 10)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= -1
    s_history = s_levels.flatten()

    # slip array as input
    s_arr = np.hstack([np.linspace(s_history[i], s_history[i + 1], 30)
                       for i in range(len(s_levels) - 1)])

    # arrays to store the values
    # nominal stress
    tau_arr = np.zeros_like(s_arr)
    # sliding stress
    tau_pi_arr = np.zeros_like(s_arr)
    # damage factor
    w_arr = np.zeros_like(s_arr)
    # sliding slip
    s_pi_arr = np.zeros_like(s_arr)
    xs_pi_arr = np.zeros_like(s_arr)

    # material parameters
    # shear modulus [MPa]
    G = 36000.0
    # damage - brittleness [MPa^-1]
    Ad = 1.0
    # Kinematic hardening modulus [MPa]
    gamma = 10
    # constant in the sliding threshold functio
    tau_pi_bar = 10
    # parameter in the sliding potential
    a = 0

    Z = lambda z: 1. / Ad * (-z) / (1 + z)
    # damage - Threshold
    s0 = 5e-3
    Y0 = 0.5 * G * s0 ** 2
    # damage function
    f_damage = lambda Yw: 1 - 1. / (1 + Ad * (Yw - Y0))

    # state variables
    tau_pi_i = 0
    alpha_i = 0.
    s_pi_i = 0
    xs_pi_i = 0
    z_i = 0.
    w_i = 0.  # damage
    X_i = gamma * alpha_i

    for i in range(1, len(s_arr)):
        print 'increment', i
        s_i = s_arr[i]
        ds_i = s_i - s_arr[i - 1]
        Yw_i = 0.5 * G * s_i ** 2
        # damage threshold function
        Ypi_i = 0.5 * G * (s_i - s_pi_i)**2
        Y_i = Yw_i + Ypi_i
        fw = Y_i - (Y0 + Z(z_i))
        # in case damage is activated

        if fw > 1e-8:
            w_i = f_damage(Y_i)
            z_i = -w_i

        tau_pi_i = w_i * G * (s_i - s_pi_i)
        f_pi_i = np.fabs(tau_pi_i - X_i) - tau_pi_bar

        if f_pi_i > 1e-6:
            f_pi_n, tau_pi_n, X_n = f_pi_i, tau_pi_i, X_i
            s_pi_n, alpha_n = s_pi_i, alpha_i
            for n in range(6):
                # sliding threshold function
                d_f_pi_tau = np.sign(tau_pi_n - X_n)
                d_f_pi_X = -np.sign(tau_pi_n - X_n)
                d_phi_pi_tau = d_f_pi_tau
                d_phi_pi_X = d_f_pi_X + a * X_n

                d_lam_pi = f_pi_n / \
                    (w_i * G * d_f_pi_tau * d_phi_pi_tau +
                     gamma * d_f_pi_X * d_phi_pi_X
                     )

                d_s_pi = d_lam_pi * d_phi_pi_tau
                d_alpha = -d_lam_pi * d_phi_pi_X

                # update sliding and alpha
                s_pi_n += d_s_pi
                alpha_n += d_alpha
                tau_pi_n = w_i * G * (s_i - s_pi_n)
                X_n = gamma * alpha_n

                f_pi_n = np.fabs(tau_pi_n - X_n) - tau_pi_bar

                if f_pi_n <= 1e-8:
                    f_pi_i, tau_pi_i, X_i = f_pi_n, tau_pi_n, X_n
                    s_pi_i, alpha_i = s_pi_n, alpha_n
                    break

            xs_pi_i = (
                w_i * G * s_i - np.sign(ds_i) * tau_pi_bar) / (w_i * G + gamma)

            print s_pi_i, xs_pi_i
            s_pi_i = xs_pi_i

        # update all the state variables
        tau = (1 - w_i) * G * s_i + w_i * G * (s_i - s_pi_i)
        tau_arr[i] = tau
        tau_pi_arr[i] = tau_pi_i
        w_arr[i] = w_i
        s_pi_arr[i] = s_pi_i
        xs_pi_arr[i] = xs_pi_i

    return s_arr, tau_arr, tau_pi_arr, w_arr, s_pi_arr, xs_pi_arr

s_arr, tau_arr, tau_pi_arr, w_arr, s_pi_arr, xs_pi_arr = get_bond_slip()
plt.subplot(221)
plt.plot(s_arr, tau_arr)  # , label='stress')
plt.plot(s_arr, tau_pi_arr)  # , label='sliding stress')
plt.xlabel('slip')
plt.ylabel('stress')
plt.legend()
plt.subplot(222)
plt.plot(s_arr, w_arr)
plt.ylim(0, 1)
plt.xlabel('slip')
plt.ylabel('damage')
plt.subplot(223)
#plt.plot(s_arr, s_pi_arr)
plt.plot(s_arr, (s_pi_arr - xs_pi_arr), label='spi2')
plt.xlabel('slip')
plt.ylabel('sliding slip')
plt.legend()
plt.subplot(224)
plt.plot(s_arr[:-1], np.sign(
    (s_arr[1:] - s_arr[:--1]) * s_pi_arr[1:]), label='spi2')
plt.ylim(-1.5, 1.5)
plt.xlabel('slip')
plt.ylabel('sliding slip')
plt.legend()
# plt.ylim(s_arr[0], s_arr[-1])
plt.show()
