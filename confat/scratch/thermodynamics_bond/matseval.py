'''
Created on 27.08.2016

@author: Yingxiong
'''
import matplotlib.pyplot as plt
import numpy as np


def get_bond_slip():
    '''for plotting the bond slip relationship
    '''
    # slip array as input
    s_arr = np.hstack((np.linspace(0, 1e-3, 10),
                       np.linspace(1e-3, 7e-4, 10),
                       np.linspace(7e-4, 1.5e-3, 10),
                       np.linspace(-1.5e-3, 7e-4, 10),
                       np.linspace(7e-4, 2.0e-3, 10),
                       ))
#     s_arr = np.linspace(0, 1e-3, 200)
    # arrays to store the values
    # nominal stress
    tau_arr = np.zeros_like(s_arr)
    # sliding stress
    tau_pi_arr = np.zeros_like(s_arr)
    # damage factor
    w_arr = np.zeros_like(s_arr)
    # sliding slip
    s_pi_arr = np.zeros_like(s_arr)

    # material parameters
    # shear modulus [MPa]
    G = 36000.
    # damage - brittleness [MPa^-1]
    Ad = 1e2
    Z = lambda z: 1. / Ad * (-z) / (1 + z)
    # damage - Threshold
    s0 = 2.5e-4
    Y0 = 0.5 * G * s0 ** 2
    # damage function
    f_damage = lambda Yw: 1 - 1. / (1 + Ad * (Yw - Y0))
    # Kinematic hardening modulus [MPa]
    gamma = 2e5
    # constant in the sliding threshold functio
    tau_pi_bar = 1
    # parameter in the sliding potential
    a = 2

    # state variables
    tau = 0.
    tau_pi = 1e-10
    alpha = 0.
    z = 0.
    s_n = 0.  # total slip
    s_pi = 0.  # sliding slip
    w = 0.  # damage
    X = gamma * alpha

    # value of sliding threshold function at previous step
    f_pi = -tau_pi_bar
    # value of sliding stress at previous step

    for i in range(1, len(s_arr)):
        print 'increment', i
        d_s = s_arr[i] - s_arr[i - 1]
        s_n1 = s_n + d_s
        Yw = 0.5 * G * s_n1 ** 2

        # damage threshold function
        fw = Yw - (Y0 + Z(z))
        # in case damage is activated
        if fw > 1e-8:
            w = f_damage(Yw)
            z = -w

            tau_pi_trial = w * G * (s_n1 - s_pi)

            f_pi = np.fabs(tau_pi_trial - X) - tau_pi_bar
            while True:
                # sliding threshold function
                print 'iteration'
                d_f_pi_tau = np.sign(tau_pi_trial - X)
                d_f_pi_X = -np.sign(tau_pi_trial - X)
                d_phi_pi_tau = d_f_pi_tau
                d_phi_pi_X = d_f_pi_X + a * X

                d_lam_pi = f_pi / \
                    (w * G * d_f_pi_tau * d_phi_pi_tau -
                     gamma * d_f_pi_X * d_phi_pi_X
                     )

                d_s_pi = d_lam_pi * d_phi_pi_tau
                d_alpha = d_lam_pi * d_phi_pi_X
                d_tau_pi = -w * G * d_s_pi
                d_X = gamma * d_alpha

                # update sliding and alpha
                s_pi += d_s_pi
                alpha += d_alpha
                tau_pi += d_tau_pi
                X += d_X

                # update the threshold value
                f_pi = np.fabs(tau_pi - X) - tau_pi_bar
                if f_pi < 1e-6:
                    break

        # update all the state variables
        tau = (1 - w) * G * s_n1 + tau_pi
        tau_arr[i] = tau
        tau_pi_arr[i] = tau_pi
        w_arr[i] = w
        s_pi_arr[i] = s_pi

#         tau_pi_old = tau_pi
#         f_pi_old = np.abs(tau_pi - X(alpha)) - tau_pi_bar

    return s_arr, tau_arr, tau_pi_arr, w_arr, s_pi_arr

s_arr, tau_arr, tau_pi_arr, w_arr, s_pi_arr = get_bond_slip()
plt.subplot(221)
plt.plot(s_arr, tau_arr, label='stress')
plt.plot(s_arr, tau_pi_arr, label='sliding stress')
plt.xlabel('slip')
plt.ylabel('stress')
plt.legend()
plt.subplot(222)
plt.plot(s_arr, w_arr)
plt.ylim(0, 1)
plt.xlabel('slip')
plt.ylabel('damage')
plt.subplot(223)
plt.plot(s_arr, s_pi_arr)
plt.xlabel('slip')
plt.ylabel('sliding slip')
# plt.ylim(s_arr[0], s_arr[-1])
plt.show()
