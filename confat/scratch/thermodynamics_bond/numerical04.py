'''
Created on 08.11.2016

@author: abaktheer
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton 

def get_bond_slip(s_arr, tau_pi_bar=10, Ad=0.5, s0=5e-3, G=36000.0):
    '''for plotting the bond slip relationship-Non analytical - Linear
    '''
    # arrays to store the values
    # nominal stress
    tau_arr = np.zeros_like(s_arr)
    # sliding stress
    tau_pi_arr = np.zeros_like(s_arr)
    # damage factor
    w_arr = np.zeros_like(s_arr)
    # sliding slip
    xs_pi_arr = np.zeros_like(s_arr)

    # material parameters
    # shear modulus [MPa]
    G = G
    # damage - brittleness [MPa^-1]
    Ad = Ad
    # Kinematic hardening modulus [MPa]
    gamma = 0
    # constant in the sliding threshold function
    tau_pi_bar = tau_pi_bar

    Z = lambda z: 1. / Ad * (-z) / (1 + z)

    # damage - Threshold
    s0 = s0
    Y0 = 0.5 * G * s0 ** 2
    # damage function
    
    # state variables
    tau_pi_i = 0
    alpha_i = 0.
    xs_pi_i = 0
    z_i = 0.
    w_i = 0.  # damage
    X_i = gamma * alpha_i
    
    
    for i in range(1, len(s_arr)):
        print 'increment', i
        s_i = s_arr[i]
        #ds_i = s_i - s_arr[i - 1]
        Yw_i = 0.5 * G * s_i ** 2
        # damage threshold function
        Ypi_i = 0.5 * G * (s_i - xs_pi_i) ** 2
        Y_i = Yw_i + Ypi_i
        fw = Yw_i - (Y0 + Z(z_i))
        #fw = Y_i - (Y0 + Z(z_i))
        # in case damage is activated

        if fw > 1e-8:
            dw = 0
            
            dw = G * (s_i )* (s_i - s_arr[i-1]) * Ad * (1 + z_i )**2 
            print 'dw=' , dw
           
            w_i = w_i + dw
            z_i = -w_i
            print 'w =' , w_i
        
        tau_pi_i = w_i * G * (s_i - xs_pi_i)
        f_pi_i = np.fabs(tau_pi_i - X_i) - tau_pi_bar

        if f_pi_i > 1e-6:
            # Return mapping
            d_lamda = f_pi_i / (w_i * G + gamma)
            tau_pi_i = tau_pi_i - w_i * G * d_lamda * np.sign(tau_pi_i - X_i)
            xs_pi_i = s_i -(tau_pi_i / (w_i * G))
            X_i = X_i + gamma * d_lamda * np.sign(tau_pi_i - X_i)
              
        # update all the state variables
        tau = (1 - w_i) * G * s_i + w_i * G * (s_i - xs_pi_i)
        tau_arr[i] = tau
        tau_pi_arr[i] = tau_pi_i
        w_arr[i] = w_i
        xs_pi_arr[i] = xs_pi_i
  
        print '------------------------ ' 

    return s_arr, tau_arr, tau_pi_arr, w_arr, xs_pi_arr


if __name__ == '__main__':
    s_levels = np.linspace(0, 200e-3, 2)
#     s_levels = np.linspace(10e-3, 10e-3, 10)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= -1
    s_history = s_levels.flatten()

    # slip array as input
    s_arr = np.hstack([np.linspace(s_history[i], s_history[i + 1], 1000)
                       for i in range(len(s_levels) - 1)])

    s_arr, tau_arr, tau_pi_arr, w_arr, xs_pi_arr = get_bond_slip(
        s_arr, tau_pi_bar=5, Ad=0.05, s0=5e-3, G=6000)
    plt.subplot(121)
    plt.plot(s_arr, tau_arr)  # , label='stress')
    plt.plot(s_arr, tau_pi_arr)  # , label='sliding stress')
    plt.xlabel('slip')
    plt.ylabel('stress')
    plt.legend()
    plt.subplot(122)
    plt.plot(s_arr, w_arr)
    plt.ylim(0, 1)
    plt.xlabel('slip')
    plt.ylabel('damage')
    plt.legend()
    

    plt.show()

