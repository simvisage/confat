'''
Created on 25.11.2016

@author: abaktheer
'''
import matplotlib.pyplot as plt
import numpy as np 

def get_bond_slip(s_arr, tau_pi_bar, K, s0, G, S, r):
    '''for plotting the bond slip fatigue - Initial version
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
    # max sliding
    s_max = np.zeros_like(s_arr)
    # max stress
    tau_max = np.zeros_like(tau_arr)

    # material parameters
    # shear modulus [MPa]
    G = G
    # Isotropic hardening  modulus [MPa]
    K = K
    # Kinematic hardening modulus [MPa]
    gamma = 0
    # constant in the sliding threshold function
    tau_pi_bar = tau_pi_bar
    # Damage strength [MPa]
    S = S
    r = r
    s0 = s0
     
    # state variables
    tau_pi_i = 0
    alpha_i = 0.
    xs_pi_i = 0
    z_i = 0.
    w_i = 0.  # damage
    X_i = gamma * alpha_i
    delta_lamda = 0
    Z = K * z_i 
    
    for i in range(1, len(s_arr)):
        print 'increment', i
        s_i = s_arr[i]
        s_max_i = np.fabs(s_i)
        
        tau_i = G * (1 - w_i) * s_i + G * w_i * (s_i - xs_pi_i)
        tau_pi_i_1 = G * (s_i - xs_pi_i)
        tau_pi_i = w_i * G * (s_i - xs_pi_i)
        
        Yw_i = 0.5 * G * s_i ** 2 
        # Threshold
        f_pi_i = np.fabs(tau_pi_i_1 - X_i) - tau_pi_bar - Z
        
        if f_pi_i > 1e-6:
            # Return mapping 
            delta_lamda = f_pi_i / (G + gamma + K)
            # update all the state variables
            xs_pi_i = xs_pi_i + delta_lamda * np.sign(tau_pi_i_1 - X_i)
            w_i = w_i + delta_lamda * (Yw_i / S) ** r
            tau_pi_i = w_i * G * (s_i - xs_pi_i)
            tau_i = G * (1 - w_i) * s_i + G * w_i * (s_i - xs_pi_i)
            X_i = X_i + gamma * delta_lamda * np.sign(tau_pi_i_1 - X_i)
            alpha_i = alpha_i + xs_pi_i
            z_i = z_i + delta_lamda
            Z = K * z_i 
             
            if w_i >= 1:
                break
             
        tau_max_i = np.fabs(tau_i)  
        tau_arr[i] = tau_i
        tau_pi_arr[i] = tau_pi_i
        w_arr[i] = w_i
        xs_pi_arr[i] = xs_pi_i
        s_max[i] = s_max_i
        tau_max[i] = tau_max_i

        print 'w =', w_i
        print 'stress =', tau_i
        print 'sliding - stress =', tau_pi_i
        print 'slip =', s_i
        print 'sliding slip =', xs_pi_i
        print '------------------------------'
        
    return s_arr, tau_arr, tau_pi_arr, w_arr, xs_pi_arr , s_max , tau_max 

if __name__ == '__main__':
    s_levels = np.linspace(0, 50e-4, 100)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= -1
    # s_levels.reshape(-1, 2)[:, 1]  = 18e-4
    s_history = s_levels.flatten()

    # slip array as input
    s_arr = np.hstack([np.linspace(s_history[i], s_history[i + 1], 100)
                       for i in range(len(s_levels) - 1)])

    s_arr, tau_arr, tau_pi_arr, w_arr, xs_pi_arr , s_max , tau_max = get_bond_slip(
        s_arr, tau_pi_bar=5 , K=0, s0=5e-3 , G=15000.0 , S=0.025 , r=1)
    print 'Max_slip', np.amax(s_max)
    print 'Max_stress', np.amax(tau_max)
    plt.subplot(121)
    plt.plot(s_arr, tau_arr)  
    plt.plot(s_arr, tau_pi_arr)
    plt.xlabel('slip')
    plt.ylabel('stress')
    plt.legend()
    plt.subplot(122)
    plt.plot(s_arr , w_arr)
    plt.ylim(0, 1)
    plt.xlabel('slip')
    plt.ylabel('damage')
    plt.legend()

    plt.show()
 
    
    
    
    
    
    
