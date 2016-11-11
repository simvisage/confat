'''
Created on 03.11.2016

@author: abaktheer
'''
import matplotlib.pyplot as plt
import numpy as np

def get_bond_slip(eps_arr, sigma_s, C , K , S , E1 , E2):
    
    # arrays to store the values
    # nominal stress
    sigma_arr = np.zeros_like(eps_arr)
    # sliding stress
    sigma_pi_arr = np.zeros_like(eps_arr)
    # damage factor
    w_arr = np.zeros_like(eps_arr)
    # sliding slip
    eps_pi_arr = np.zeros_like(eps_arr)
    # max strain
    eps_max = np.zeros_like(eps_arr)
    # max stress
    sigma_max = np.zeros_like(sigma_arr)
    
    'material parameters'
    # Young modulus
    E1 = E1
    E2 = E2
    # constant in the sliding threshold function
    sigma_s = sigma_s
    # Hardening modulus [MPa]
    C = C
    # Consolidation modulus [MPa]
    K = K
    # Damage strength [MPa]
    S = S
    
    'state variables'
    sigma_pi_i = 0.
    alpha_i = 0.
    r_i = 0.
    eps_pi_i = 0
    R_i = K * r_i
    w_i = 0.  # damage
    X_i = C * alpha_i
    
    
    for i in range(1, len(eps_arr)):
        print 'increment', i
        eps_i = eps_arr[i]
        
        eps_max_i = np.fabs(eps_i)
        
        d_eps_i = eps_i - eps_arr[i - 1]
        
        sigma_i = E1 * (1- w_i) * eps_i + E2 * (1- w_i) * (eps_i - eps_pi_i)
        sigma_pi_i = E2 * (1 - w_i) * (eps_i - eps_pi_i)
        
        Y_i = 0.5 * E1 * eps_i **2 + 0.5 * E2 * (eps_i - eps_pi_i)**2
        f_pi_i = np.fabs(sigma_pi_i /(1- w_i) - X_i) - R_i - sigma_s 
        
        if f_pi_i > 1e-6:
            # Return mapping
            delta_pi = f_pi_i / (E2 + (C + K)*(1 - w_i))
            # update all the state variables
            eps_pi_i = eps_pi_i + delta_pi * np.sign(sigma_pi_i /(1- w_i) - X_i) 
            Y_i = 0.5 * E1 * eps_i **2 + 0.5 * E2 * (eps_i - eps_pi_i)**2
            w_i = w_i + delta_pi * (Y_i / S)
            r_i = r_i + (1 - w_i) * delta_pi
            sigma_pi_i = E2 *  (1 - w_i) * (eps_i - eps_pi_i)
            sigma_i = E1 * (1 - w_i) * eps_i + sigma_pi_i
            X_i = X_i + C * (1 - w_i) * delta_pi
        
            
        
        if w_i >= 0.9:
         break
     
        sigma_max_i = np.fabs(sigma_i)  
        sigma_arr[i] = sigma_i
        sigma_pi_arr[i] = sigma_pi_i
        w_arr[i] = w_i
        eps_pi_arr[i] = eps_pi_i
        eps_max[i]= eps_max_i
        sigma_max[i]= sigma_max_i
          
        print 'Damage w =', w_i
        #print 'f_pi_i=', f_pi_i
        #print 'Y_i=', Y_i
        print 'stress=',sigma_i
        print 'sliding stress=',sigma_pi_i
        print 'strain=',eps_i
        print 'sliding strain=',eps_pi_i
        print '------------------------------'
        
    return eps_arr, sigma_arr, sigma_pi_arr, w_arr, eps_pi_arr , eps_max , sigma_max


if __name__ == '__main__':
    eps_levels = np.linspace(0, -12e-4, 100)
#     s_levels = np.linspace(10e-3, 10e-3, 10)
    eps_levels[0] = 0
    eps_levels.reshape(-1, 2)[:,0] *= -1
    eps_history = eps_levels.flatten()

    # slip array as input
    eps_arr = np.hstack([np.linspace(eps_history[i], eps_history[i + 1], 30)
                       for i in range(len(eps_levels) - 1)])

    eps_arr, sigma_arr, sigma_pi_arr, w_arr, eps_pi_arr , eps_max ,sigma_max = get_bond_slip(
        eps_arr,sigma_s=9, C=0 , K=0 , S=324e-6 , E1=20000.0 , E2=15000.0)
    print 'Max_strain', np.amax(eps_max)
    print 'Max_strain', np.amax(sigma_max)
    plt.subplot(121)
    plt.plot(eps_arr, sigma_arr)  # , label='stress')
    plt.plot(eps_arr, sigma_pi_arr)  # , label='sliding stress')
    plt.xlabel('strain')
    plt.ylabel('stress')
    plt.legend()
    plt.subplot(122)
    plt.plot(eps_arr , w_arr )
    plt.ylim(0, 1)
    plt.xlabel('strain')
    plt.ylabel('damage')
    plt.legend()

    plt.show()
 
    
    


   
        
        