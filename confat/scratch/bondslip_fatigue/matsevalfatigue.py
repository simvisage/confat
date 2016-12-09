'''
Created on 05.12.2016

@author: abaktheer
'''

from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List
import numpy as np



class MATSEvalFatigue(HasTraits):

    E_m = Float(28484, tooltip='Stiffness of the matrix [MPa]',
                auto_set=True, enter_set=True)

    E_f = Float(170000, tooltip='Stiffness of the fiber [MPa]',
                auto_set=False, enter_set=False)

    E_b = Float(6000,
                    label="E_b",
                    desc="Bond stiffness [MPa]",
                    enter_set=True,
                    auto_set=False)
    
    gamma = Float(0,
                    label="Gamma",
                    desc="Kinematic hardening modulus",
                    enter_set=True,
                    auto_set=False)
    
    K = Float(0,
                    label="K",
                    desc="Isotropic harening",
                    enter_set=True,
                    auto_set=False)
    
    S = Float(0.1,
                    label="S",
                    desc="Damage cumulation parameter",
                    enter_set=True,
                    auto_set=False)
    
    r = Float(1,
                    label="r",
                    desc="Damage cumulation parameter",
                    enter_set=True,
                    auto_set=False)
    
    tau_pi_bar = Float(5,
                    label="Tau_pi_bar",
                    desc="Reversibility limit",
                    enter_set=True,
                    auto_set=False)
    
    n_s = Constant(4)

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, xs_pi, alpha, z, w):
      
        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f
        
        Y = 0.5 * self.E_b * eps[:, :, 1] ** 2
        # sig[:, :, 1] = (1 - w) * self.E_b * eps[:, :, 1] + w * self.E_b * (eps[:, :, 1] - xs_pi)
        # sig_pi = w * self.E_b * (eps[:, :, 1] - xs_pi)
        sig_pi_trial = self.E_b * (eps[:, :, 1] - xs_pi)
        Z = self.K * z
        X = self.gamma * alpha
        f = np.fabs(sig_pi_trial - X) - self.tau_pi_bar - Z
        
        elas = f <= 1e-6
        plas = f > 1e-6
        
        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig
        
        # Return mapping 
        delta_lamda = f / (self.E_b + self.gamma + self.K) * plas
        # update all the state variables
        xs_pi = xs_pi + delta_lamda * np.sign(sig_pi_trial - X)
        # sig_pi = w * self.E_b * (eps[:, :, 1] - xs_pi)
        sig[:, :, 1] = self.E_b * (1 - w) * eps[:, :, 1] + self.E_b * w * (eps[:, :, 1] - xs_pi)
            
        X = X + self.gamma * delta_lamda * np.sign(sig_pi_trial - X)
        alpha = alpha + delta_lamda * np.sign(sig_pi_trial - X)
        z = z + delta_lamda
        
        w_1 = w + ((Y / self.S) ** self.r) * delta_lamda #/(1 - w)
        
        befor = w_1 < 1
        after = w_1 > 1
        
        w = w_1 * befor + 0.999 * after 
        
        
         
            # Consistent tangent operator
        D_ed = self.E_b - (w * (self.E_b) ** 2) / (self.E_b + self.gamma + self.K)\
        - (((self.E_b) ** 2) * xs_pi * ((Y / self.S) ** self.r) * np.sign(sig_pi_trial - X)) /  (self.E_b + self.gamma + self.K)
             
        D[:, :, 1, 1] = (1 - w) * self.E_b * elas + D_ed * plas
        
        #print'D_ed=', D[:, :, 1, 1]    
        return sig, D, xs_pi, alpha, z, w   
   
    
  
