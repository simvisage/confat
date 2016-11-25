'''
Created on 18.11.2016

@author: abaktheer

'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton 


class MATS1DBondSlip():
  
    G = 6000
    gamma = 0
    Ad = 0.05
    tau_pi_bar = 5
    s0 = 5e-3
    sctx = np.zeros(5)
    
    def get_corr_pred(self, sctx, s , ds): 
         
        xs_pi = sctx[0]
        X = sctx[1]
        w = self.get_omega(sctx , s , ds)
        print'w__=' , w
        
        tau = (1 - w) * self.G * s + w * self.G * (s - xs_pi)
        tau_pi = w * self.G * (s - xs_pi)
        D_ed = self.get_stiffness(sctx, s, ds)
        
        f_pi = np.fabs(tau_pi - X) - self.tau_pi_bar
    
        if f_pi > 1e-8:
            # Return mapping 
            delta_lamda = f_pi / (w * self.G + self.gamma)
            # update variables
            xs_pi = xs_pi + delta_lamda * np.sign(tau_pi - X)
            tau_pi = w * self.G * (s - xs_pi)
            tau = self.G * (1 - w) * s + self.G * w * (s - xs_pi)
            
        return tau, tau_pi , D_ed
      
    def get_state_variables(self, sctx, s , ds):
        
        xs_pi = sctx[0]
        X = sctx[1]
        alpha = sctx[2]
        w = self.get_omega(sctx , s, ds)
        z = self.sctx[3]
        
        tau_pi = w * self.G * (s - xs_pi)
        f_pi = np.fabs(tau_pi - X) - self.tau_pi_bar
    
        if f_pi > 1e-6:
            # Return mapping 
            delta_lamda_pi = f_pi / (self.G + self.gamma + (1 / (self.Ad * (1 + z) ** 2)))
            # update all the state variables
            xs_pi = xs_pi + delta_lamda_pi * np.sign(tau_pi - X)
            X = X + self.gamma * delta_lamda_pi * np.sign(tau_pi - X)
            alpha = alpha + xs_pi
            
            self.sctx[0] = xs_pi
            self.sctx[1] = X
            self.sctx[2] = alpha
        return  xs_pi, X, alpha, z, w
 
    def get_omega(self, sctx, s , ds):
        
        z = sctx[3] 
        w = sctx[4]
        
        Y = 0.5 * self.G * s ** 2
        Z = lambda z: 1. / self.Ad * (-z) / (1 + z)
        f_w = Y - (0.5 * self.G * self.s0 ** 2) - Z(z)
        
        'Analytical'
        '''
        if f_w > 1e-6:
            Y0 = 0.5 * self.G * self.s0 ** 2 
            f = lambda Y_w: 1 - 1. / (1 + self.Ad * (Y_w - Y0))
            w = f(Y)
            z = -w
        '''
        
        'Nonlinear'
        '''
        if f_w > 1e-6:
            f_dw_n = lambda dw_n :  dw_n - self.G * (s) * ds * self.Ad * (1 + z - dw_n) ** 2 
            f_dw_n2 = lambda dw_n : 1 + 2 * self.G * (s) * ds * self.Ad * (1 + z - dw_n) 
            dw_n = newton(f_dw_n, 0., fprime=f_dw_n2 , tol=1e-6, maxiter=40)
            w = w + dw_n
            z = z - dw_n
        '''
        
        'Linear'
        if f_w > 1e-6:   
            dw_l = self.G * s * ds * self.Ad * (1 + z) ** 2 
            w = w + dw_l
            z = z - dw_l
            
            
        self.sctx[4] = w 
        self.sctx[3] = z 
        return  w   
     
    def get_stiffness(self, sctx, s , ds):
       
        xs_pi = sctx[0]
        X = sctx[1]
        z = sctx[3]
        
        w = self.get_omega(sctx , s , ds)
        tau_pi = w * self.G * (s - xs_pi)
        D_ed = (1 - w) * self.G 
        
        f_pi = np.fabs(tau_pi - X) - self.tau_pi_bar
    
        if f_pi > 1e-8:
            # Return mapping 
            delta_lamda = f_pi / (w * self.G + self.gamma)
            # update variables
            xs_pi = xs_pi + delta_lamda * np.sign(tau_pi - X)
            tau_pi = w * self.G * (s - xs_pi)
    
            D_ed = self.G - ((w * self.G) ** 2) / (w * self.G + self.gamma) - ((self.G) ** 2) * s * self.Ad * ((1 + z) ** 2) * xs_pi 
            
        print 'D_ed =', D_ed    
        return D_ed
             

a = MATS1DBondSlip()


s_levels = np.linspace(0, 200e-3, 30)
s_levels[0] = 0
s_levels.reshape(-1, 2)[:, 0] *= -1
s_history = s_levels.flatten()

s_arr = np.hstack([np.linspace(s_history[i], s_history[i + 1], 200)
                       for i in range(len(s_levels) - 1)])

tau = np.zeros_like(s_arr) ; tau_pi = np.zeros_like(s_arr) ; D_ed = np.zeros_like(s_arr) ; w = np.zeros_like(s_arr)

sctx_0 = np.array([0, 0, 0, 0, 0])                      
for i in range(0, len(s_arr)):
        state = a.get_state_variables(sctx_0, s_arr[i], s_arr[i] - s_arr[i - 1])
        corr_pre = a.get_corr_pred(sctx_0, s_arr[i], s_arr[i] - s_arr[i - 1])
        sctx_0 = state

        tau[i] = corr_pre[0] ; tau_pi[i] = corr_pre[1] ; D_ed[i] = corr_pre[2] ; w[i] = state[4]
        
        
plt.subplot(121)
plt.plot(s_arr, tau)  # , label='stress')
plt.plot(s_arr, tau_pi)  # , label='sliding stress')
plt.xlabel('slip')
plt.ylabel('stress')
plt.legend()
plt.subplot(122)
plt.plot(s_arr, w)
plt.ylim(0, 1)
plt.xlabel('slip')
plt.ylabel('damage')
plt.legend()

plt.show()
        
