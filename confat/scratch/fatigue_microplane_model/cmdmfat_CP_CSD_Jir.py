'''
Created on 29.03.2017

@author: abaktheer

Microplane Fatigue model v1.2 (compression plasticiy + Damage sliding)
'''

from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats_eval import \
    IMATSEval
from numpy import \
    array, zeros, dot, trace, \
    tensordot, einsum, zeros_like,\
    identity, sign, linspace, hstack, \
    sqrt, copy
from numpy.linalg import norm
from scipy.linalg import \
    eigh
from traits.api import \
    Constant, implements,\
    Bool, Enum, Float, HasTraits, \
    Property, cached_property
from traitsui.api import \
    Item, View, Group, Spring, Include
import matplotlib.pyplot as plt


class MATSEvalMicroplaneFatigue(HasTraits):
    #--------------------------
    # model material parameters
    #--------------------------

    E = Float(34000,
              label="G",
              desc="Young modulus",
              enter_set=True,
              auto_set=False)

    nu = Float(0.2,
               label="G",
               desc="poission ratio",
               enter_set=True,
               auto_set=False)

    #---------------------------------------
    # Tangential constitutive law parameters
    #---------------------------------------
    gamma = Float(100,
                  label="Gamma",
                  desc="Kinematic hardening modulus",
                  enter_set=True,
                  auto_set=False)

    K = Float(0,
              label="K",
              desc="Isotropic harening",
              enter_set=True,
              auto_set=False)

    S = Float(0.00001,
              label="S",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    r = Float(1.0,
              label="r",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    c = Float(1.0,
              label="c",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    tau_pi_bar = Float(4.0,
                       label="Tau_pi_bar",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    a = Float(0.0,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    #-------------------------------------------
    # Normal_Tension constitutive law parameters
    #-------------------------------------------
    Ad = Float(0.1,
               label="a",
               desc="Lateral pressure coefficient",
               enter_set=True,
               auto_set=False)

    eps_0 = Float(5e-10,
                  label="a",
                  desc="Lateral pressure coefficient",
                  enter_set=True,
                  auto_set=False)

    #-----------------------------------------------
    # Normal_Compression constitutive law parameters
    #-----------------------------------------------
    K = Float(1000,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    gamma_N = Float(5000,
                    label="a",
                    desc="Lateral pressure coefficient",
                    enter_set=True,
                    auto_set=False)

    sigma_0 = Float(10,
                    label="a",
                    desc="Lateral pressure coefficient",
                    enter_set=True,
                    auto_set=False)

    def get_normal_compression_Law(self, eps, sctx):

        #--------------------------------------------------------------
        # microplane constitutive law (normal-compression)-(Plasticity)
        #--------------------------------------------------------------
        alpha_N = sctx[2]
        r_N = sctx[3]
        eps_N_p = sctx[4]

        sigma_n_trial = self.E * (eps - eps_N_p)
        Z = self.K * r_N
        X = self.gamma_N * alpha_N

        f_trial = abs(sigma_n_trial - X) - self.sigma_0 - Z

        if f_trial > 1e-6:
            delta_lamda = f_trial / (self.E + self.K + self.gamma_N)
            eps_N_p = eps_N_p + delta_lamda * sign(sigma_n_trial - X)
            r_N = r_N + delta_lamda
            alpha_N = alpha_N + delta_lamda * sign(sigma_n_trial - X)

        new_sctx = zeros(3)
        new_sctx[0] = alpha_N
        new_sctx[1] = r_N
        new_sctx[2] = eps_N_p
        return new_sctx

    def get_normal_tension_Law(self, eps, sctx):
        #------------------------------------------------------
        # microplane constitutive law (normal-tension)-(Damage)
        #------------------------------------------------------
        w_N = sctx[0]
        z_N = sctx[1]

        Z = lambda z_N: 1. / self.Ad * (-z_N) / (1 + z_N)
        Y = 0.5 * self.E * eps ** 2
        Y_0 = 0.5 * self.E * self.eps_0 ** 2
        f = Y - (Y_0 + Z(z_N))

        if f > 1e-6:
            f_w = lambda Y: 1 - 1. / (1 + self.Ad * (Y - Y_0))

            w_N = f_w(Y)
            z_N = - w_N
        new_sctx = zeros(2)
        new_sctx[0] = w_N
        new_sctx[1] = z_N
        return new_sctx

    def get_tangential_Law(self, e_T, sctx, sigma_kk):
        #-------------------------------------------------------------
        # microplane constitutive law (Tangential)-(Pressure sensitive
        # cumulative damage)
        #-------------------------------------------------------------
        G = self.E / (1 + 2.0 * self.nu)
        w_T = sctx[5]
        z_T = sctx[6]
        alpha_T = sctx[7:10]
        eps_T_pi = sctx[10:13]

        sig_pi_trial = G * (e_T - eps_T_pi)
        Z = self.K * z_T
        X = self.gamma * alpha_T
        f = norm(sig_pi_trial - X) - self.tau_pi_bar - \
            Z + self.a * sigma_kk / 3

        if f > 1e-6:
            delta_lamda = f / \
                (G / (1 - w_T) + self.gamma + self.K)
            eps_T_pi = eps_T_pi + delta_lamda * \
                ((sig_pi_trial - X) / (1 - w_T)) / norm(sig_pi_trial - X)
            Y = 0.5 * G * dot((e_T - eps_T_pi), (e_T - eps_T_pi))
            w_T += ((1 - w_T) ** self.c) * \
                (delta_lamda * (Y / self.S) ** self.r)
            # print 'w', w
            X = X + self.gamma * delta_lamda * \
                (sig_pi_trial - X) / norm(sig_pi_trial - X)
            alpha_T = alpha_T + delta_lamda * \
                (sig_pi_trial - X) / norm(sig_pi_trial - X)
            z_T = z_T + delta_lamda

        new_sctx = zeros(8)
        new_sctx[0:2] = w_T, z_T
        new_sctx[2:5] = alpha_T
        new_sctx[5:9] = eps_T_pi
        return new_sctx


class MATSXDMicroplaneDamageFatigueJir(MATSEvalMicroplaneFatigue):

    '''
    Microplane Damage Fatigue Model.
    '''
    #-------------------------------------------------------------------------
    # Configuration parameters
    #-------------------------------------------------------------------------

    model_version = Enum("compliance", "stiffness")

    symmetrization = Enum("product-type", "sum-type")

    regularization = Bool(False,
                          desc='Flag to use the element length projection'
                          ' in the direction of principle strains',
                          enter_set=True,
                          auto_set=False)

    elastic_debug = Bool(False,
                         desc='Switch to elastic behavior - used for debugging',
                         auto_set=False)

    double_constraint = Bool(False,
                             desc='Use double constraint to evaluate microplane elastic and fracture energy (Option effects only the response tracers)',
                             auto_set=False)

    #-------------------------------------------------------------------------
    # View specification
    #-------------------------------------------------------------------------

    config_param_vgroup = Group(Item('model_version', style='custom'),
                                #     Item('stress_state', style='custom'),
                                Item('symmetrization', style='custom'),
                                Item('elastic_debug@'),
                                Item('double_constraint@'),
                                Spring(resizable=True),
                                label='Configuration parameters',
                                show_border=True,
                                dock='tab',
                                id='ibvpy.mats.matsXD.MATSXD_cmdm.config',
                                )

    traits_view = View(Include('polar_fn_group'),
                       dock='tab',
                       id='ibvpy.mats.matsXD.MATSXD_cmdm',
                       kind='modal',
                       resizable=True,
                       scrollable=True,
                       width=0.6, height=0.8,
                       buttons=['OK', 'Cancel']
                       )

    #-------------------------------------------------------------------------
    # Setup for computation within a supplied spatial context
    #-------------------------------------------------------------------------
    D4_e = Property

    def _get_D4_e(self):
        # Return the elasticity tensor
        return self.elasticity_tensors

    #-------------------------------------------------------------------------
    # MICROPLANE-DISCRETIZATION RELATED METHOD
    #-------------------------------------------------------------------------

    # get the dyadic product of the microplane normals
    _MPNN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPNN(self):
        # dyadic product of the microplane normals

        MPNN_nij = einsum('ni,nj->nij', self._MPN, self._MPN)
        return MPNN_nij

    # get the third order tangential tensor (operator) for each microplane
    @cached_property
    def _get__MPTT(self):
        # Third order tangential tensor for each microplane
        delta = identity(3)
        MPTT_nijr = 0.5 * (einsum('ni,jr -> nijr', self._MPN, delta) +
                           einsum('nj,ir -> njir', self._MPN, delta) - 2 *
                           einsum('ni,nj,nr -> nijr', self._MPN, self._MPN, self._MPN))
        return MPTT_nijr

    def _get_e_vct_arr(self, eps_eng):
        # Projection of apparent strain onto the individual microplanes

        e_ni = einsum('nj,ji->ni', self._MPN, eps_eng)
        return e_ni

    def _get_e_N_arr(self, e_vct_arr):
        # get the normal strain array for each microplane

        eN_n = einsum('ni,ni->n', e_vct_arr, self._MPN)
        return eN_n

    def _get_e_T_vct_arr(self, e_vct_arr):
        # get the tangential strain vector array for each microplane
        eN_n = self._get_e_N_arr(e_vct_arr)

        eN_vct_ni = einsum('n,ni->ni', eN_n, self._MPN)

        return e_vct_arr - eN_vct_ni

    #-------------------------------------------------
    # Alternative methods for the kinematic constraint
    #-------------------------------------------------

    def _get_e_N_arr_2(self, eps_eng):

        #eps_mtx = self.map_eps_eng_to_mtx(eps_eng)
        return einsum('nij,ij->n', self._MPNN, eps_eng)

    def _get_e_T_vct_arr_2(self, eps_eng):

        #eps_mtx = self.map_eps_eng_to_mtx(eps_eng)
        MPTT_ijr = self._get__MPTT()
        return einsum('nijr,ij->nr', MPTT_ijr, eps_eng)

    def _get_e_vct_arr_2(self, eps_eng):

        return self._e_N_arr_2 * self._MPN + self._e_t_vct_arr_2

    def _get_state_variables(self, sctx, eps_app_eng, sigma_kk):
        #--------------------------------------------------------
        # return the state variables (Damage , inelastic strains)
        #--------------------------------------------------------
        e_N_arr = self._get_e_N_arr_2(eps_app_eng)
        e_T_vct_arr = self._get_e_T_vct_arr_2(eps_app_eng)

        sctx_arr = zeros((28, 13))

        for i in range(0, self.n_mp):

            if e_N_arr[i] > 0:
                sctx_N_ten = self.get_normal_tension_Law(
                    e_N_arr[i], sctx[i, :])
                sctx_N_comp = zeros(3)
            else:
                sctx_N_comp = self.get_normal_compression_Law(
                    e_N_arr[i], sctx[i, :])
                sctx_N_ten = zeros(2)

            sctx_tangential = self.get_tangential_Law(
                e_T_vct_arr[i, :], sctx[i, :], sigma_kk)

            sctx_arr[i, 0:2] = sctx_N_ten
            sctx_arr[i, 2:5] = sctx_N_comp
            sctx_arr[i, 5:13] = sctx_tangential

        return sctx_arr

    def _get_eps_N_p_arr(self, sctx, eps_app_eng, sigma_kk):
        #-----------------------------------------------------------------
        # Returns a list of the plastic normal strain  for all microplanes.
        #-----------------------------------------------------------------
        eps_N_p = self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 4]

        return eps_N_p

    def _get_eps_T_pi_arr(self, sctx, eps_app_eng, sigma_kk):
        #----------------------------------------------------------------
        # Returns a list of the sliding strain vector for all microplanes.
        #----------------------------------------------------------------

        eps_T_pi_vct_arr = self._get_state_variables(
            sctx, eps_app_eng, sigma_kk)[:, 10:13]

        return eps_T_pi_vct_arr

    '''
    #---------------------------------------------------------------------
    # Extra homogenization of damge in case of different damage parameters
    #---------------------------------------------------------------------
    
    def _get_beta_N_arr(self, sctx, eps_app_eng, sigma_kk):
       
        #Returns a list of the normal integrity factors for all microplanes.
        
        beta_N_arr = 1 - \
            self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 5]

        return beta_N_arr

    def _get_beta_T_arr(self, sctx, eps_app_eng, sigma_kk):
        
        #Returns a list of the tangential integrity factors for all microplanes.
       
        beta_T_arr = 1 - \
            self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 5]

        return beta_T_arr

    def _get_beta_tns(self, sctx, eps_app_eng, sigma_kk):
        
        #Returns the 4th order damage tensor 'beta4' using 
        #(cf. [Baz99], Eq.(63))
       
        delta = identity(3)
        beta_N_n = self._get_beta_N_arr(sctx, eps_app_eng, sigma_kk)
        beta_T_n = self._get_beta_T_arr(sctx, eps_app_eng, sigma_kk)

        beta_ijkl = einsum('n, n, ni, nj, nk, nl -> ijkl', self._MPW, beta_N_n, self._MPN, self._MPN, self._MPN, self._MPN ) + \
            0.25 * (einsum('n, n, ni, nk, jl -> ijkl', self._MPW, beta_T_n, self._MPN, self._MPN, delta) +
                    einsum('n, n, ni, nl, jk -> ijkl', self._MPW, beta_T_n, self._MPN, self._MPN, delta) +
                    einsum('n, n, nj, nk, il -> ijkl', self._MPW, beta_T_n, self._MPN, self._MPN, delta) +
                    einsum('n, n, nj, nl, ik -> ijkl', self._MPW, beta_T_n, self._MPN, self._MPN, delta) -
                    4 * einsum('n, n, ni, nj, nk, nl -> ijkl', self._MPW, beta_T_n, self._MPN, self._MPN, self._MPN, self._MPN))

        print 'beta_ijkl =', beta_ijkl
        return beta_ijkl
        '''

    def _get_phi_arr(self, sctx, eps_app_eng, sigma_kk):
        #-------------------------------------------------------------
        # Returns a list of the integrity factors for all microplanes.
        #-------------------------------------------------------------
        phi_arr = sqrt(1 -
                       self._get_state_variables(sctx, eps_app_eng, sigma_kk)[:, 5])

        return phi_arr

    def _get_phi_mtx(self, sctx, eps_app_eng, sigma_kk):
        #----------------------------------------------
        # Returns the 2nd order damage tensor 'phi_mtx'
        #----------------------------------------------

        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_app_eng, sigma_kk)

        # integration terms for each microplanes
        phi_ij = einsum('n,n,nij->ij', phi_arr, self._MPW, self._MPNN)

        return phi_ij

    def _get_beta_tns_sum_type(self, sctx, eps_app_eng, sigma_kk):
        #----------------------------------------------------------------------
        # Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
        #(cf. [Jir99], Eq.(21))
        #----------------------------------------------------------------------
        delta = identity(3)

        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)

        # use numpy functionality (einsum) to evaluate [Jir99], Eq.(21)
        beta_ijkl = 0.25 * (einsum('ik,jl->ijkl', phi_mtx, delta) +
                            einsum('il,jk->ijkl', phi_mtx, delta) +
                            einsum('jk,il->ijkl', phi_mtx, delta) +
                            einsum('jl,ik->ijkl', phi_mtx, delta))

        return beta_ijkl

    def _get_beta_tns_product_type(self, sctx, eps_app_eng, sigma_kk):
        #----------------------------------------------------------------------
        # Returns the 4th order damage tensor 'beta4' using product-type symmetrization
        #(cf. [Baz97], Eq.(87))
        #----------------------------------------------------------------------
        delta = identity(3)

        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng, sigma_kk)

        n_dim = 3
        phi_eig_value, phi_eig_mtx = eigh(phi_mtx)
        phi_eig_value_real = array([pe.real for pe in phi_eig_value])
        phi_pdc_mtx = zeros((n_dim, n_dim), dtype=float)
        for i in range(n_dim):
            phi_pdc_mtx[i, i] = phi_eig_value_real[i]
        # w_mtx = tensorial square root of the second order damage tensor:
        w_pdc_mtx = sqrt(phi_pdc_mtx)

        # transform the matrix w back to x-y-coordinates:
        w_mtx = einsum('ik,kl,lj -> ij', phi_eig_mtx, w_pdc_mtx, phi_eig_mtx)
        #w_mtx = dot(dot(phi_eig_mtx, w_pdc_mtx), transpose(phi_eig_mtx))

        beta_ijkl = 0.5 * \
            (einsum('ik,jl -> ijkl', w_mtx, w_mtx) +
             einsum('il,jk -> ijkl', w_mtx, w_mtx))

        return beta_ijkl

    def _get_eps_p_mtx(self, sctx, eps_app_eng, sigma_kk):
        #-----------------------------------------------------------
        # Integration of the (inelastic) strains for each microplane
        #-----------------------------------------------------------

        # plastic normal strains
        eps_N_P_n = self._get_eps_N_p_arr(sctx, eps_app_eng, sigma_kk)

        # sliding tangential strains
        eps_T_pi_ni = self._get_eps_T_pi_arr(sctx, eps_app_eng, sigma_kk)
        delta = identity(3)

        # 2-nd order plastic (inelastic) tensor
        eps_p_ij = einsum('n,n,ni,nj -> ij', self._MPW, eps_N_P_n, self._MPN, self._MPN) + \
            0.5 * (einsum('n,nr,ni,rj->ij', self._MPW, eps_T_pi_ni, self._MPN, delta) +
                   einsum('n,nr,nj,ri->ij', self._MPW, eps_T_pi_ni, self._MPN, delta))

        return eps_p_ij

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, sigma_kk):

        # Corrector predictor computation.

        # ----------------------------------------------------------------------------------------------
        # for debugging purposes only: if elastic_debug is switched on, linear elastic material is used
        # -----------------------------------------------------------------------------------------------
        if self.elastic_debug:
            # NOTE: This must be copied otherwise self.D2_e gets modified when
            # essential boundary conditions are inserted
            D2_e = copy(self.D2_e)
            sig_eng = tensordot(D2_e, eps_app_eng, [[1], [0]])
            return sig_eng, D2_e

        #----------------------------------------------------------------------
        # if the regularization using the crack-band concept is on calculate the
        # effective element length in the direction of principle strains
        #----------------------------------------------------------------------
        # if self.regularization:
        #    h = self.get_regularizing_length(sctx, eps_app_eng)
        #    self.phi_fn.h = h

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------
        beta_ijkl = self._get_beta_tns_sum_type(sctx, eps_app_eng, sigma_kk)

        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------

        D4_mdm_ijmn = einsum(
            'ijkl,klsr,mnsr->ijmn', beta_ijkl, self.D4_e, beta_ijkl)

        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------

        # plastic strain tensor
        eps_p_ij = self._get_eps_p_mtx(sctx, eps_app_eng, sigma_kk)

        # elastic strain tensor
        eps_e_mtx = eps_app_eng - eps_p_ij

        # calculation of the stress tensor
        sig_eng = einsum('ijmn,mn -> ij', D4_mdm_ijmn, eps_e_mtx)

        return sig_eng, D4_mdm_ijmn


class MATS3DMicroplaneDamageJir(MATSXDMicroplaneDamageFatigueJir, MATS3DEval):

    implements(IMATSEval)

    #-----------------------------------------------
    # number of microplanes - currently fixed for 3D
    #-----------------------------------------------
    n_mp = Constant(28)

    #-----------------------------------------------
    # get the normal vectors of the microplanes
    #-----------------------------------------------
    _MPN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPN(self):
        # microplane normals:
        return array([[.577350259, .577350259, .577350259],
                      [.577350259, .577350259, -.577350259],
                      [.577350259, -.577350259, .577350259],
                      [.577350259, -.577350259, -.577350259],
                      [.935113132, .250562787, .250562787],
                      [.935113132, .250562787, -.250562787],
                      [.935113132, -.250562787, .250562787],
                      [.935113132, -.250562787, -.250562787],
                      [.250562787, .935113132, .250562787],
                      [.250562787, .935113132, -.250562787],
                      [.250562787, -.935113132, .250562787],
                      [.250562787, -.935113132, -.250562787],
                      [.250562787, .250562787, .935113132],
                      [.250562787, .250562787, -.935113132],
                      [.250562787, -.250562787, .935113132],
                      [.250562787, -.250562787, -.935113132],
                      [.186156720, .694746614, .694746614],
                      [.186156720, .694746614, -.694746614],
                      [.186156720, -.694746614, .694746614],
                      [.186156720, -.694746614, -.694746614],
                      [.694746614, .186156720, .694746614],
                      [.694746614, .186156720, -.694746614],
                      [.694746614, -.186156720, .694746614],
                      [.694746614, -.186156720, -.694746614],
                      [.694746614, .694746614, .186156720],
                      [.694746614, .694746614, -.186156720],
                      [.694746614, -.694746614, .186156720],
                      [.694746614, -.694746614, -.186156720]])

    #-------------------------------------
    # get the weights of the microplanes
    #-------------------------------------
    _MPW = Property(depends_on='n_mp')

    @cached_property
    def _get__MPW(self):
        # Note that the values in the array must be multiplied by 6 (cf. [Baz05])!
        # The sum of of the array equals 0.5. (cf. [BazLuz04]))
        # The values are given for an Gaussian integration over the unit
        # hemisphere.
        return array([.0160714276, .0160714276, .0160714276, .0160714276, .0204744730,
                      .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                      .0204744730, .0204744730, .0204744730, .0204744730, .0204744730,
                      .0204744730, .0158350505, .0158350505, .0158350505, .0158350505,
                      .0158350505, .0158350505, .0158350505, .0158350505, .0158350505,
                      .0158350505, .0158350505, .0158350505]) * 6.0

    #-------------------------------------------------------------------------
    # Cached elasticity tensors
    #-------------------------------------------------------------------------

    elasticity_tensors = Property(
        depends_on='E, nu, dimensionality, stress_state')

    @cached_property
    def _get_elasticity_tensors(self):
        '''
        Intialize the fourth order elasticity tensor for 3D or 2D plane strain or 2D plane stress
        '''
        # ----------------------------------------------------------------------------
        # Lame constants calculated from E and nu
        # ----------------------------------------------------------------------------

        # first Lame paramter
        la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        # second Lame parameter (shear modulus)
        mu = self.E / (2 + 2 * self.nu)

        # -----------------------------------------------------------------------------------------------------
        # Get the fourth order elasticity and compliance tensors for the 3D-case
        # -----------------------------------------------------------------------------------------------------

        # construct the elasticity tensor (using Numpy - einsum function)
        delta = identity(3)
        D_ijkl = (einsum(',ij,kl->ijkl', la, delta, delta) +
                  einsum(',ik,jl->ijkl', mu, delta, delta) +
                  einsum(',il,jk->ijkl', mu, delta, delta))

        return D_ijkl

    #-------------------------------------------------------------------------
    # Dock-based view with its own id
    #-------------------------------------------------------------------------

    traits_view = View(Include('polar_fn_group'),
                       dock='tab',
                       id='ibvpy.mats.mats3D.mats_3D_cmdm.MATS3D_cmdm',
                       kind='modal',
                       resizable=True,
                       scrollable=True,
                       width=0.6, height=0.8,
                       buttons=['OK', 'Cancel']
                       )


if __name__ == '__main__':

    # Check the model behavior
    n = 100
    s_levels = linspace(0, -0.1, 2)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= -1
    #s_levels[0] = 0
    #s_levels.reshape(-1, 2)[:, 1] = -0.006
    s_history = s_levels.flatten()

    # cyclic loading
    s_history_2 = [-0, -0.01, -0.007, -0.015, -0.011, -0.023, -0.017, -0.031, -.023, -
                   0.04, -0.032, -0.05, -0.041, -0.06, -0.050, -.07, -.058, -.08, -0.066, -.09, -0.075, - 0.1]

    s_arr_1 = hstack([linspace(s_history_2[i], s_history_2[i + 1], n)
                      for i in range(len(s_history_2) - 1)])
    s_arr_2 = hstack([linspace(s_history[i], s_history[i + 1], 500)
                      for i in range(len(s_levels) - 1)])

    eps_1 = array([array([[s_arr_1[i], 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]) for i in range(0, len(s_arr_1))])

    eps_2 = array([array([[s_arr_2[i], 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]) for i in range(0, len(s_arr_2))])

    m1 = MATSEvalMicroplaneFatigue()
    m2 = MATS3DMicroplaneDamageJir()
    sigma_1 = zeros_like(eps_1)
    sigma_kk_1 = zeros(len(s_arr_1) + 1)
    w_1 = zeros((len(eps_1[:, 0, 0]), 28))
    eps_P_N_1 = zeros((len(eps_1[:, 0, 0]), 28))
    eps_Pi_T_1 = zeros((len(eps_1[:, 0, 0]), 28, 3))
    e_1 = zeros((len(eps_1[:, 0, 0]), 28, 3))
    e_T_1 = zeros((len(eps_1[:, 0, 0]), 28, 3))
    e_N_1 = zeros((len(eps_1[:, 0, 0]), 28))
    sctx_1 = zeros((len(eps_1[:, 0, 0]) + 1, 28, 13))

    sigma_2 = zeros_like(eps_2)
    sigma_kk_2 = zeros(len(s_arr_2) + 1)
    w_2 = zeros((len(eps_2[:, 0, 0]), 28))
    eps_P_N_2 = zeros((len(eps_2[:, 0, 0]), 28))
    eps_Pi_T_2 = zeros((len(eps_2[:, 0, 0]), 28, 3))
    e_2 = zeros((len(eps_2[:, 0, 0]), 28, 3))
    e_T_2 = zeros((len(eps_2[:, 0, 0]), 28, 3))
    e_N_2 = zeros((len(eps_2[:, 0, 0]), 28))
    sctx_2 = zeros((len(eps_2[:, 0, 0]) + 1, 28, 13))

    for i in range(0, len(eps_1[:, 0, 0])):

        sigma_1[i, :] = m2.get_corr_pred(
            sctx_1[i, :], eps_1[i, :], sigma_kk_1[i])[0]
        sigma_kk_1[i + 1] = trace(sigma_1[i, :])
        sctx_1[
            i + 1] = m2._get_state_variables(sctx_1[i, :], eps_1[i, :], sigma_kk_1[i])
        w_1[i, :] = sctx_1[i, :, 5]
        eps_P_N_1[i, :] = sctx_1[i, :, 4]
        eps_Pi_T_1[i, :, :] = sctx_1[i, :, 10:13]
        e_1[i, :] = m2._get_e_vct_arr(eps_1[i, :])
        e_T_1[i, :] = m2._get_e_T_vct_arr_2(eps_1[i, :])
        e_N_1[i, :] = m2._get_e_N_arr(e_1[i, :])

    for i in range(0, len(eps_2[:, 0, 0])):

        sigma_2[i, :] = m2.get_corr_pred(
            sctx_2[i, :], eps_2[i, :], sigma_kk_2[i])[0]
        sigma_kk_2[i + 1] = trace(sigma_2[i, :])
        sctx_2[
            i + 1] = m2._get_state_variables(sctx_2[i, :], eps_2[i, :], sigma_kk_2[i])
        w_2[i, :] = sctx_2[i, :, 5]
        eps_P_N_2[i, :] = sctx_2[i, :, 4]
        eps_Pi_T_2[i, :, :] = sctx_2[i, :, 10:13]
        e_2[i, :] = m2._get_e_vct_arr(eps_2[i, :])
        e_T_2[i, :] = m2._get_e_T_vct_arr_2(eps_2[i, :])
        e_N_2[i, :] = m2._get_e_N_arr(e_2[i, :])

    plt.subplot(221)
    plt.plot(eps_1[:, 0, 0], sigma_1[:, 0, 0],
             linewidth=1, label='sigma_11_(cyclic)')
    #plt.plot(eps_1[:, 0, 0], sigma_1[:, 1, 1], linewidth=1, label='sigma_22')
    #plt.plot(eps[:, 0, 0], sigma[:, 0, 1], linewidth=1, label='sigma_12')
    plt.plot(eps_2[:, 0, 0], sigma_2[:, 0, 0],
             linewidth=1, label='sigma_11_(monotonic)')
    plt.xlabel('Strain')
    plt.ylabel('Stress(MPa)')
    plt.legend()

    plt.subplot(222)
    for i in range(0, 28):
        plt.plot(
            eps_1[:, 0, 0], w_1[:, i], linewidth=1.0, label='cyclic', alpha=1)
        plt.plot(
            eps_2[:, 0, 0], w_2[:, i], linewidth=1.0, label='monotonic', alpha=1)

        plt.xlabel('Strain')
        plt.ylabel('Damage')
        # plt.legend()

    plt.subplot(223)
    for i in range(0, 28):
        plt.plot(
            eps_1[:, 0, 0], e_N_1[:, i], linewidth=1, label='plastic_strain')
        plt.plot(
            eps_2[:, 0, 0], e_N_2[:, i], linewidth=1, label='plastic_strain')

        plt.xlabel('Strain')
        plt.ylabel('normal_strain')
        # plt.legend()

    plt.subplot(224)
    for i in range(0, 28):
        plt.plot(
            eps_1[:, 0, 0], eps_Pi_T_1[:, i, 1], linewidth=1, label='sliding strain')
        plt.plot(
            eps_2[:, 0, 0], eps_Pi_T_2[:, i, 1], linewidth=1, label='sliding strain')

        plt.xlabel('Strain')
        plt.ylabel('sliding_strain')

    plt.show()
