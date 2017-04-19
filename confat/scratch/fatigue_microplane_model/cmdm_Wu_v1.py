'''
Created on 24.03.2017

@author: abaktheer

New implementation of microplane model - (Wu)
'''

from ibvpy.core.rtrace_eval import \
    RTraceEval
from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats_eval import \
    IMATSEval
from numpy import \
    array, zeros, outer, inner, transpose, dot, trace, \
    fabs, identity, tensordot, einsum, zeros_like,\
    float_, identity, sign, fabs, linspace, hstack, \
    sqrt, exp, copy
from numpy.linalg import norm
from scipy.linalg import \
    eigh, inv
from traits.api import \
    Constant,  Property, cached_property, implements,\
    Bool, Callable, Enum, Float, HasTraits, \
    Int, Trait, on_trait_change, \
    Dict, Property, cached_property
from traitsui.api import \
    Item, View, Group, Spring, Include

import matplotlib.pyplot as plt


class MATSEvalMicroplaneFatigue(HasTraits):

    E = Float(34000,
              label="E",
              desc="Elastic modulus",
              enter_set=True,
              auto_set=False)

    nu = Float(0.2,
               label="nu",
               desc="poisson's ratio",
               enter_set=True,
               auto_set=False)

    ep = Float(59e-6,
               label="ep",
               desc="",
               enter_set=True,
               auto_set=False)

    ef = Float(150e-6,
               label="ef",
               desc="",
               enter_set=True,
               auto_set=False)

    c_T = Float(1.0,
                label="c_T",
                desc="",
                enter_set=True,
                auto_set=False)

    zeta_G = Float(1.0,
                   label="zeta_G",
                   desc="",
                   enter_set=True,
                   auto_set=False)

    def _get_phi(self, e, sctx):
        #----------------------------------------------
        # constitutive law - damage model
        #----------------------------------------------

        phi = zeros(len(e))
        d = zeros(len(e))

#         k_0 = 0.000038
#         alpha = 1
#         beta = 8000

        for i in range(0, len(e)):

            if e[i] >= self.ep:
                phi[i] = sqrt(
                    (self.ep / e[i]) * exp(- (e[i] - self.ep) / (self.ef)))

            else:
                phi[i] = 1

#             if e[i] >= k_0:
#                 d[i] = 1 - (k_0 / e[i]) * \
#                     (1 - alpha + alpha * exp(beta * (k_0 - e[i])))
#             else:
#                 d[i] = 0

        return phi, d


class MATSXDMicroplaneDamageFatigueWu(MATSEvalMicroplaneFatigue):

    '''
    Microplane Damage Model.
    '''

    # specification of the model dimension (2D, 3D)
    n_dim = Int

    # specification of number of engineering strain and stress components
    n_eng = Int

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
        return self.elasticity_tensors[0]

    #-------------------------------------------------------------------------
    # MICROPLANE-DISCRETIZATION RELATED METHOD
    #-------------------------------------------------------------------------

    # get the dyadic product of the microplane normals
    _MPNN = Property(depends_on='n_mp')

    @cached_property
    def _get__MPNN(self):
        # dyadic product of the microplane normals
        # return array([outer(mpn, mpn) for mpn in self._MPN]) # old
        # implementation
        # n identify the microplane
        MPNN_nij = einsum('ni,nj->nij', self._MPN, self._MPN)

        return MPNN_nij

    # get Third order tangential tensor (operator) for each microplane
    _MPTT = Property(depends_on='n_mp')

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

    # Alternative methods for the kinematic constraint

    def _get_e_N_arr_2(self, eps_eng):

        #eps_mtx = self.map_eps_eng_to_mtx(eps_eng)
        return einsum('nij,ij->n', self._MPNN, eps_eng)

    def _get_e_t_vct_arr_2(self, eps_eng):

        #eps_mtx = self.map_eps_eng_to_mtx(eps_eng)
        MPTT_ijr = self._get__MPTT()
        return einsum('nijr,ij->nr', MPTT_ijr, eps_eng)

    def _get_e_vct_arr_2(self, eps_eng):

        return self._e_N_arr_2 * self._MPN + self._e_t_vct_arr_2

    def _get_I_vol_4(self):
        # The fourth order volumetric-identity tensor
        delta = identity(3)
        I_vol_ijkl = (1.0 / 3.0) * einsum('ij,kl -> ijkl', delta, delta)
        return I_vol_ijkl

    def _get_I_dev_4(self):
        # The fourth order deviatoric-identity tensor
        delta = identity(3)
        I_dev_ijkl = 0.5 * (einsum('ik,jl -> ijkl', delta, delta) +
                            einsum('il,jk -> ijkl', delta, delta)) \
            - (1 / 3.0) * einsum('ij,kl -> ijkl', delta, delta)

        return I_dev_ijkl

    def _get_P_vol(self):
        delta = identity(3)
        P_vol_ij = (1 / 3.0) * delta
        return P_vol_ij

    def _get_P_dev(self):
        delta = identity(3)
        P_dev_njkl = 0.5 * einsum('ni,ij,kl -> njkl', self._MPN, delta, delta)
        return P_dev_njkl

    def _get_PP_vol_4(self):
        # outer product of P_vol
        delta = identity(3)
        PP_vol_ijkl = (1 / 9.) * einsum('ij,kl -> ijkl', delta, delta)
        return PP_vol_ijkl

    def _get_PP_dev_4(self):
        # inner product of P_dev
        delta = identity(3)
        PP_dev_nijkl = 0.5 * (0.5 * (einsum('ni,nk,jl -> nijkl', self._MPN, self._MPN, delta) +
                                     einsum('ni,nl,jk -> nijkl', self._MPN, self._MPN, delta)) +
                              0.5 * (einsum('ik,nj,nl -> nijkl',  delta, self._MPN, self._MPN) +
                                     einsum('il,nj,nk -> nijkl',  delta, self._MPN, self._MPN))) -\
            (1 / 3.) * (einsum('ni,nj,kl -> nijkl', self._MPN, self._MPN, delta) +
                        einsum('ij,nk,nl -> nijkl', delta, self._MPN, self._MPN)) +\
            (1 / 9.) * einsum('ij,kl -> ijkl', delta, delta)

        return PP_dev_nijkl

    def _get_e_equiv_arr(self, e_vct_arr):
        '''
        Returns a list of the microplane equivalent strains
        based on the list of microplane strain vectors
        '''
        # magnitude of the normal strain vector for each microplane
        # @todo: faster numpy functionality possible?
        e_N_arr = self._get_e_N_arr(e_vct_arr)
        # positive part of the normal strain magnitude for each microplane
        e_N_pos_arr = (fabs(e_N_arr) + e_N_arr) / 2
        # normal strain vector for each microplane
        # @todo: faster numpy functionality possible?
        e_N_vct_arr = einsum('n,ni -> ni', e_N_arr, self._MPN)
        # tangent strain ratio
        c_T = self.c_T
        # tangential strain vector for each microplane
        e_T_vct_arr = e_vct_arr - e_N_vct_arr
        # squared tangential strain vector for each microplane
        e_TT_arr = einsum('ni,ni -> n', e_T_vct_arr, e_T_vct_arr)
        # equivalent strain for each microplane
        e_equiv_arr = sqrt(e_N_pos_arr * e_N_pos_arr + c_T * e_TT_arr)
        return e_equiv_arr

    def _get_e_max(self, e_equiv_arr, e_max_arr):
        '''
        Compares the equivalent microplane strain of a single microplane with the
        maximum strain reached in the loading history for the entire array
        '''
        bool_e_max = e_equiv_arr >= e_max_arr

        # [rch] fixed a bug here - this call was modifying the state array
        # at any invocation.
        #
        # The new value must be created, otherwise side-effect could occur
        # by writing into a state array.
        #
        new_e_max_arr = copy(e_max_arr)
        new_e_max_arr[bool_e_max] = e_equiv_arr[bool_e_max]
        return new_e_max_arr

    def _get_state_variables(self, sctx, eps_app_eng):
        '''
        Compares the list of current equivalent microplane strains with
        the values in the state array and returns the higher values
        '''
        e_vct_arr = self._get_e_vct_arr(eps_app_eng)
        e_equiv_arr = self._get_e_equiv_arr(e_vct_arr)
        #e_max_arr_old = sctx.mats_state_array
        #e_max_arr_new = self._get_e_max(e_equiv_arr, e_max_arr_old)
        return e_equiv_arr

    def _get_phi_arr(self, sctx, eps_app_eng):
        # Returns a list of the integrity factors for all microplanes.
        e_max_arr = self._get_state_variables(sctx, eps_app_eng)

        phi_arr = self._get_phi(e_max_arr, sctx)[0]

        return phi_arr

    def _get_phi_mtx(self, sctx, eps_app_eng):
        # Returns the 2nd order damage tensor 'phi_mtx'

        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_app_eng)

        # integration terms for each microplanes
        phi_ij = einsum('n,n,nij->ij', phi_arr, self._MPW, self._MPNN)

        # print 'phi_ij', phi_ij

        return phi_ij

    def _get_d_scalar(self, sctx, eps_app_eng):

        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_app_eng)

        d_arr = 1 - phi_arr

        d = (1.0 / 3.0) * einsum('n,n->',  d_arr, self._MPW)

        print d

        return d

    def _get_M_vol_tns(self, sctx, eps_app_eng):

        d = self._get_d_scalar(sctx, eps_app_eng)
        delta = identity(3)

        I_4th_ijkl = einsum('ik,jl -> ijkl', delta, delta)

        return (1 - d) * I_4th_ijkl

    def _get_M_dev_tns(self, phi_mtx):
        '''
        Returns the 4th order deviatoric damage tensor
        '''
        delta = identity(3)

        # use numpy functionality (einsum) to evaluate [Jir99], Eq.(21)
        # M_dev_ijkl = 0.25 * (einsum('ik,jl->ijkl', phi_mtx, delta) +
        #                    einsum('il,jk->ijkl', phi_mtx, delta) +
        #                     einsum('jk,il->ijkl', phi_mtx, delta) +
        #                     einsum('jl,ik->ijkl', phi_mtx, delta))

        M_dev_ijkl = 0.5 * (0.5 * (einsum('ik,jl->ijkl', delta, phi_mtx) +
                                   einsum('il,jk->ijkl', delta, phi_mtx)) +
                            0.5 * (einsum('ik,jl->ijkl', phi_mtx, delta) +
                                   einsum('il,jk->ijkl', phi_mtx, delta)))

        # print 'M_dev_ijkl', M_dev_ijkl

        return M_dev_ijkl

    #-------------------------------------------------------------------------
    # Secant stiffness (irreducible decomposition based on ODFs)
    #-------------------------------------------------------------------------

    def _get_S_1_tns(self, sctx, eps_app_eng):
        #----------------------------------------------------------------------
        # Returns the fourth order secant stiffness tensor (eq.1)
        #----------------------------------------------------------------------
        K0 = self.E / (1. - 2. * self.nu)
        G0 = self.E / (1. + self.nu)

        e_max_arr = self._get_state_variables(sctx, eps_app_eng)
        d_n = self._get_phi(e_max_arr, sctx)[0]

        PP_vol_4 = self._get_PP_vol_4()
        PP_dev_4 = self._get_PP_dev_4()
        delta = identity(3)
        I_4th_ijkl = einsum('ik,jl -> ijkl', delta, delta)
        I_dev_4 = self._get_I_dev_4()

        S_1_ijkl = K0 * einsum('n,n,ijkl->ijkl', d_n, self._MPW, PP_vol_4) + \
            G0 * 2 * self.zeta_G * einsum('n,n,nijkl->ijkl', d_n, self._MPW, PP_dev_4) - (1. / 3.) * (
                2 * self.zeta_G - 1) * G0 * einsum('n,n,ijkl->ijkl', d_n, self._MPW, I_dev_4)

        return S_1_ijkl

    def _get_S_2_tns(self, sctx, eps_app_eng):
        #----------------------------------------------------------------------
        # Returns the fourth order secant stiffness tensor (eq.2)
        #----------------------------------------------------------------------
        K0 = self.E / (1. - 2. * self.nu)
        G0 = self.E / (1. + self.nu)

        I_vol_ijkl = self._get_I_vol_4()
        I_dev_ijkl = self._get_I_dev_4()
        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng)
        M_vol_ijkl = self._get_M_vol_tns(sctx, eps_app_eng)
        M_dev_ijkl = self._get_M_dev_tns(phi_mtx)

        S_2_ijkl = K0 * einsum('ijmn,mnrs,rskl -> ijkl', I_vol_ijkl, M_vol_ijkl, I_vol_ijkl ) \
            + G0 * einsum('ijmn,mnrs,rskl -> ijkl', I_dev_ijkl, M_dev_ijkl, I_dev_ijkl)\


        print 'S_vol = ', K0 * einsum('ijmn,mnrs,rskl -> ijkl', I_vol_ijkl, M_vol_ijkl, I_vol_ijkl)
        print 'S_dev = ', G0 * einsum('ijmn,mnrs,rskl -> ijkl', I_dev_ijkl, M_dev_ijkl, I_dev_ijkl)

        return S_2_ijkl

    def _get_S_3_tns(self, sctx, eps_app_eng):
        #----------------------------------------------------------------------
        # Returns the fourth order secant stiffness tensor (eq.3)
        #----------------------------------------------------------------------
        K0 = self.E / (1. - 2. * self.nu)
        G0 = self.E / (1. + self.nu)

        I_vol_ijkl = self._get_I_vol_4()
        I_dev_ijkl = self._get_I_dev_4()

        S_0_ijkl = K0 * I_vol_ijkl + G0 * I_dev_ijkl

        e_max_arr = self._get_state_variables(sctx, eps_app_eng)

        d_n = 1 - self._get_phi(e_max_arr, sctx)[0]

        PP_vol_4 = self._get_PP_vol_4()
        PP_dev_4 = self._get_PP_dev_4()

        delta = identity(3)
        I_4th_ijkl = einsum('ik,jl -> ijkl', delta, delta)

        D_ijkl = einsum('n,n,ijkl->ijkl', d_n, self._MPW, PP_vol_4) + \
            einsum('n,n,nijkl->ijkl', d_n, self._MPW, PP_dev_4)

        phi_ijkl = (I_4th_ijkl - D_ijkl)

        S_ijkl = einsum('ijmn,mnkl', phi_ijkl, S_0_ijkl)

        return S_ijkl

    def _get_S_4_tns(self, sctx, eps_app_eng):
        #----------------------------------------------------------------------
        # Returns the fourth order secant stiffness tensor (double orthotropic)
        #----------------------------------------------------------------------

        K0 = self.E / (1. - 2. * self.nu)
        G0 = self.E / (1. + self.nu)

        I_vol_ijkl = self._get_I_vol_4()
        I_dev_ijkl = self._get_I_dev_4()
        delta = identity(3)
        phi_mtx = self._get_phi_mtx(sctx, eps_app_eng)
        D_ij = delta - phi_mtx
        d = (1. / 3.) * trace(D_ij)
        D_bar_ij = self.zeta_G * (D_ij - d * delta)

        S_4_ijkl = (1 - d) * K0 * I_vol_ijkl + (1 - d) * G0 * I_dev_ijkl + (2 / 3.) * (G0 - K0) * \
            (einsum('ij,kl -> ijkl', delta, D_bar_ij) +
             einsum('ij,kl -> ijkl', D_bar_ij, delta)) + 0.5 * (K0 - 2 * G0) * \
            (0.5 * (einsum('ik,jl -> ijkl', delta, D_bar_ij) + einsum('il,jk -> ijkl', D_bar_ij, delta)) +
             0.5 * (einsum('il,jk -> ijkl', D_bar_ij, delta) + einsum('ik,jl -> ijkl', delta, D_bar_ij)))

        return S_4_ijkl

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''

        # -----------------------------------------------------------------------------------------------
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

        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------
        eps_e_mtx = eps_app_eng

        S_ijkl = self._get_S_4_tns(sctx, eps_app_eng)[:]
        sig_ij = einsum('ijkl,kl -> ij', S_ijkl, eps_e_mtx)

        return sig_ij


class MATS3DMicroplaneDamageWu(MATSXDMicroplaneDamageFatigueWu, MATS3DEval):

    implements(IMATSEval)
    # number of spatial dimensions
    #
    n_dim = Constant(3)

    # number of components of engineering tensor representation
    #
    n_eng = Constant(6)

    #-------------------------------------------------------------------------
    # PolarDiscr related data
    #-------------------------------------------------------------------------
    #
    # number of microplanes - currently fixed for 3D
    #
    n_mp = Constant(28)

    # get the normal vectors of the microplanes
    #
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

    # get the weights of the microplanes
    #
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

    n = 100
    s_levels = linspace(0, 0.005, 2)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= -1
    #s_levels.reshape(-1, 2)[:, 1] *= -1
    s_history = s_levels.flatten()
    s_arr = hstack([linspace(s_history[i], s_history[i + 1], n)
                    for i in range(len(s_levels) - 1)])

    m = 0.0
    n = 0.0
    eps = array([array([[s_arr[i], m * s_arr[i], 0],
                        [m * s_arr[i], n * s_arr[i], 0],
                        [0, 0, 0]]) for i in range(0, len(s_arr))])

    m2 = MATS3DMicroplaneDamageWu()
    sigma_Wu = zeros_like(eps)
    sigma_Jir = zeros_like(eps)
    sigma_kk = zeros(len(s_arr) + 1)
    w = zeros((len(eps[:, 0, 0]), 28))
    e_pi = zeros((len(eps[:, 0, 0]), 28))
    e = zeros((len(eps[:, 0, 0]), 28, 3))
    e_T = zeros((len(eps[:, 0, 0]), 28, 3))
    e_N = zeros((len(eps[:, 0, 0]), 28))
    sctx = zeros((len(eps[:, 0, 0]) + 1, 28))

    for i in range(0, len(eps[:, 0, 0])):

        sigma_Wu[i, :] = m2.get_corr_pred(sctx[i, :], eps[i, :])
        sctx[i + 1] = m2._get_state_variables(sctx[i, :], eps[i, :])
        w[i, :] = 1 - m2._get_phi_arr(sctx[i], eps[i, :])
        #e_pi[i, :] = sctx[i, :, 5]
        #e[i, :] = m2._get_e_vct_arr(eps[i, :])
        #e_T[i, :] = m2._get_e_t_vct_arr_2(eps[i, :])
        #e_N[i, :] = m2._get_e_N_arr(e[i, :])

    plt.subplot(221)
    plt.plot(eps[:, 0, 0], sigma_Wu[:, 0, 0], 'k--',
             linewidth=1, label='sigma_11_[Wu]')
    plt.plot(eps[:, 0, 0], sigma_Wu[:, 1, 1], 'k',
             linewidth=1, label='sigma_22_[Wu]')
#     plt.plot(eps[:, 0, 0], sigma_Jir[:, 0, 0], 'r--',
#              linewidth=1, label='sigma_11_[Jir]')
#     plt.plot(eps[:, 0, 0], sigma_Jir[:, 1, 1], 'r',
#              linewidth=1, label='sigma_22_[Jir]')
    plt.xlabel('Strain')
    plt.ylabel('Stress(MPa)')
    plt.legend()

    plt.subplot(222)
    for i in range(0, 28):
        plt.plot(
            eps[:, 0, 0], w[:, i], linewidth=1, label='Damage of the microplanes', alpha=1)

        plt.xlabel('Strain')
        plt.ylabel('Damage of the microplanes')
        # plt.legend()

    plt.subplot(223)
    plt.plot(eps[:, 0, 0], sigma_Wu[:, 1, 1],
             linewidth=1, label='sigma_22_[Wu]')
#     plt.plot(eps[:, 0, 0], sigma_Jir[:, 1, 1],
#              linewidth=1, label='sigma_22_[Jir]')
    plt.xlabel('Strain')
    plt.ylabel('Stress(MPa)')
    plt.legend()


#     plt.subplot(223)
#     for i in range(0, 28):
#         plt.plot(
#             eps[:, 0, 0], e_T[:, i, 0], linewidth=1, label='Tangential_strain')
#
#         plt.xlabel('Strain')
#         plt.ylabel('Tangential_strain')
#         # plt.legend()
#
#     plt.subplot(224)
#     for i in range(0, 28):
#         plt.plot(
#             eps[:, 0, 0], e_N[:, i], linewidth=1, label='Normal_strain')
#
#         plt.xlabel('Strain')
#         plt.ylabel('Normal_strain')

    plt.show()
