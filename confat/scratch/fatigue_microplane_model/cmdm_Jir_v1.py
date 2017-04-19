'''
Created on 31.03.2017

@author: abaktheer
'''

'''
Microplane damage model - Jirasek
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
              label="G",
              desc="Elastic modulus",
              enter_set=True,
              auto_set=False)

    poisson = Float(0.2,
                    label="G",
                    desc="poisson's ratio",
                    enter_set=True,
                    auto_set=False)

    ep = Float(59e-6,
               label="a",
               desc="Lateral pressure coefficient",
               enter_set=True,
               auto_set=False)

    ef = Float(150e-6,
               label="a",
               desc="Lateral pressure coefficient",
               enter_set=True,
               auto_set=False)

    c_T = Float(1.0,
                label="a",
                desc="Lateral pressure coefficient",
                enter_set=True,
                auto_set=False)

    def _get_phi(self, e, sctx):

        phi = zeros(len(e))

        for i in range(0, len(e)):

            if e[i] >= self.ep:
                phi[i] = sqrt(
                    self.ep / e[i] * exp(- (e[i] - self.ep) / (self.ef)))
            else:
                phi[i] = 1

        return phi


class MATSXDMicroplaneDamageFatigueJir(MATSEvalMicroplaneFatigue):

    '''
    Microplane Damage Fatigue Model.
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
        return self.elasticity_tensors

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

        phi_arr = sqrt(self._get_phi(e_max_arr, sctx)[:])

        # print 'phi_arr', phi_arr

        return phi_arr

    def _get_phi_mtx(self, sctx, eps_app_eng):
        # Returns the 2nd order damage tensor 'phi_mtx'

        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_app_eng)

        # integration terms for each microplanes
        phi_ij = einsum('n,n,nij->ij', phi_arr, self._MPW, self._MPNN)

        print 'phi_ij', phi_ij

        return phi_ij

    def _get_beta_tns(self, phi_mtx):
        '''
        Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
        (cf. [Jir99], Eq.(21))
        '''
        n_dim = self.n_dim
        delta = identity(3)

        # The following line correspond to the tensorial expression:
        #
        #        beta4 = zeros((n_dim,n_dim,n_dim,n_dim),dtype=float)
        #        for i in range(0,n_dim):
        #            for j in range(0,n_dim):
        #                for k in range(0,n_dim):
        #                    for l in range(0,n_dim):
        #                        beta4[i,j,k,l] = 0.25 * ( phi_mtx[i,k] * delta[j,l] + phi_mtx[i,l] * delta[j,k] +\
        #                                                  phi_mtx[j,k] * delta[i,l] + phi_mtx[j,l] * delta[i,k] )
        #

        # use numpy functionality (swapaxes) to evaluate [Jir99], Eq.(21)
        #beta_ijkl = outer(phi_mtx, delta).reshape(n_dim, n_dim, n_dim, n_dim)
        #beta_ikjl = beta_ijkl.swapaxes(1, 2)
        #beta_iljk = beta_ikjl.swapaxes(2, 3)
        #beta_jlik = beta_iljk.swapaxes(0, 1)
        #beta_jkil = beta_jlik.swapaxes(2, 3)
        #beta4 = 0.25 * (beta_ikjl + beta_iljk + beta_jkil + beta_jlik)

        # use numpy functionality (einsum) to evaluate [Jir99], Eq.(21)
        beta_ijkl = 0.25 * (einsum('ik,jl->ijkl', phi_mtx, delta) +
                            einsum('il,jk->ijkl', phi_mtx, delta) +
                            einsum('jk,il->ijkl', phi_mtx, delta) +
                            einsum('jl,ik->ijkl', phi_mtx, delta))

        return beta_ijkl

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        # -----------------------------------------------------------------------------------------------
        # check if average strain is to be used for damage evaluation
        # -----------------------------------------------------------------------------------------------
        # if eps_avg != None:
        #    pass
        # else:
        #    eps_avg = eps_app_eng

        # -----------------------------------------------------------------------------------------------
        # for debugging purposes only: if elastic_debug is switched on, linear elastic material is used
        # -----------------------------------------------------------------------------------------------
        if self.elastic_debug:
            # NOTE: This must be copied otherwise self.D2_e gets modified when
            # essential boundary conditions are inserted
            D2_e = copy(self.D2_e)
            sig_eng = tensordot(D2_e, eps_app_eng, [[1], [0]])
            return sig_eng, D2_e

        # print 'sctx_correc', sctx.shape
        # -----------------------------------------------------------------------------------------------
        # update state variables
        # -----------------------------------------------------------------------------------------------
        # if sctx.update_state_on:
        #    #eps_n = eps_avg - d_eps
        #   e_max = self._get_state_variables(sctx, eps_app_eng)
        #    sctx.mats_state_array[:] = e_max

        #----------------------------------------------------------------------
        # if the regularization using the crack-band concept is on calculate the
        # effective element length in the direction of principle strains
        #----------------------------------------------------------------------
        # if self.regularization:
        #    h = self.get_regularizing_length(sctx, eps_app_eng)
        #    self.phi_fn.h = h

        #------------------------------------------------------------------
        # Damage tensor (2th order):
        #------------------------------------------------------------------

        phi_ij = self._get_phi_mtx(sctx, eps_app_eng)

        # print 'phi_ij', phi_ij

        #------------------------------------------------------------------
        # Damage tensor (4th order) using product- or sum-type symmetrization:
        #------------------------------------------------------------------
        beta_ijkl = self._get_beta_tns(phi_ij)

        # print 'beta_ijkl', beta_ijkl

        # print 'beta_ijkl', beta_ijkl

        #------------------------------------------------------------------
        # Damaged stiffness tensor calculated based on the damage tensor beta4:
        #------------------------------------------------------------------
        # (cf. [Jir99] Eq.(7): C = beta * D_e * beta^T),
        # minor symmetry is tacitly assumed ! (i.e. beta_ijkl = beta_jilk)
        # D4_mdm = tensordot(
        # beta_ijkl, tensordot(self.D4_e, beta_ijkl, [[2, 3], [2, 3]]),
        # [[2, 3], [0, 1]])

        D4_mdm_ijmn = einsum(
            'ijkl,klsr,mnsr->ijmn', beta_ijkl, self.D4_e, beta_ijkl)

        # print self.D4_e

        # print 'D4_mdm_ijmn', D4_mdm_ijmn
        # print 'Damged stiffness', D4_mdm_ijmn
        # print self.D4_e
        #------------------------------------------------------------------
        # Reduction of the fourth order tensor to a matrix assuming minor and major symmetry:
        #------------------------------------------------------------------
        #D2_mdm = self.map_tns4_to_tns2(D4_mdm_ijmn)

        # print'D2_mdm', D2_mdm.shape

        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------

        eps_e_mtx = eps_app_eng
        # print'eps_e_mtx', eps_e_mtx.shape

        #sig_eng = tensordot(D2_mdm, eps_e_mtx, [[1], [0]])
        sig_eng = einsum('ijmn,mn -> ij', D4_mdm_ijmn, eps_e_mtx)

        # print 'sig_eng', sig_eng

        return sig_eng, D4_mdm_ijmn


class MATS3DMicroplaneDamageJir(MATSXDMicroplaneDamageFatigueJir, MATS3DEval):

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
        #E = self.E
        #nu = self.nu

        E = self.E
        nu = self.poisson
        # first Lame paramter
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        # second Lame parameter (shear modulus)
        mu = E / (2 + 2 * nu)

        # -----------------------------------------------------------------------------------------------------
        # Get the fourth order elasticity and compliance tensors for the 3D-case
        # -----------------------------------------------------------------------------------------------------

        # The following line correspond to the tensorial expression:
        # (using numpy functionality in order to avoid the loop):
        #
        # D4_e_3D = zeros((3,3,3,3),dtype=float)
        # C4_e_3D = zeros((3,3,3,3),dtype=float)
        # delta = identity(3)
        # for i in range(0,3):
        #     for j in range(0,3):
        #         for k in range(0,3):
        #             for l in range(0,3):
        #                 # elasticity tensor (cf. Jir/Baz Inelastic analysis of structures Eq.D25):
        #                 D4_e_3D[i,j,k,l] = la * delta[i,j] * delta[k,l] + \
        #                                    mu * ( delta[i,k] * delta[j,l] + delta[i,l] * delta[j,k] )
        #                 # elastic compliance tensor (cf. Simo, Computational Inelasticity, Eq.(2.7.16) AND (2.1.16)):
        #                 C4_e_3D[i,j,k,l] = (1+nu)/(E) * \
        #                                    ( delta[i,k] * delta[j,l] + delta[i,l]* delta[j,k] ) - \
        #                                    nu / E * delta[i,j] * delta[k,l]
        # NOTE: swapaxes returns a reference not a copy!
        # (the index notation always refers to the initial indexing (i=0,j=1,k=2,l=3))
        #delta = identity(3)
        #delta_ijkl = outer(delta, delta).reshape(3, 3, 3, 3)
        #delta_ikjl = delta_ijkl.swapaxes(1, 2)
        #delta_iljk = delta_ikjl.swapaxes(2, 3)
        #D4_e_3D = la * delta_ijkl + mu * (delta_ikjl + delta_iljk)
        # C4_e_3D = -nu / E * delta_ijkl + \
        #    (1 + nu) / (2 * E) * (delta_ikjl + delta_iljk)

        # construct the elasticity tensor (using Numpy - einsum function)
        delta = identity(3)
        D_ijkl = (einsum(',ij,kl->ijkl', la, delta, delta) +
                  einsum(',ik,jl->ijkl', mu, delta, delta) +
                  einsum(',il,jk->ijkl', mu, delta, delta))
        # -----------------------------------------------------------------------------------------------------
        # Get the fourth order elasticity and compliance tensors for the 3D-case
        # -----------------------------------------------------------------------------------------------------

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

    n = 100
    s_levels = linspace(0, 0.005, 2)
    s_levels[0] = 0
    s_levels.reshape(-1, 2)[:, 0] *= -1
    #s_levels.reshape(-1, 2)[:, 1] *= -1
    s_history = s_levels.flatten()
    s_arr = hstack([linspace(s_history[i], s_history[i + 1], n)
                    for i in range(len(s_levels) - 1)])

    eps = array([array([[s_arr[i], 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]) for i in range(0, len(s_arr))])

    m2 = MATS3DMicroplaneDamageJir()
    sigma = zeros_like(eps)
    sigma_kk = zeros(len(s_arr) + 1)
    w = zeros((len(eps[:, 0, 0]), 28))
    e_pi = zeros((len(eps[:, 0, 0]), 28))
    e = zeros((len(eps[:, 0, 0]), 28, 3))
    e_T = zeros((len(eps[:, 0, 0]), 28, 3))
    e_N = zeros((len(eps[:, 0, 0]), 28))
    sctx = zeros((len(eps[:, 0, 0]) + 1, 28))

    for i in range(0, len(eps[:, 0, 0])):

        sigma[i, :] = m2.get_corr_pred(sctx[i, :], eps[i, :])[0]
        sctx[i + 1] = m2._get_state_variables(sctx[i, :], eps[i, :])

        w[i, :] = 1 - m2._get_phi_arr(sctx[i], eps[i, :])
        #e_pi[i, :] = sctx[i, :, 5]
        #e[i, :] = m2._get_e_vct_arr(eps[i, :])
        #e_T[i, :] = m2._get_e_t_vct_arr_2(eps[i, :])
        #e_N[i, :] = m2._get_e_N_arr(e[i, :])

    plt.subplot(221)
    plt.plot(eps[:, 0, 0], sigma[:, 0, 0], linewidth=1, label='sigma_11')
    plt.plot(eps[:, 0, 0], sigma[:, 1, 1], linewidth=1, label='sigma_22')
    plt.plot(eps[:, 0, 0], sigma[:, 0, 1], linewidth=1, label='sigma_12')
    plt.plot(eps[:, 0, 0], sigma[:, 1, 2], linewidth=1, label='sigma_23')
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
    for i in range(0, 28):
        plt.plot(
            eps[:, 0, 0], sctx[0:100, i], linewidth=1, label='Damage of the microplanes', alpha=1)

        plt.xlabel('Strain')
        plt.ylabel('Damage of the microplanes')
        # plt.legend()

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
