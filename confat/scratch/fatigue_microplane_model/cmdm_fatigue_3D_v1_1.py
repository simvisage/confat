'''
Created on 10.03.2017

@author: abaktheer
'''
'''
Microplane Fatigue Model _ Double mapping v1.1
'''
from ibvpy.core.rtrace_eval import \
    RTraceEval
from ibvpy.mats.mats3D.mats3D_eval import MATS3DEval
from ibvpy.mats.mats_eval import \
    IMATSEval
from numpy import \
    array, zeros, outer, inner, transpose, dot, trace, \
    fabs, identity, tensordot, einsum, \
    float_, identity, sign, fabs, \
    sqrt as arr_sqrt, copy
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


class MATSEvalMicroplaneFatigue(HasTraits):

    G = Float(200,
              label="G",
              desc="Shear Stiffness",
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

    S = Float(1,
              label="S",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    r = Float(1,
              label="r",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    c = Float(1,
              label="c",
              desc="Damage cumulation parameter",
              enter_set=True,
              auto_set=False)

    tau_pi_bar = Float(5,
                       label="Tau_pi_bar",
                       desc="Reversibility limit",
                       enter_set=True,
                       auto_set=False)

    a = Float(1.7,
              label="a",
              desc="Lateral pressure coefficient",
              enter_set=True,
              auto_set=False)

    n_s = Constant(4)

    def get_phi_epspi(self, eps, sctx, sigma_kk):

        w = sctx[0]
        z = sctx[1]
        alpha = sctx[2:4]
        xs_pi = sctx[5:7]

        sig_pi_trial = self.G * (eps - xs_pi)
        Z = self.K * z
        X = self.gamma * alpha
        f = norm(sig_pi_trial - X) - self.tau_pi_bar - \
            Z + self.a * sigma_kk / 3

        elas = f <= 1e-6
        plas = f > 1e-6

        delta_lamda = f / (self.E_b / (1 - w) + self.gamma + self.K) * plas

        xs_pi = xs_pi + delta_lamda * \
            norm(sig_pi_trial - X) / ((sig_pi_trial - X) * (1 - w))
        Y = 0.5 * self.G * dot((eps - xs_pi), (eps - xs_pi))
        w = w + (1 - w) ** self.c * (delta_lamda * (Y / self.S) ** self.r)

        X = X + self.gamma * delta_lamda * \
            norm(sig_pi_trial - X) / (sig_pi_trial - X)
        alpha = alpha + delta_lamda * \
            norm(sig_pi_trial - X) / (sig_pi_trial - X)
        z = z + delta_lamda

        return w, z, alpha, xs_pi

class MATSXDMicroplaneDamageFatigue(MATSEvalMicroplaneFatigue):

    '''
    Microplane Damage Fatigue Model.
    '''
    #-------------------------------------------------------------------------
    # Classification traits (supplied by the dimensional subclasses)
    #-------------------------------------------------------------------------

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
    '''
    def get_state_array_size(self):
        # In the state array the largest equivalent microplane
        # strains reached in the loading history are saved
        return self.n_mp
    '''

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
        # Switch from engineering notation to tensor notation
        eps_mtx = self.map_eps_eng_to_mtx(eps_eng)
        # slower: e_vct_arr = array( [ dot( eps_mtx, mpn ) for mpn in self._MPN ] )
        # slower: e_vct_arr = transpose( dot( eps_mtx, transpose(self._MPN) ))
        # due to the symmetry of the strain tensor eps_mtx = transpose(eps_mtx)
        # and so this is equal
        #e_vct_arr = dot(self._MPN, eps_mtx)

        e_ni = einsum('nj,ji->ni', self._MPN, eps_mtx)
        return e_ni

    def _get_e_N_arr(self, e_vct_arr):
        # get the normal strain array for each microplane

        # return array([dot(e_vct, mpn)
        #             for e_vct, mpn in zip(e_vct_arr, self._MPN)])

        eN_n = einsum('ni,ni->n', e_vct_arr, self._MPN)
        return eN_n

    def _get_e_T_vct_arr(self, e_vct_arr):
        # get the tangential strain vector array for each microplane
        eN_n = self._e_N_arr

        # e_N_vct_arr_ni = array([self._MPN[i, :] * e_N_arr_n[i]
        #                        for i in range(0, self.n_mp)])

        eN_vct_ni = einsum('n,ni->ni', eN_n, self._MPN)

        return e_vct_arr - eN_vct_ni

    # Alternative methods for the kinematic constraint
    def _get_e_N_arr_2(self, eps_eng):

        eps_mtx = self.map_eps_eng_to_mtx(eps_eng)
        return einsum('nij,ij->n', self._MPNN, eps_mtx)

    def _get_e_t_vct_arr_2(self, eps_eng):

        eps_mtx = self.map_eps_eng_to_mtx(eps_eng)
        return einsum('nijk,jk->ni', self._MPTT, eps_mtx)

    def _get_e_vct_arr_2(self, eps_eng):

        return self._e_N_arr_2 * self._MPN + self._e_t_vct_arr_2
    

    def _get_state_variables(self, sctx, eps_app_eng):

        e_vct_arr = self._e_vct_arr(eps_app_eng)

        sigma_kk = trace(self.get_corr_pred(sctx, eps_app_eng)[0])

        sctx_arr = array([self.get_phi_epspi(e_vct_arr[i], sctx[i], sigma_kk)
                          for i in range(0, self.n_mp)])

        return sctx_arr

    def _get_phi_arr(self, sctx, eps_app_eng):
        '''
        Returns a list of the integrity factors for all microplanes.
        '''
        phi_arr = 1 - self._get_state_variables(sctx, eps_app_eng)[:, 0]
        return phi_arr

    def _get_eps_pi_arr(self, sctx, eps_app_eng):
        '''
        Returns a list of the sliding strain vector for all microplanes.

        '''
        e_vct_arr = self._e_vct_arr(eps_app_eng)

        eps_pi_arr = self._get_state_variables(sctx, e_vct_arr )[:, 5:7]

        return eps_pi_arr

    def _get_phi_mtx(self, sctx, eps_app_eng):
        '''
        Returns the 2nd order damage tensor 'phi_mtx'
        '''
        # scalar integrity factor for each microplane
        phi_arr = self._get_phi_arr(sctx, eps_app_eng)
        # integration terms for each microplanes
        # phi_mtx_arr = array([phi_arr[i] * self._MPNN[i, :, :] * self._MPW[i]
        #                     for i in range(0, self.n_mp)]) # old implementation
        #phi_mtx = phi_mtx_arr.sum(0)

        phi_ij = einsum('n,n,nij->ij', phi_arr, self._MPW, self._MPNN)
        # sum of contributions from all microplanes
        # sum over the first dimension (over the microplanes)

        return phi_ij

    def _get_beta_tns_sum_type(self, phi_mtx):
        '''
        Returns the 4th order damage tensor 'beta4' using sum-type symmetrization
        (cf. [Jir99], Eq.(21))
        '''
        n_dim = self.n_dim
        delta = self.identity_tns

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

    def _get_eps_pi_mtx(self, sctx, eps_app_eng):

        # Vector integration of sliding (inelastic) strain for each microplane
        eps_pi_ni = self._get_eps_pi_arr(sctx, eps_app_eng)

        # eps_pi_mtx_arr = array([0.5 * (outer(self._MPN[i], eps_pi_vct_arr[i]) +
        #                               outer(eps_pi_vct_arr[i], self._MPN[i])) * self._MPW[i]
        #                       for i in range(0, self.n_mp)])
        #eps_pi_mtx = eps_pi_mtx_arr.sum(0)

        delta = identity(3)
        # eps_pi_ij = 0.5 * (einsum('n,nr,ni,rj->ij', self._MPW, eps_pi_nr, self._MPN, delta) +
        # einsum('n,nr,nj,ri->ij', self._MPW, eps_pi_nr, self._MPN, delta))

        eps_pi_ij = 0.5 * (einsum('n,ni,nj -> ij', self._MPW, eps_pi_ni, self._MPN) +
                           einsum('n,nj,ni -> ij', self._MPW, eps_pi_ni, self._MPN))

        return eps_pi_ij

    #-------------------------------------------------------------------------
    # Evaluation - get the corrector and predictor
    #-------------------------------------------------------------------------

    def get_corr_pred(self, sctx, eps_app_eng, d_eps, tn, tn1, eps_avg=None):
        '''
        Corrector predictor computation.
        @param eps_app_eng input variable - engineering strain
        '''
        # -----------------------------------------------------------------------------------------------
        # check if average strain is to be used for damage evaluation
        # -----------------------------------------------------------------------------------------------
        if eps_avg != None:
            pass
        else:
            eps_avg = eps_app_eng

        # -----------------------------------------------------------------------------------------------
        # for debugging purposes only: if elastic_debug is switched on, linear elastic material is used
        # -----------------------------------------------------------------------------------------------
        if self.elastic_debug:
            # NOTE: This must be copied otherwise self.D2_e gets modified when
            # essential boundary conditions are inserted
            D2_e = copy(self.D2_e)
            sig_eng = tensordot(D2_e, eps_app_eng, [[1], [0]])
            return sig_eng, D2_e

        # -----------------------------------------------------------------------------------------------
        # update state variables
        # -----------------------------------------------------------------------------------------------
        #if sctx.update_state_on:
        #    eps_n = eps_avg - d_eps
        #    e_max = self._get_state_variables(sctx, eps_n)
        #    sctx.mats_state_array[:] = e_max

        #----------------------------------------------------------------------
        # if the regularization using the crack-band concept is on calculate the
        # effective element length in the direction of principle strains
        #----------------------------------------------------------------------
        if self.regularization:
            h = self.get_regularizing_length(sctx, eps_app_eng)
            self.phi_fn.h = h

        #----------------------------------------------------------------------
        # stiffness version:
        #----------------------------------------------------------------------
        if self.model_version == 'stiffness':

            #------------------------------------------------------------------
            # Damage tensor (2th order):
            #------------------------------------------------------------------

            phi_ij = self._get_phi_mtx(sctx, eps_avg)

            #------------------------------------------------------------------
            # Damage tensor (4th order) using product- or sum-type symmetrization:
            #------------------------------------------------------------------
            beta_ijkl = self._get_beta_tns(phi_ij)

            #------------------------------------------------------------------
            # Damaged stiffness tensor calculated based on the damage tensor beta4:
            #------------------------------------------------------------------
            # (cf. [Jir99] Eq.(7): C = beta * D_e * beta^T),
            # minor symmetry is tacitly assumed ! (i.e. beta_ijkl = beta_jilk)
            #D4_mdm = tensordot(
            #    beta_ijkl, tensordot(self.D4_e, beta_ijkl, [[2, 3], [2, 3]]), [[2, 3], [0, 1]])

            D4_mdm_ijmn = einsum('ijkl,klsr,mnsr->ijmn', beta_ijkl, self.D4_e, beta_ijkl)
            #------------------------------------------------------------------
            # Reduction of the fourth order tensor to a matrix assuming minor and major symmetry:
            #------------------------------------------------------------------
            D2_mdm = self.map_tns4_to_tns2(D4_mdm_ijmn)

        #----------------------------------------------------------------------
        # Return stresses (corrector) and damaged secant stiffness matrix (predictor)
        #----------------------------------------------------------------------
        eps_pi_ij = self._eps_pi_mtx(sctx, eps_app_eng)
        eps_e_mtx = eps_app_eng - eps_pi_ij

        #sig_eng = tensordot(D2_mdm, eps_e_mtx, [[1], [0]])
        sig_eng = einsum('ij,jk -> ik',D2_mdm, eps_e_mtx)
        return sig_eng, D2_mdm
    
class MATS3DMicroplaneDamage(MATSXDMicroplaneDamageFatigue, MATS3DEval):

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
        E = self.E
        nu = self.nu

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
        D2_e_3D = self.map_tns4_to_tns2(D_ijkl)

        return D_ijkl, D2_e_3D

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
    
    
    