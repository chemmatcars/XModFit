####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
from scipy.stats import multivariate_normal

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines'))
from functools import lru_cache
from itertools import combinations
####Please do not remove lines above####

####Import your modules below if needed####
from Chemical_Formula import Chemical_Formula
from PeakFunctions import LogNormal, Gaussian
from Structure_Factors import hard_sphere_sf, sticky_sphere_sf
# from ff_cylinder import ff_cylinder_ml_asaxs
from utils import find_minmax, calc_rho, create_steps

from numba import njit, prange
from scipy.special import j1
import numba_scipy.special

@njit(parallel=True,cache=True, fastmath=True)
def parallelopiped_ml(q, L, B, H, rho, Nphi, Npsi, HggtLB=True):
    if HggtLB:
        tfac=2*np.pi/q/H
        Nphi=1
        dphi=1.0
        Nqt=-1.0
    else:
        tfac=np.ones_like(q)
        dphi=np.pi/Nphi
        Nqt=1.0
    dpsi=2.0*np.pi/Npsi
    fft = np.zeros_like(q)
    L=np.cumsum(L)
    B=np.cumsum(B)
    Nlayers=len(L)
    V = L[:-1]*B[:-1]*H
    drho=2.0*np.diff(np.array(rho))*V
    dphidpsi=dphi*dpsi
    for i in prange(len(q)):
        for iphi in prange(0, Nphi):
            phi = iphi*dphi
            qh = q[i]*H*np.cos(phi)/2.0
            shfac=((1.0-Nqt)+(1.0+Nqt)*np.sinc(qh/np.pi))/2.0
            sphi=((1.0-Nqt)+(1.0+Nqt)*np.sin(phi))/2.0
            qt = q[i]*sphi/2.0#((1.0-Nqt)+(1.0+Nqt)*q[i]*np.sin(phi)/2.0)/2.0
            for ipsi in prange(0, Npsi):
                psi = ipsi*dpsi
                tft = 0.0j
                sc=qt*np.cos(psi)
                ss=qt*np.sin(psi)
                for k in prange(Nlayers-1):
                    ql=L[k]*sc
                    qb=B[k]*ss
                    fac=np.sinc(ql/np.pi)*np.sinc(qb/np.pi)*shfac
                    tft += drho[k] * fac
                fft[i] += np.abs(tft) ** 2 * sphi
    fft*=dphidpsi
    return fft*tfac

class Parallelepiped_Uniform_Edep_2: #Please put the class name same as the function name
    def __init__(self, x=0, Np=10, error_factor=1.0, dist='LogNormal', Energy=None, relement='Au', NrDep='True',# L=1.0, B=1.0,
                 H=1.0, norm=1.0, norm_err=0.01, normQ=0, bkg=0.0, D=1.0, phi=0.1, U=-1.0, SF='None',Nphi=200,Npsi=400, tol=1e-3,HggtLB=True,
                 mpar={'Layers': {'Material': ['Au', 'H2O'], 'Density': [19.32, 1.0], 'SolDensity': [1.0, 1.0],
                                  'Rmoles': [1.0, 1.0], 'L': [1.0, 0.0],'Lsig':[1.0,0.0],'B':[1.0,0.0],'Bsig':[1.0,0.0]}}):
        """
        Documentation
        Calculates the Energy dependent form factor of multilayered Parallelopiped with fixed height with different materials
        and Monte-Carlo integration done over particle size distribution

        x           : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array
        relement    : Resonant element of the nanoparticle. Default: 'Au'
        Energy      : Energy of X-rays in keV at which the form-factor is calculated. Default: None
        Np          : No. of points with which the size distribution will be computed. Default: 10
        H           : Height of the parallelopiped in Angs
        NrDep       : Energy dependence of the non-resonant element. Default= 'False' (Energy independent), 'True' (Energy independent)
        dist        : The probability distribution function for the radii of different interfaces in the nanoparticles. Default: Gaussian
        Nphi        : Number of polar angle points for angular averaging
        Npsi        : Number of azimuthal angle for angular averaging
        norm        : The density of the nanoparticles in nanoMolar (nanoMoles/Liter)
        norm_err    : Percentage of error on normalization to simulated energy dependent SAXS data
        normQ       : power of 'Q' to normalize the Intensity such as [Q^0, Q^1, Q^2, Q^3, Q^4]
        bkg         : Constant incoherent background
        tol         : Tolerence for Monte-Carlo Integration
        error_factor: Error-factor to simulate the error-bars
        D           : Hard Sphere Diameter
        phi         : Volume fraction of particles
        U           : The sticky-sphere interaction energy
        SF          : Type of structure factor. Default: 'None'
        term        : 'SAXS-term' or 'Cross-term' or 'Resonant-term'
        mpar        : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('Sol_Density' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        L and B (Thicknesses in Angs) of the layers starting with 0 for the first layer as the first layer thickness provided by L,B, and H values
                        Lsig and Bsig (Width in Angs) of the distribution of dL and dB
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.norm=norm
        self.norm_err = norm_err
        self.normQ = normQ
        self.bkg=bkg
        self.dist=dist
        self.Np=Np
        self.H=H
        self.HggtLB=HggtLB
        self.Nphi=Nphi
        self.Npsi=Npsi
        self.Energy=Energy
        self.relement=relement
        self.NrDep=NrDep
        self.tol=tol
        self.error_factor=error_factor
        self.D=D
        self.phi=phi
        self.U=U
        self.__mpar__=mpar #If there is any multivalued parameter
        self.SF=SF
        self.choices={'dist':['Gaussian','LogNormal'],'NrDep':['True','False'],
                      'SF':['None','Hard-Sphere', 'Sticky-Sphere'],
                      'HggtLB':['False','True'],
                      'normQ': [0, 1, 2, 3, 4]
                      } #If there are choices available for any fixed parameters
        self.filepaths = {}  # If a parameter is a filename with path
        self.__cf__=Chemical_Formula()
        self.__fit__=False
        self.output_params={'scaler_parameters':{}}
        self.__mkeys__=list(self.__mpar__.keys())
        self.init_params()

    def init_params(self):
        """
        Define all the fitting parameters like
        self.params.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params=Parameters()
        self.params.add('norm',value=self.norm,vary=0, min = -np.inf, max = np.inf, expr = None, brute_step = 0.1)
        self.params.add('D', value=self.D, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('phi', value=self.phi, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('H', value=self.H, vary=0, min=1e-3, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('bkg',value=self.bkg,vary=0, min = -np.inf, max = np.inf, expr = None, brute_step = 0.1)
        self.params.add('U', value=self.U, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        for mkey in self.__mpar__.keys():
            for key in self.__mpar__[mkey].keys():
                if key != 'Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=0.0,max=np.inf, expr=None, brute_step=0.1)
                if key == 'L':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=0.01,max=np.inf, expr=None, brute_step=0.1)
                if key == 'B':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=0.01,max=np.inf, expr=None, brute_step=0.1)

    @lru_cache(maxsize=10)
    def calc_LBdist(self, L, Lsig, B, Bsig, dist, N, seed=1):
        L = np.array(L)
        Lsig = np.array(Lsig)
        covL = np.diag(Lsig[:-1] ** 2)
        B = np.array(B)
        Bsig = np.array(Bsig)
        covB = np.diag(Bsig[:-1] ** 2)
        if dist == 'Gaussian':
            mnormL = multivariate_normal(L[:-1], covL,allow_singular=True)
            Lt = mnormL.rvs(N, random_state=seed)
            Ldist = mnormL.pdf(Lt)
            mnormB = multivariate_normal(B[:-1], covB,allow_singular=True)
            Bt = mnormB.rvs(N, random_state=seed)
            Bdist = mnormB.pdf(Bt)
        else:
            mnormL = multivariate_normal(np.log(L[:-1]), covL,allow_singular=True)
            Lt = np.exp(mnormL.rvs(N, random_state=seed))
            Ldist = mnormL.pdf(np.log(Lt))
            mnormB = multivariate_normal(np.log(B[:-1]), covB,allow_singular=True)
            Bt = np.exp(mnormB.rvs(N, random_state=seed))
            Bdist = mnormB.pdf(np.log(Bt))
        Lt = np.vstack((Lt.T, np.zeros(Lt.shape[0]))).T
        Bt = np.vstack((Bt.T, np.zeros(Bt.shape[0]))).T
        return Lt, Bt, Ldist, Bdist

    @lru_cache(maxsize=10)
    def parallelopiped(self, q, L, Lsig, B, Bsig, H, rho, dist='Gaussian', Np=10, Nphi=200, Npsi=400,tol=1e-3, HggtLB=False):
        q = np.array(q)
        Lt, Bt, Ldist, Bdist = self.calc_LBdist(L, Lsig, B, Bsig, dist, Np)
        form = np.zeros_like(q)
        pfac = (2.818e-5 * 1.0e-8) ** 2
        last=np.zeros_like(q)
        for i in range(Np):
            fft = parallelopiped_ml(q, Lt[i], Bt[i], H, rho, Nphi, Npsi, HggtLB)
            form += Ldist[i]*Bdist[i]*fft
            if i>10 and np.mod(i,10)==0:
                chisq=np.sum(((form-last)/form)**2)/len(q)
                last=1.0*form
                if chisq<tol:
                    break
            dsum = np.sum(Ldist[:i] * Bdist[:i])
        return pfac * form/dsum, Lt[:i], Ldist[:i], Bt[:i], Bdist[:i] # pfac * for/dsum in cm^2

    def update_params(self):
        mkey = self.__mkeys__[0]
        key = 'Density'
        Nmpar = len(self.__mpar__[mkey][key])
        self.__density__ = tuple([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)])
        key = 'SolDensity'
        self.__solDensity__ = tuple([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)])
        key = 'Rmoles'
        self.__Rmoles__ = tuple([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)])
        key = 'L'
        self.__L__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'B'
        self.__B__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'Lsig'
        self.__Lsig__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'Bsig'
        self.__Bsig__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'Material'
        self.__material__ = tuple([self.__mpar__[mkey][key][i] for i in range(Nmpar)])

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        scale = 1e27 / 6.022e23
        svol = 1.5*0.0172**2/370**2  # scattering volume in cm^3
        self.update_params()
        tL=np.array(self.__L__)
        tB=np.array(self.__B__)
        self.__L__[1:]= 2*tL[1:] #This is done to convert thickness of the layer to the total Length of the parallelopiped
        self.__B__[1:] = 2*tB[1:] #This is done to convert thickness of the layer to the total Breadth of the parallelopiped
        self.__Lsig__ = np.array(self.__Lsig__)
        self.__Bsig__ = np.array(self.__Bsig__)
        # self.__L__[0]=self.L
        # self.__B__[0]=self.B
        if type(self.x) == dict:
            sqf = {}
            for key in self.x.keys():
                sq = []
                Energy = float(key.split('_')[1].split('@')[1])
                rho, eirho, adensity, rhor, eirhor, adensityr, cdensityr = calc_rho(R=tuple(self.__L__),
                                                                         material=self.__material__,
                                                                         relement=self.relement,
                                                                         density=self.__density__,
                                                                         sol_density=self.__solDensity__,
                                                                         Energy=Energy, Rmoles=self.__Rmoles__,
                                                                         NrDep=self.NrDep)
                sq, L, Ldist, B, Bdist=self.parallelopiped(tuple(self.x[key]), tuple(self.__L__), tuple(self.__Lsig__),
                                       tuple(self.__B__), tuple(self.__Bsig__),
                                       self.H, tuple(rho),
                                       dist = self.dist, Np = self.Np, Nphi = self.Nphi, Npsi = self.Npsi, tol=self.tol, HggtLB=self.HggtLB)
                sqf[key] = self.norm * 1e-9 * 6.022e20 * sq
            if self.SF is None:
                struct = np.ones_like(self.x[key])  # hard_sphere_sf(self.x[key], D = self.D, phi = 0.0)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x[key], D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x[key], D=self.D, phi=self.phi, U=self.U, delta=0.01)
            sqf[key] = (sqf[key] * struct + self.bkg) * self.x[key]**self.normQ
            if not self.__fit__:
                keys = list(self.output_params.keys())
                for key1 in keys:
                    if key1.startswith('simulated_w_err') or key1.startswith('L_') or key.startswith('B_'):
                        self.output_params.pop(key1, None)
                # L = np.diff(L, axis=1,append=L[:,-1])
                # B = np.diff(L, axis=1,append=B[:,-1])
                if len(self.__L__) > 2:
                    for i, j in combinations(range(len(self.__L__[:-1])), 2):
                        self.output_params['L_%d_%d' % (i + 1, j + 1)] = {'x': L[:, i], 'y': L[:, j],
                                                                          'names':['L_%d'%(i+1),'L_%d'%(j+1)]}
                        self.output_params['B_%d_%d' % (i + 1, j + 1)] = {'x': B[:, i], 'y': B[:, j],
                                                                          'names':['B_%d'%(i+1),'B_%d'%(j+1)]}
                else:
                    iLsort = np.argsort(L[:, 0])
                    iBsort = np.argsort(B[:, 0])
                    self.output_params['L_1'] = {'x': L[iLsort, 0], 'y': Ldist[iLsort]}
                    self.output_params['B_1'] = {'x': B[iBsort, 0], 'y': Bdist[iBsort]}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1]}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1]}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1]}
                self.output_params['Structure_Factor'] = {'x': self.x[key], 'y': struct}

                tkeys = list(self.output_params.keys())
                for key in tkeys:
                    if 'simulated_w_err' in key:
                        del self.output_params[key]

                for key in self.x.keys():
                    Energy = key.split('_')[1].split('@')[1]
                    # sqerr = np.sqrt(self.flux * sqf[key] * svol)
                    # sqwerr = sqf[key] * svol * self.flux + 2 * (0.5 - np.random.rand(len(sqerr))) * sqerr
                    signal = sqf[key]/self.x[key]**self.normQ
                    minsignal = np.min(signal)
                    normsignal = signal / minsignal
                    norm = np.random.normal(self.norm, scale=self.norm_err / 100.0)
                    sqerr = np.random.normal(normsignal * norm, scale=self.error_factor)
                    meta = {'Energy': Energy}
                    self.output_params['simulated_w_err_' + Energy + 'keV'] = {'x': self.x[key], 'y': sqerr * minsignal,
                                                                               'yerr': np.sqrt(
                                                                                   normsignal) * minsignal * self.error_factor,
                                                                               'meta': meta,
                                                                               'names':['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)', 'Ierr (cm<sup>-1</sup>)']}
        else:
            rho, eirho, adensity, rhor, eirhor, adensityr, cdensityr = calc_rho(R=tuple(self.__L__), material=self.__material__,
                                                                     relement=self.relement,
                                                                     density=self.__density__,
                                                                     sol_density=self.__solDensity__,
                                                                     Energy=self.Energy, Rmoles=self.__Rmoles__,
                                                                     NrDep=self.NrDep)
            if self.SF is None:
                struct = np.ones_like(self.x)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x, D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x, D=self.D, phi=self.phi, U=self.U, delta=0.01)

            sq, L, Ldist, B, Bdist = self.parallelopiped(tuple(self.x), tuple(self.__L__), tuple(self.__Lsig__),
                                                         tuple(self.__B__), tuple(self.__Bsig__),
                                                         self.H, tuple(rho),
                                                         dist=self.dist, Np=self.Np, Nphi=self.Nphi, Npsi=self.Npsi,
                                                         tol=self.tol,HggtLB=self.HggtLB)
            sqf = (self.norm * 1e-9 * np.array(sq) * 6.022e20 * struct + self.bkg)*self.x**self.normQ  # in cm^-1 if not normalized by q
            if not self.__fit__:
                keys = list(self.output_params.keys())
                for key in keys:
                    if key.startswith('simulated_w_err') or key.startswith('L_') or key.startswith('B_'):
                        self.output_params.pop(key, None)
                # L = np.absolute(np.diff(L, axis=1))
                # B = np.absolute(np.diff(B, axis=1))
                if len(self.__L__)>2:
                    for i, j in combinations(range(len(self.__L__[:-1])), 2):
                        self.output_params['L_%d_%d' % (i + 1, j + 1)] = {'x': L[:, i], 'y': L[:, j],
                                                                          'names': ['L_%d' % (i + 1),
                                                                                        'L_%d' % (j + 1)]}
                        self.output_params['B_%d_%d' % (i + 1, j + 1)] = {'x': B[:, i], 'y': B[:, j],
                                                                          'names': ['B_%d' % (i + 1),
                                                                                        'B_%d' % (j + 1)]}
                else:
                    iLsort=np.argsort(L[:,0])
                    iBsort=np.argsort(B[:,0])
                    self.output_params['L_1']={'x':L[iLsort,0],'y':Ldist[iLsort]}
                    self.output_params['B_1'] = {'x': B[iBsort, 0], 'y': Bdist[iBsort]}
                signal = sqf[0:]/self.x**self.normQ
                minsignal = np.min(signal)
                normsignal = signal / minsignal
                norm = np.random.normal(self.norm, scale=self.norm_err / 100.0)
                sqerr = np.random.normal(normsignal * norm, scale=self.error_factor)
                meta = {'Energy': self.Energy}
                tkeys = list(self.output_params.keys())
                for key in tkeys:
                    if 'simulated_w_err' in key:
                        del self.output_params[key]
                if self.Energy is not None:
                    self.output_params['simulated_w_err_%.3fkeV' % self.Energy] = {'x': self.x, 'y': sqerr * minsignal,
                                                                                   'yerr': np.sqrt(
                                                                                       normsignal) * minsignal * self.error_factor,
                                                                                   'meta': meta,
                                                                                   'names':['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)', 'Ierr (cm<sup>-1</sup>)']}
                else:
                    self.output_params['simulated_w_err'] = {'x': self.x, 'y': sqerr * minsignal,
                                                             'yerr': np.sqrt(
                                                                 normsignal) * minsignal * self.error_factor,
                                                             'meta': meta,
                                                             'names':['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)', 'Ierr (cm<sup>-1</sup>)']}
                self.output_params['Structure_Factor'] = {'x': self.x, 'y': struct}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1]}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1]}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1]}
        return sqf



if __name__=='__main__':
    x = {'Total_E@11.919': np.logspace(np.log10(0.003), np.log10(0.15), 500),}
    fun=Parallelepiped_Uniform_Edep_2(x=x)
    print(fun.y())
