####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
from scipy.stats import multivariate_normal
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines'))
####Please do not remove lines above####

####Import your modules below if needed####
from FormFactors.Sphere import Sphere
# from ff_sphere import ff_sphere_ml
from Chemical_Formula import Chemical_Formula
from PeakFunctions import LogNormal, Gaussian
from Structure_Factors import hard_sphere_sf, sticky_sphere_sf
from utils import find_minmax, calc_rho, create_steps
from functools import lru_cache
# import jscatter as js

from itertools import combinations

from numba import njit, prange
from ASAXS.Sphere_Uniform import ff_sphere_ml

# @njit(parallel=True, cache=True)
# def ff_sphere_ml(q,R,rho):
#     Nlayers=len(R)
#     aff=np.ones_like(q)*complex(0,0)
#     ff=np.zeros_like(q)
#     for i in prange(len(q)):
#         fact = 0.0
#         rt = 0.0
#         for j in prange(1,Nlayers):
#             rt += R[j - 1]
#             fact += (rho[j - 1] - rho[j]) * (np.sin(q[i] * rt) - q[i] * rt * np.cos(q[i] * rt)) / q[i] ** 3
#         aff[i] = fact
#         ff[i] = abs(fact) ** 2
#     return ff,aff

class Bimodal_Sphere_Uniform_2: #Please put the class name same as the function name
    def __init__(self, x=0, Np=20, error_factor=1.0, term='Total',dist='Gaussian', Energy=None, relement='Au', NrDep='False',
                 norm=1.0, norm_err=0.01, sbkg=0.0, cbkg=0.0, abkg=0.0, D1=1.0, phi1=0.0, U1=-1.0, mode1_fraction=1.0,
                 mode2_fraction=0.0, D2=1.0, phi2=0.0, U2=-1.0, SF='None',tol=1e-3,
                 mpar={'Mode1':{'Material':['Au','H2O'],
                                'Density':[19.32,1.0],
                                'SolDensity':[1.0,1.0],
                                'Rmoles':[1.0,0.0],
                                'R':[1.0,0.0],
                                'Rsig':[1.0,1.0]},
                       'Mode2':{'Material':['Au','H2O'],
                                'Density':[19.32,1.0],
                                'SolDensity': [1.0, 1.0],
                                'Rmoles':[1.0,0.0],
                                'R':[1.0,0.0],
                                'Rsig':[1.0,1.0]}
                       }):
        """
        Documentation
        Calculates the Energy dependent form factor of bimodal multilayered spherical nanoparticles

        x             : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array
        relement      : Resonant element of the nanoparticle. Default: 'Au'
        Energy        : Energy of X-rays in keV at which the form-factor is calculated. Default: None
        Np            : No. of points with which the size distribution will be computed. Default: 10
        NrDep         : Energy dependence of the non-resonant element. Default= 'False' (Energy independent), 'True' (Energy dependent)
        dist          : The probability distribution function for the radii of different interfaces in the nanoparticles. Default: Gaussian
        norm          : The density of the nanoparticles in nanoMolar (nanoMoles/Liter)
        norm_err      : Percentage of error on normalization to simulated energy dependent SAXS data
        sbkg          : Constant incoherent background for SAXS-term
        cbkg          : Constant incoherent background for cross-term
        abkg          : Constant incoherent background for Resonant-term
        error_factor  : Error-factor to simulate the error-bars
        term          : 'SAXS-term' or 'Cross-term' or 'Resonant-term' or 'Total'
        D1            : Hard Sphere Diameter of first kind of nanoparticles
        phi1          : Volume fraction of first kind of nanoparticles
        U1            : The sticky-sphere interaction energy between first kind of nanoparticles
        mode1_fraction: Number fractionn of 1st kind of nanoparticles
        mode2_fraction: Number fractionn of 2nd kind of nanoparticles
        D2            : Hard Sphere Diameter of 2nd kind of nanoparticles
        phi2          : Volume fraction of second kind of nanoparticles
        U2            : The sticky-sphere interaction energy between 2nd kind of nanoparticles
        SF            : Type of structure factor. Default: 'None'
        tol           : Tolerence for the Monte-Carlo Integration
        mpar          : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('SolDensity' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        Mean Radii ('R' in Angs), and
                        Distribution of Radii ('Rsig' in Angs)

        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.norm=norm
        self.norm_err = norm_err
        self.sbkg=sbkg
        self.cbkg=cbkg
        self.abkg=abkg
        self.dist=dist
        self.Np=Np
        self.Energy=Energy
        self.relement=relement
        self.NrDep=NrDep
        #self.rhosol=rhosol
        self.error_factor=error_factor
        self.D1=D1
        self.phi1=phi1
        self.U1=U1
        self.mode1_fraction = mode1_fraction
        self.mode2_fraction = mode2_fraction
        self.D2 = D2
        self.phi2 = phi2
        self.U2 = U2
        self.__mpar__=mpar #If there is any multivalued parameter
        self.SF=SF
        self.term=term
        self.tol=tol
        self.__Density__={}
        self.__SolDensity__={}
        self.__R__={}
        self.__Rsig__={}
        self.__Rmoles__={}
        self.__material__={}
        self.choices={'dist':['Gaussian','LogNormal'],'NrDep':['True','False'],'SF':['None','Hard-Sphere', 'Sticky-Sphere'],
                      'term':['SAXS-term','Cross-term','Resonant-term','Total']} #If there are choices available for any fixed parameters
        self.filepaths = {}  # If a parameter is a filename with path
        self.__fit__=False
        self.__mkeys__=list(self.__mpar__.keys())
        self.init_params()


    def init_params(self):
        """
        Define all the fitting parameters like
        self.params.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params=Parameters()
        self.params.add('norm',value=self.norm,vary=0, min = -np.inf, max = np.inf, expr = None, brute_step = 0.1)
        self.params.add('D1', value=self.D1, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('phi1', value=self.phi1, vary=0, min=0.0, max=1, expr=None, brute_step=0.1)
        self.params.add('U1', value=self.U1, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('mode1_fraction', value=self.mode1_fraction, vary=0, min=0.0, max=1.0, expr=None, brute_step=0.1)
        self.params.add('mode2_fraction', value=self.mode2_fraction, vary=0, min=0.0, max=1.0, expr=None, brute_step=0.1)
        self.params.add('D2', value=self.D2, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('phi2', value=self.phi2, vary=0, min=0.0, max=1.0, expr=None, brute_step=0.1)
        self.params.add('U2', value=self.U2, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sbkg',value=self.sbkg,vary=0, min = -np.inf, max = np.inf, expr = None, brute_step = 0.1)
        self.params.add('cbkg', value=self.cbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('abkg', value=self.abkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key != 'Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=-np.inf, max=np.inf, expr=None, brute_step=0.1)

    @lru_cache(maxsize=10)
    def calc_Rdist(self, R, Rsig, dist, N, seed=1):
        R = np.array(R)
        Rsig = np.array(Rsig)
        cov = np.diag(Rsig[:-1] ** 2)
        if dist == 'Gaussian':
            mnorm = multivariate_normal(R[:-1], cov)
            Rl = mnorm.rvs(N, random_state=seed)
            rdist = mnorm.pdf(Rl)
        else:
            mnorm = multivariate_normal(np.log(R[:-1]), cov)
            Rl = np.exp(mnorm.rvs(N, random_state=seed))
            rdist = mnorm.pdf(np.log(Rl))
        Rl = np.vstack((Rl.T, np.zeros(Rl.shape[0]))).T
        return Rl, rdist

    @lru_cache(maxsize=10)
    def new_sphere(self, q, R, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, tol=1e-3):
        q = np.array(q)
        Rl, rdist = self.calc_Rdist(R, Rsig, dist, Np, seed=1)
        form = np.zeros_like(q)
        eiform = np.zeros_like(q)
        aform = np.zeros_like(q)
        cform = np.zeros_like(q)
        pfac = (4 * np.pi * 2.818e-5 * 1.0e-8) ** 2
        last = np.zeros_like(q)
        for i in range(Np):
            ff, mff = ff_sphere_ml(q, Rl[i], rho)
            form += rdist[i] * ff
            eiff, meiff = ff_sphere_ml(q, Rl[i], eirho)
            eiform += rdist[i] * eiff
            aff, maff = ff_sphere_ml(q, Rl[i], adensity)
            aform += rdist[i] * aff
            cform += rdist[i] * (meiff * maff.conjugate() + meiff.conjugate() * maff).real
            if i > 10 and np.mod(i, 10) == 0:
                chisq = np.sum(((form - last) / form) ** 2) / len(q)
                last = 1.0 * form
                if chisq < tol:
                    break
        sdist = np.sum(rdist[:i])
        form = pfac * form / sdist
        eiform = pfac * eiform / sdist
        aform = pfac * aform / sdist
        cform = np.abs(pfac * cform) / 2 / sdist
        return form, eiform, aform, cform, Rl[:i], rdist[:i]

    @lru_cache(maxsize=10)
    def new_sphere_dict(self, q, R, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, tol=1e-3):
        form, eiform, aform, cform, r, rdist = self.new_sphere(q, R, Rsig, rho, eirho, adensity, dist=dist, Np=Np, tol=tol)
        result={'SAXS-term': eiform,
                'Resonant-term': aform,
                'Cross-term': cform,
                'Total': form}
        return result, r, rdist


    def update_params(self):
        for mkey in self.__mkeys__:
            key = 'Density'
            Nmpar=len(self.__mpar__[mkey][key])
            self.__Density__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)])
            key = 'SolDensity'
            self.__SolDensity__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)])
            key = 'Rmoles'
            self.__Rmoles__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)])
            key = 'R'
            self.__R__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)])
            key = 'Rsig'
            self.__Rsig__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)])
            key = 'Material'
            self.__material__[mkey] = tuple([self.__mpar__[mkey][key][i] for i in range(Nmpar)])


    def y(self):
        """
        Define the function in terms of x to return some value
        """
        svol = 1.5 * 0.0172 ** 2 / 370 ** 2  # scattering volume in cm^3
        self.output_params = {'scaler_parameters': {}}
        self.update_params()
        scale = 1e27 / 6.022e23
        frac={'Mode1':self.mode1_fraction*self.norm * 1e-9*6.022e20,'Mode2':self.mode2_fraction*self.norm * 1e-9*6.022e20}
        if type(self.x) == dict:
            sqf = {}
            xkeys = list(self.x.keys())
            for key in xkeys:
                sqf[key] = np.zeros_like(self.x[xkeys[0]])
            total = np.zeros_like(self.x[xkeys[0]])
            struct={}
            Rl={}
            rdist={}
            rho={}
            eirho={}
            adensity={}
            for mkey in self.__mkeys__:
                rho[mkey], eirho[mkey], adensity[mkey], rhor, eirhor, adensityr, cdensityr = calc_rho(R=self.__R__[mkey],
                                                                         material=self.__material__[mkey],
                                                                         relement=self.relement,
                                                                         density=self.__Density__[mkey],
                                                                         sol_density=self.__SolDensity__[mkey],
                                                                         Energy=self.Energy,
                                                                         Rmoles=self.__Rmoles__[mkey],
                                                                         NrDep=self.NrDep)
                sphere, Rl[mkey], rdist[mkey] = self.new_sphere_dict(tuple(self.x[xkeys[0]]), self.__R__[mkey],
                                                         self.__Rsig__[mkey], tuple(rho[mkey]), tuple(eirho[mkey]),
                                                         tuple(adensity[mkey]), dist=self.dist,
                                                         Np=self.Np, tol=self.tol, keys=tuple(xkeys))  # in cm^-1

                if mkey=='Mode1':
                    D,phi,U=self.D1,self.phi1,self.U1
                else:
                    D,phi,U=self.D2,self.phi2,self.U2

                if self.SF is None:
                    struct[mkey] = np.ones_like(self.x[xkeys[0]])
                    # hard_sphere_sf(self.x[key], D = self.D, phi = 0.0)
                elif self.SF == 'Hard-Sphere':
                    struct[mkey] = hard_sphere_sf(self.x[xkeys[0]], D=D, phi=phi)
                    # struct[mkey] = js.sf.PercusYevick(self.x[xkeys[0]], D, eta=phi)[1]
                else:
                    struct[mkey] = sticky_sphere_sf(self.x[xkeys[0]], D=D, phi=phi, U=U, delta=0.01)

                for key in xkeys:
                    sqf[key] = sqf[key] + frac[mkey] * sphere[key] * struct[mkey]
                total = total + frac[mkey] * sphere['Total'] * struct[mkey]

            sqf['SAXS-term']=sqf['SAXS-term']+self.sbkg
            sqf['Resonant-term']=sqf['Resonant-term']+self.abkg
            sqf['Cross-term']=sqf['Cross-term']+self.sbkg
            total=total+self.sbkg

            if not self.__fit__:
                signal = total
                minsignal = np.min(signal)
                normsignal = signal / minsignal
                norm = np.random.normal(self.norm, scale=self.norm_err / 100.0)
                sqerr = np.random.normal(normsignal * norm, scale=self.error_factor)
                meta = {'Energy': self.Energy}
                if self.Energy is not None:
                    self.output_params['simulated_w_err_%.4fkeV' % self.Energy] = {'x': self.x[key],
                                                                                   'y': sqerr * minsignal,
                                                                                   'yerr': np.sqrt(
                                                                                       normsignal) * minsignal * self.error_factor,
                                                                                   'meta': meta}
                self.output_params['Simulated_total_wo_err'] = {'x': self.x[xkeys[0]], 'y': total}
                for key in xkeys:
                    self.output_params[key] = {'x': self.x[xkeys[0]], 'y': sqf[key]}
                for mkey in self.__mkeys__:
                    xtmp, ytmp = create_steps(x=np.array(self.__R__[mkey]), y=rho[mkey])
                    self.output_params[mkey+'_rho_r'] = {'x': xtmp, 'y': ytmp}
                    xtmp, ytmp = create_steps(x=np.array(self.__R__[mkey]), y=eirho[mkey])
                    self.output_params[mkey+'_eirho_r'] = {'x': xtmp, 'y': ytmp}
                    xtmp, ytmp = create_steps(x=np.array(self.__R__[mkey]),y=adensity[mkey])
                    self.output_params[mkey+'_adensity_r'] = {'x': xtmp, 'y': ytmp*scale}
                    self.output_params[mkey+'_Structure_Factor'] = {'x': np.array(self.x[xkeys[0]]), 'y': struct[mkey]}
                    for i, j in combinations(range(len(self.__R__[mkey][:-1])), 2):
                        self.output_params[mkey+'_R_%d_%d' % (i + 1, j + 1)] = {'x': Rl[mkey][:, i], 'y': Rl[mkey][:, j],
                                                                          'names': ['R_%d' % (i + 1), 'R_%d' % (j + 1)]}

        else:
            sqf=np.zeros_like(self.x)
            asqf = np.zeros_like(self.x)
            eisqf = np.zeros_like(self.x)
            csqf = np.zeros_like(self.x)
            Rl={}
            rdist={}
            rho={}
            eirho={}
            adensity={}
            struct={}
            for mkey in self.__mkeys__:
                if mkey == 'Mode1':
                    D, phi, U = self.D1, self.phi1, self.U1
                else:
                    D, phi, U = self.D2, self.phi2, self.U2
                if self.SF is None:
                    struct[mkey] = np.ones_like(self.x)
                    # hard_sphere_sf(self.x[key], D = self.D, phi = 0.0)
                elif self.SF == 'Hard-Sphere':
                    struct[mkey] = hard_sphere_sf(self.x, D=D, phi=phi)
                    # struct[mkey] = js.sf.PercusYevick(self.x, D, eta=phi)[1]
                else:
                    struct[mkey] = sticky_sphere_sf(self.x, D=D, phi=phi, U=U, delta=0.01)
                    # tau = np.exp(U) * (D + 0.01) / 12.0 / 0.01
                    # struct[mkey] = js.sf.adhesiveHardSphere(self.x, D / 2, tau, 0.01, eta=phi).array[1, :]

                rho[mkey], eirho[mkey], adensity[mkey], rhor, eirhor, adensityr, cdensityr = calc_rho(R=self.__R__[mkey],
                                                                                    material=self.__material__[mkey],
                                                                                    relement=self.relement,
                                                                                    density=self.__Density__[mkey],
                                                                                    sol_density=self.__SolDensity__[mkey],
                                                                                    Energy=self.Energy,
                                                                                    Rmoles=self.__Rmoles__[mkey],
                                                                                    NrDep=self.NrDep)
                tsqf, teisqf, tasqf, tcsqf, Rl[mkey], rdist[mkey] = self.new_sphere(tuple(self.x), self.__R__[mkey], self.__Rsig__[mkey], tuple(rho[mkey]),
                                                      tuple(eirho[mkey]), tuple(adensity[mkey]),dist=self.dist,Np=self.Np, tol=self.tol)
                sqf = sqf + frac[mkey] * np.array(tsqf) * struct[mkey]  # in cm^-1
                asqf = asqf + frac[mkey] * np.array(tasqf) * struct[mkey] # in cm^-1
                eisqf = eisqf + frac[mkey] * np.array(teisqf) * struct[mkey]  # in cm^-1
                csqf = csqf + frac[mkey] * np.array(tcsqf) * struct[mkey]  # in cm^-1
            asqf = asqf + self.abkg
            eisqf = eisqf + self.sbkg
            csqf = csqf + self.cbkg
            sqf = sqf + self.sbkg

            if not self.__fit__: #Generate all the quantities below while not fitting
                signal = np.array(sqf)
                minsignal = np.min(signal)
                normsignal = signal / minsignal
                norm = np.random.normal(self.norm, scale=self.norm_err / 100.0)
                sqerr = np.random.normal(normsignal * norm, scale=self.error_factor)
                meta = {'Energy': self.Energy}
                if self.Energy is not None:
                    self.output_params['simulated_w_err_%.4fkeV' % self.Energy] = {'x': self.x, 'y': sqerr * minsignal,
                                                                                   'yerr': np.sqrt(
                                                                                       normsignal) * minsignal * self.error_factor,
                                                                                   'meta': meta}
                else:
                    self.output_params['simulated_w_err'] = {'x': self.x, 'y': sqerr * minsignal,
                                                             'yerr': np.sqrt(normsignal) * minsignal * self.error_factor,
                                                             'meta': meta}
                self.output_params['Total'] = {'x': self.x, 'y': sqf}
                self.output_params['Resonant-term'] = {'x': self.x, 'y': asqf}
                self.output_params['SAXS-term'] = {'x': self.x, 'y': eisqf}
                self.output_params['Cross-term'] = {'x': self.x, 'y': csqf}
                for mkey in self.__mkeys__:
                    self.output_params[mkey+'_Structure_Factor'] = {'x': self.x, 'y': struct[mkey]}
                    xtmp, ytmp = create_steps(x=np.array(self.__R__[mkey]), y=rho[mkey])
                    self.output_params[mkey+'_rho_r'] = {'x': xtmp, 'y': ytmp}
                    xtmp, ytmp = create_steps(x=np.array(self.__R__[mkey]), y=eirho[mkey])
                    self.output_params[mkey+'_eirho_r'] = {'x': xtmp, 'y': ytmp}
                    xtmp, ytmp = create_steps(x=np.array(self.__R__[mkey]), y=adensity[mkey])
                    self.output_params[mkey+'_adensity_r'] = {'x': xtmp, 'y': ytmp*scale}
                    for i, j in combinations(range(len(self.__R__[mkey][:-1])), 2):
                        self.output_params[mkey+'_R_%d_%d' % (i + 1, j + 1)] = {'x': Rl[mkey][:, i], 'y': Rl[mkey][:, j],
                                                                          'names': ['R_%d' % (i + 1), 'R_%d' % (j + 1)]}

            sqf = self.output_params[self.term]['y']
        return sqf


if __name__=='__main__':
    x=np.logspace(-3,0,200)
    fun=Bimodal_Sphere_Uniform_2(x=x)
    print(fun.y())
