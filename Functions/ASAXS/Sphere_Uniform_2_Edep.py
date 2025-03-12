####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
from scipy.stats import multivariate_normal
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines'))
from functools import lru_cache
from itertools import combinations
####Please do not remove lines above####

####Import your modules below if needed####
# from FormFactors.Sphere import Sphere
# from ff_sphere import ff_sphere_ml
from PeakFunctions import LogNormal, Gaussian
from utils import find_minmax, calc_rho
from Structure_Factors import hard_sphere_sf, sticky_sphere_sf

from ASAXS.Sphere_Uniform import ff_sphere_ml

# @njit(parallel=True, cache=True)
# def ff_sphere_ml(q, R, rho):
#     Nlayers = len(R)
#     aff = np.zeros_like(q, dtype=np.complex128)
#     ff = np.zeros_like(q)
#
#     for i in prange(len(q)):
#         fact = 0.0
#         rt = 0.0
#         for j in range(1, Nlayers):
#             rt += R[j - 1]
#             q_rt = q[i] * rt
#             fact += (rho[j - 1] - rho[j]) * (np.sin(q_rt) - q_rt * np.cos(q_rt)) / q[i] ** 3
#         aff[i] = fact
#         ff[i] = np.abs(fact) ** 2
#     return ff, aff


class Sphere_Uniform_2_Edep: #Please put the class name same as the function name
    def __init__(self, x=0, Np=1000, error_factor=1, bkg=0.0,dist='Gaussian', relement='Au', Energy=None, NrDep='True',
                 D=1.0, phi=0.1, U=-1.0, SF='None',tol=1e-3,norm=1.0,norm_err=0.01,normQ=0, Energy_Offset=0.0,
                 mpar={'Multilayers':{'Material':['Au','H2O'],'Density':[19.32,1.0],'SolDensity':[1.0,1.0],'Rmoles':[1.0,1.0],'R':[20.0,0.0],'Rsig':[1.0,1.0]}}):
        """
        Documentation
        Calculates the Energy dependent form factor of multilayered nanoparticles with size distribution over all the layers done usine Monte-Carlo Method

        x           : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array. For energy dependence you need to
        provide a dictionary like {'E_11.919':linspace(0.001,1.0,1000)}
        relement    : Resonant element of the nanoparticle. Default: 'Au'
        Np          : No. of points with which the size distribution will be computed. Default: 10
        Energy      : Energy of the X-rays in keV
        Energy_Offset: Energy offset of X-rays in keV of the resonant element
        NrDep       : Energy dependence of the non-resonant element. Default= 'False' (Energy independent), 'True' (Energy dependent)
        dist        : The probability distribution function for the radii of different interfaces in the nanoparticles. Default: Gaussian
        norm        : The density of the nanoparticles in NanoMolar (NanoMoles/Liter)
        norm_err    : Percentage of error on normalization to simulated energy dependent SAXS data
        normQ       : power of 'Q' to normalize the Intensity such as [Q^0, Q^1, Q^2, Q^3, Q^4]
        error_factor : Error-factor to simulate the error-bars
        Rsig        : Widths of the distributions ('Rsig' in Angs) of radii of all the interfaces present in the nanoparticle system.
        bkg         : In-coherrent scattering background
        D           : Hard Sphere Diameter
        phi         : Volume fraction of particles
        U           : The sticky-sphere interaction energy
        SF          : Type of structure factor ['None','Hard-Sphere','Sticky-Sphere','Fractal']. Default: 'None'
        tol         : Tolerence for the Monte-Carlo Integration
        mpar        : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('SolDensity' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        Radii ('R' in Angs), and
                        Width of distribution in Radii ('Rsig' in Angs)
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.D=D
        self.phi=phi
        self.U=U
        # self.xi=xi
        # self.df=df
        self.SF=SF
        self.norm=norm
        self.norm_err=norm_err
        self.normQ=normQ
        self.dist=dist
        self.Np=Np
        self.tol=tol
        self.relement=relement
        self.NrDep=NrDep
        self.Energy=Energy
        self.Energy_Offset=Energy_Offset
        #self.rhosol=rhosol
        self.error_factor=error_factor
        self.bkg=bkg
        self.__mpar__=mpar #If there is any multivalued parameter
        self.choices={'dist':['Gaussian','LogNormal'],'NrDep':['True','False'],
                      'SF':['None','Hard-Sphere','Sticky-Sphere','Fractal'],
                      'normQ':[0,1,2,3,4]} #If there are choices available for any fixed parameters
        self.filepaths = {}  # If a parameter is a filename with path
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
        self.params.add('D',value=self.D,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('U',value=self.U,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('phi',value=self.phi,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('bkg',value=self.bkg,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('Energy_Offset',value=self.Energy_Offset,vary=0,min=-1e-2,max=1e-2,expr=None,brute_step=1e-3)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key!='Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d'%(mkey,key,i),value=self.__mpar__[mkey][key][i],vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)

    def calc_Rdist(self, R, Rsig, dist, N, seed=1):
        R=np.array(R)
        Rsig=np.array(Rsig)
        cov=np.diag(Rsig[:-1]**2)
        if dist=='Gaussian':
            mnorm = multivariate_normal(R[:-1], cov)
            Rl=mnorm.rvs(N,random_state=seed)
            rdist = mnorm.pdf(Rl)
        else:
            mnorm=multivariate_normal(np.log(R[:-1]), cov)
            Rl=np.exp(mnorm.rvs(N,random_state=seed))
            rdist = mnorm.pdf(np.log(Rl))
        Rl = np.vstack((Rl.T, np.zeros(Rl.shape[0]))).T
        return Rl, rdist


    # @lru_cache(maxsize=10)
    # def calc_Rdist(self, R, Rsig, dist, N):
    #     R = np.array(R)
    #     Rsig = np.array(Rsig)
    #     cov = np.diag(Rsig[:-1] ** 2)
    #     if dist == 'Gaussian':
    #         mnorm = multivariate_normal(R[:-1], cov)
    #         cmd = 'np.mgrid['
    #         for i, r in enumerate(R[:-1]):
    #             Rmin = max(r - 5 * Rsig[i], 0.0)
    #             Rmax = r + 5 * Rsig[i]
    #             cmd += '%.3f:%.3f:%dj,' % (Rmin, Rmax, N)
    #         cmd = cmd[:-1] + '].reshape(%d,-1).T' % len(R[:-1])
    #         Rl = eval(cmd)
    #         rdist = mnorm.pdf(Rl)
    #     else:
    #         mnorm = multivariate_normal(np.log(R[:-1]), cov)
    #         cmd = 'np.mgrid['
    #         for i, r in enumerate(R[:-1]):
    #             Rmin = max(-3, np.log(r) - 5 * Rsig[i])
    #             Rmax = np.log(r) + 5 * Rsig[i]
    #             cmd += '%.3f:%.3f:%dj,' % (Rmin, Rmax, N)
    #         cmd = cmd[:-1] + '].reshape(%d,-1).T' % len(R[:-1])
    #         Rl = np.exp(eval(cmd))
    #         rdist = mnorm.pdf(np.log(Rl))
    #     Rl = np.vstack((Rl.T, np.zeros(Rl.shape[0]))).T
    #     rdist = rdist / np.sum(rdist)
    #     if not self.__fit__:
    #         Rt = np.sqrt(np.sum(Rl ** 2, axis=1))
    #         self.output_params['Distribution'] = {'x': np.sort(Rt), 'y': rdist[np.argsort(Rt)]}
    #     return Rl, rdist

    @lru_cache(maxsize=10)
    def new_sphere(self, q, R, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, tol=1e-3):
        q = np.array(q)
        Rl, rdist = self.calc_Rdist(R, Rsig, dist, Np, seed=1)
        form = np.zeros_like(q)
        # eiform = np.zeros_like(q)
        # aform = np.zeros_like(q)
        # cform = np.zeros_like(q)
        pfac = (4 * np.pi * 2.818e-5 * 1.0e-8) ** 2
        last=np.zeros_like(q)
        for i in range(Np):
            ff, mff = ff_sphere_ml(q, Rl[i], rho)
            form += rdist[i] * ff
            if i>10 and i%10 == 0:
                chisq=np.sum(((form-last)/form)**2)/len(q)
                last=1.0*form
                if chisq<tol:
                    break
        return pfac * form/np.sum(rdist[:i]), Rl[:i], rdist[:i]

    def update_params(self):
        mkey=self.__mkeys__[0]
        key='Density'
        Nmpar=len(self.__mpar__[mkey][key])
        self.__density__=tuple([self.params['__%s_%s_%03d'%(mkey,key,i)].value for i in range(Nmpar)])
        key='SolDensity'
        self.__solDensity__=tuple([self.params['__%s_%s_%03d'%(mkey,key,i)].value for i in range(Nmpar)])
        key='Rmoles'
        self.__Rmoles__=tuple([self.params['__%s_%s_%03d'%(mkey,key,i)].value for i in range(Nmpar)])
        key='R'
        self.__R__=tuple([self.params['__%s_%s_%03d'%(mkey,key,i)].value for i in range(Nmpar)])
        key = 'Rsig'
        self.__Rsig__ = tuple([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)])
        key='Material'
        self.__material__=tuple([self.__mpar__[mkey][key][i] for i in range(Nmpar)])


    def y(self):
        """
        Define the function in terms of x to return some value
        """
        scale = 1e27 / 6.022e23
        self.update_params()
        if type(self.x) == dict:
            sqf={}
            for key in self.x.keys():
                sq=[]
                Energy=float(key.split('_')[1].split('@')[1])+self.Energy_Offset
                rho,eirho,adensity,rhor,eirhor,adensityr,cdensityr=calc_rho(R=self.__R__,material=self.__material__,relement=self.relement,
                                                                       density=self.__density__, sol_density=self.__solDensity__,
                                                                       Energy=Energy, Rmoles= self.__Rmoles__, NrDep=self.NrDep)
                sphere, Rl, rdist = self.new_sphere(tuple(self.x[key]), tuple(self.__R__),
                                                                       tuple(self.__Rsig__), tuple(rho), tuple(eirho),
                                                                       tuple(adensity), dist=self.dist,
                                                                       Np=self.Np,tol=self.tol)  # in cm^-1
                sqf[key] = self.norm * 1e-9 * 6.022e20 * sphere
                if self.SF is None:
                    struct = np.ones_like(self.x[key])  # hard_sphere_sf(self.x[key], D = self.D, phi = 0.0)
                elif self.SF == 'Hard-Sphere':
                    struct = hard_sphere_sf(self.x[key], D=self.D, phi=self.phi)
                    # struct = js.sf.PercusYevick(self.x[key], self.D/2, eta=self.phi)[1]
                elif self.SF == 'Sticky-Sphere':
                    struct = sticky_sphere_sf(self.x[key], D=self.D, phi=self.phi, U=self.U, delta=0.01)
                    # struct = js.sf.stickyHardSphere(self.x[key],self.D,0.01,self.U,phi=self.phi)[1]
                    # tau = np.exp(self.U) * (self.D + 0.01) / 12.0 / 0.01
                    # struct = js.sf.adhesiveHardSphere(self.x[key], self.D / 2, tau, 0.01, eta=self.phi).array[1,:]
                # elif self.SF == 'Fractal':
                #     struct = js.sf.fractal(self.x[key],self.xi,self.D,df=self.df).array[1,:]*js.sf.PercusYevick(self.x[key], self.D/2, eta=self.phi)[1]
                sqf[key]=(sqf[key]*struct+self.bkg)*self.x[key]**self.normQ


            if not self.__fit__:
                for i, j in combinations(range(len(self.__R__[:-1])), 2):
                    self.output_params['R_%d_%d' % (i+1, j+1)] = {'x': Rl[:, i], 'y': Rl[:, j], 'names':['R_%d'%(i+1),'R_%d'%(j+1)]}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1], 'names':['r (Angs)', 'rho (el/Angs<sup>3</sup>)']}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1], 'names':['r (Angs)', 'eirho (el/Angs<sup>3</sup>)']}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1]*scale, 'names':['r (Angs)', 'adensity (Molar)']}
                self.output_params['cdensity_r'] = {'x': cdensityr[:, 0], 'y': cdensityr[:, 1], 'names':['r (Angs)', 'Density (el/Angs<sup>3</sup>)']}
                self.output_params['Structure_Factor']={'x':self.x[key],'y':struct, 'names':['q (Angs)', 'S(q)']}
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
                    sqerr = np.random.normal(normsignal*norm,scale=self.error_factor)
                    meta={'Energy':Energy}
                    self.output_params['simulated_w_err_' + Energy+'keV'] = {'x': self.x[key], 'y': sqerr * minsignal,
                                                                             'yerr': np.sqrt(normsignal) * minsignal*self.error_factor,
                                                                             'meta':meta,
                                                                             'names':['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)', 'I_err (cm<sup>-1</sup>)']}
        else:
            if self.Energy is not None:
                Energy = self.Energy + self.Energy_Offset
            else:
                Energy = None
            rho, eirho, adensity, rhor, eirhor, adensityr, cdensityr = calc_rho(R=self.__R__,material=self.__material__,relement=self.relement,
                                                                       density=self.__density__, sol_density=self.__solDensity__,
                                                                                Energy=Energy, Rmoles= self.__Rmoles__, NrDep=self.NrDep)
            if self.SF is None:
                struct = np.ones_like(self.x)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x, D=self.D, phi=self.phi)
                # struct = js.sf.PercusYevick(self.x, self.D/2, eta=self.phi)[1]
            elif self.SF == 'Sticky-Sphere':
                struct = sticky_sphere_sf(self.x, D=self.D, phi=self.phi, U=self.U, delta=0.01)
                # tau = np.exp(self.U) * (self.D + 0.01) / 12.0 / 0.01
                # struct = js.sf.adhesiveHardSphere(self.x, self.D/2, tau, 0.01, eta=self.phi).array[1,:]
            # elif self.SF == 'Fractal':
            #     struct = js.sf.fractal(self.x, self.xi, self.D, df=self.df).array[1, :]*js.sf.PercusYevick(self.x, self.D/2, eta=self.phi)[1]

            tsqf, Rl, rdist = self.new_sphere(tuple(self.x), self.__R__, self.__Rsig__, tuple(rho),
                                                      tuple(eirho), tuple(adensity), dist=self.dist, Np=self.Np,tol=self.tol)
            sqf=(6.022e20 * self.norm * 1e-9 * np.array(tsqf) * struct + self.bkg)*self.x**self.normQ
            if not self.__fit__:
                for i, j in combinations(range(len(self.__R__[:-1])),2):
                    self.output_params['R_%d_%d' % (i+1,j+1)] = {'x': Rl[:, i], 'y': Rl[:,j], 'names':['R_%d'%(i+1),'R_%d'%(j+1)]}
                signal = sqf/self.x**self.normQ
                minsignal = np.min(signal)
                normsignal = signal / minsignal
                norm = np.random.normal(self.norm, scale = self.norm_err / 100.0)
                sqerr = np.random.normal(normsignal * norm, scale = self.error_factor)
                meta={'Energy':self.Energy}
                tkeys = list(self.output_params.keys())
                for key in tkeys:
                    if 'simulated_w_err' in key:
                        del self.output_params[key]
                if self.Energy is not None:
                    self.output_params['simulated_w_err_%.3fkeV'%self.Energy] = {'x': self.x, 'y': sqerr * minsignal*self.x**self.normQ,
                                                                                 'yerr': np.sqrt(normsignal) * minsignal*self.error_factor*self.x**self.normQ,
                                                                                 'meta':meta,
                                                                                 'names':['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)', 'I_err (cm<sup>-1</sup>)']}
                else:
                    self.output_params['simulated_w_err'] = {'x': self.x, 'y': sqerr * minsignal*self.x**self.normQ,
                                                             'yerr': np.sqrt(normsignal) * minsignal*self.error_factor*self.x**self.normQ,
                                                             'meta':meta,
                                                             'names':['q (Angs<sup>-1</sup>)', 'I (cm<sup>-1</sup>)','I_err (cm<sup>-1</sup>)']}
                self.output_params['Structure_Factor'] = {'x': self.x, 'y': struct}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1], 'names':['r (Angs)', 'rho (el/Angs<sup>3</sup>)']}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1], 'names':['r (Angs)', 'eirho (el/Angs<sup>3</sup>)']}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1]*scale, 'names':['r (Angs)', 'adensity (Molar)']}
                self.output_params['cdensity_r'] = {'x': cdensityr[:, 0], 'y': cdensityr[:, 1], 'names':['r (Angs)', 'Density (g/cm<sup>3</sup>)']}
        return sqf



if __name__=='__main__':
    x = {'Total_E@11.9126': np.logspace(np.log10(0.003), np.log10(0.15), 500)}
    # ,'Total_E@11.9098': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.9072': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.9037': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8984': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8914': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8830': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8714': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8564': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8364': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8098': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.7748': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.7288': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.6673': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.5860': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.4796': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.3396': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.1567': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@10.9190': np.logspace(np.log10(0.003), np.log10(0.15), 500)}
    # x = np.linspace(0.003, 0.15, 500)
    fun=Sphere_Uniform_2_Edep(x=x)
    print(fun.y())
