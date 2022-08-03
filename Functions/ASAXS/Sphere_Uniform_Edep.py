####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines'))
from functools import lru_cache
####Please do not remove lines above####

####Import your modules below if needed####
# from FormFactors.Sphere import Sphere
# from ff_sphere import ff_sphere_ml
from PeakFunctions import LogNormal, Gaussian
from utils import find_minmax, calc_rho
from Structure_Factors import hard_sphere_sf, sticky_sphere_sf

from numba import njit, prange

@njit(parallel=True,cache=True)
def ff_sphere_ml(q,R,rho):
    Nlayers=len(R)
    aff=np.ones_like(q)*complex(0,0)
    ff=np.zeros_like(q)
    for i in prange(len(q)):
        fact = 0.0
        rt = 0.0
        for j in prange(1,Nlayers):
            rt = rt + R[j - 1]
            fact += (rho[j - 1] - rho[j]) * (np.sin(q[i] * rt) - q[i] * rt * np.cos(q[i] * rt)) / q[i] ** 3
        aff[i] = fact
        ff[i] = abs(fact) ** 2
    return ff,aff


class Sphere_Uniform_Edep: #Please put the class name same as the function name
    def __init__(self, x=0, Np=20, error_factor=1, bkg=0.0,dist='Gaussian', relement='Au', Energy=None, NrDep='False', norm=1.0,
                 D=1.0, phi=0.1, U=-1.0, SF='None',Rsig=0.0,
                 mpar={'Multilayers':{'Material':['Au','H2O'],'Density':[19.32,1.0],'SolDensity':[1.0,1.0],'Rmoles':[1.0,1.0],'R':[1.0,0.0]}}):
        """
        Documentation
        Calculates the Energy dependent form factor of multilayered nanoparticles with different materials

        x           : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array. For energy dependence you need to
        provide a dictionary like {'E_11.919':linspace(0.001,1.0,1000)}
        relement    : Resonant element of the nanoparticle. Default: 'Au'
        Np          : No. of points with which the size distribution will be computed. Default: 10
        Energy      : Energy of the X-rays
        NrDep       : Energy dependence of the non-resonant element. Default= 'False' (Energy independent), 'True' (Energy dependent)
        dist        : The probability distribution function for the radii of different interfaces in the nanoparticles. Default: Gaussian
        norm        : The density of the nanoparticles in nanoMolar (nanoMoles/Liter)
        error_factor : Error-factor to simulate the error-bars
        Rsig        : Widths of the distributions ('Rsig' in Angs) of radii of all the interfaces present in the nanoparticle system.
        bkg         : In-coherent scattering background
        D           : Hard Sphere Diameter
        phi         : Volume fraction of particles
        U           : The sticky-sphere interaction energy
        SF          : Type of structure factor. Default: 'None'
        mpar        : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('SolDensity' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        Radii ('R' in Angs), and
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.D=D
        self.phi=phi
        self.U=U
        self.SF=SF
        self.norm=norm
        self.dist=dist
        self.Np=Np
        self.relement=relement
        self.NrDep=NrDep
        self.Energy=Energy
        #self.rhosol=rhosol
        self.error_factor=error_factor
        self.Rsig=Rsig
        self.bkg=bkg
        self.__mpar__=mpar #If there is any multivalued parameter
        self.choices={'dist':['Gaussian','LogNormal'],'NrDep':['True','False'],
                      'SF':['None','Hard-Sphere','Sticky-Sphere']} #If there are choices available for any fixed parameters
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
        self.params.add('Rsig',value=self.Rsig,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('D',value=self.D,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('U',value=self.U,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('phi',value=self.phi,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('bkg',value=self.bkg,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key!='Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d'%(mkey,key,i),value=self.__mpar__[mkey][key][i],vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)


    @lru_cache(maxsize=10)
    def calc_Rdist(self, R, Rsig, dist, N):
        R = np.array(R)
        totalR = np.sum(R[:-1])
        if Rsig > 0.001:
            fdist = eval(dist + '.' + dist + '(x=0.001, pos=totalR, wid=Rsig)')
            if dist == 'Gaussian':
                rmin, rmax = max(0.001, totalR - 5 * Rsig), totalR + 5 * Rsig
                dr = np.linspace(rmin, rmax, N)
            else:
                rmin, rmax = max(-3, np.log(totalR) - 5 * Rsig), np.log(totalR) + 5 * Rsig
                dr = np.logspace(rmin, rmax, N,base=np.exp(1.0))
            fdist.x = dr
            rdist = fdist.y()
            sumdist = np.sum(rdist)
            rdist = rdist / sumdist
            return dr, rdist, totalR
        else:
            return [totalR], [1.0], totalR

    @lru_cache(maxsize=10)
    def new_sphere(self, q, R, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10):
        q = np.array(q)
        dr, rdist, totalR = self.calc_Rdist(R, Rsig, dist, Np)
        form = np.zeros_like(q)
        # eiform = np.zeros_like(q)
        # aform = np.zeros_like(q)
        # cform = np.zeros_like(q)
        pfac = (4 * np.pi * 2.818e-5 * 1.0e-8) ** 2
        for i in range(len(dr)):
            r = np.array(R) * (1 + (dr[i] - totalR) / totalR)
            ff, mff = ff_sphere_ml(q, r, rho)
            form = form + rdist[i] * ff
            # eiff, meiff = ff_sphere_ml(q, r, eirho)
            # eiform = eiform + rdist[i] * eiff
            # aff, maff = ff_sphere_ml(q, r, adensity)
            # aform = aform + rdist[i] * aff
            # cform = cform + rdist[i] * (meiff * maff.conjugate()+meiff.conjugate()*maff)
        return pfac * form#, pfac * eiform, pfac * aform, pfac * np.abs(cform)/2.0  # in cm^2

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
        key='Material'
        self.__material__=tuple([self.__mpar__[mkey][key][i] for i in range(Nmpar)])


    def y(self):
        """
        Define the function in terms of x to return some value
        """
        self.update_params()
        if type(self.x) == dict:
            sqf={}
            for key in self.x.keys():
                sq=[]
                Energy=float(key.split('_')[1].split('@')[1])
                rho,eirho,adensity,rhor,eirhor,adensityr=calc_rho(R=self.__R__,material=self.__material__,relement=self.relement,
                                                                       density=self.__density__, sol_density=self.__solDensity__,
                                                                       Energy=Energy, Rmoles= self.__Rmoles__, NrDep=self.NrDep)
                sqf[key] = self.norm*1e-9 * 6.022e20 * self.new_sphere(tuple(self.x[key]), tuple(self.__R__),
                                                                       self.Rsig, tuple(rho), tuple(eirho),
                                                                       tuple(adensity), dist=self.dist,
                                                                       Np=self.Np)  # in cm^-1
                if self.SF is None:
                    struct = np.ones_like(self.x[key])  # hard_sphere_sf(self.x[key], D = self.D, phi = 0.0)
                elif self.SF == 'Hard-Sphere':
                    struct = hard_sphere_sf(self.x[key], D=self.D, phi=self.phi)
                else:
                    struct = sticky_sphere_sf(self.x[key], D=self.D, phi=self.phi, U=self.U, delta=0.01)
                sqf[key]=sqf[key]*struct+self.bkg


            if not self.__fit__:
                dr, rdist, totalR = self.calc_Rdist(tuple(self.__R__), self.Rsig, self.dist, self.Np)
                self.output_params['Distribution'] = {'x': dr, 'y': rdist}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1]}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1]}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1]}
                self.output_params['Structure_Factor']={'x':self.x[key],'y':struct}
                tkeys=list(self.output_params.keys())
                for key in tkeys:
                    if 'simulated_w_err' in key:
                        del self.output_params[key]
                for key in self.x.keys():
                    Energy = key.split('_')[1].split('@')[1]
                    # sqerr = np.sqrt(self.flux * sqf[key] * svol)
                    # sqwerr = sqf[key] * svol * self.flux + 2 * (0.5 - np.random.rand(len(sqerr))) * sqerr
                    signal = sqf[key]
                    minsignal = np.min(signal)
                    normsignal = signal / minsignal
                    sqerr = np.random.normal(normsignal,scale=self.error_factor)
                    meta={'Energy':Energy}
                    self.output_params['simulated_w_err_' + Energy+'keV'] = {'x': self.x[key], 'y': sqerr * minsignal,
                                                                   'yerr': np.sqrt(normsignal) * minsignal*self.error_factor, 'meta':meta}


        else:
            rho, eirho, adensity, rhor, eirhor, adensityr = calc_rho(R=self.__R__,material=self.__material__,relement=self.relement,
                                                                       density=self.__density__, sol_density=self.__solDensity__,
                                                                       Energy=self.Energy, Rmoles= self.__Rmoles__, NrDep=self.NrDep)
            if self.SF is None:
                struct = np.ones_like(self.x)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x, D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x, D=self.D, phi=self.phi, U=self.U, delta=0.01)

            tsqf = self.new_sphere(tuple(self.x), tuple(self.__R__), self.Rsig, tuple(rho),
                                                      tuple(eirho), tuple(adensity), dist=self.dist, Np=self.Np)

            self.output_params['Total'] = {'x': self.x, 'y': self.norm*1e-9 * np.array(tsqf) * 6.022e20 * struct + self.bkg}

            if not self.__fit__:
                dr, rdist, totalR = self.calc_Rdist(tuple(self.__R__), self.Rsig, self.dist, self.Np)
                self.output_params['Distribution'] = {'x': dr, 'y': rdist}
                signal = 6.022e20 * self.norm*1e-9 * np.array(tsqf) * struct + self.bkg
                minsignal = np.min(signal)
                normsignal = signal / minsignal
                sqerr = np.random.normal(normsignal,scale=self.error_factor)
                meta={'Energy':self.Energy}
                tkeys=list(self.output_params.keys())
                for key in tkeys:
                    if 'simulated_w_err' in key:
                        del self.output_params[key]
                if self.Energy is not None:
                    self.output_params['simulated_w_err_%.3fkeV'%self.Energy] = {'x': self.x, 'y': sqerr * minsignal,'yerr': np.sqrt(normsignal) * minsignal*self.error_factor,'meta':meta}
                else:
                    self.output_params['simulated_w_err'] = {'x': self.x, 'y': sqerr * minsignal,
                                                                                   'yerr': np.sqrt(
                                                                                       normsignal) * minsignal*self.error_factor,'meta':meta}
                self.output_params['Structure_Factor'] = {'x': self.x, 'y': struct}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1]}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1]}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1]}
            sqf = tsqf
        return sqf



if __name__=='__main__':
    x = {'Total_E@25.514': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.5043': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.5001': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.4963': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.4914': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.4848': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.4767': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.4664': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.4534': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.4368': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.4147': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.3865': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.3507': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.3041': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.2453': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.1682': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@25.0625': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@24.7508': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@24.5140': np.logspace(np.log10(0.003), np.log10(0.15), 500)}
    #x = {'Total_E@11.9126': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.9098': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.9072': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.9037': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8984': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8914': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8830': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8714': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8564': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8364': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.8098': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.7748': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.7288': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.6673': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.5860': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.4796': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.3396': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@11.1567': np.logspace(np.log10(0.003), np.log10(0.15), 500),'Total_E@10.9190': np.logspace(np.log10(0.003), np.log10(0.15), 500)}
    # x = np.linspace(0.003, 0.15, 500)
    fun=Sphere_Uniform_Edep(x=x)
    print(fun.y())
