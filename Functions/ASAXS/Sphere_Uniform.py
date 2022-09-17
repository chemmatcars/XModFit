####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines'))
####Please do not remove lines above####

####Import your modules below if needed####
# from FormFactors.Sphere import Sphere
# from ff_sphere import ff_sphere_ml
from Chemical_Formula import Chemical_Formula
from PeakFunctions import LogNormal, Gaussian
from Structure_Factors import hard_sphere_sf, sticky_sphere_sf
from utils import find_minmax, calc_rho, create_steps
from functools import lru_cache
import time

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




class Sphere_Uniform: #Please put the class name same as the function name
    def __init__(self, x=0, Np=20, error_factor=1.0, term='Total',dist='Gaussian', Energy=None, relement='Au', NrDep='True',
                 norm=1.0, norm_err=0.01, sbkg=0.0, cbkg=0.0, abkg=0.0, D=1.0, phi=0.1, U=-1.0, SF='None',Rsig=0.0,
                 mpar={'Layers':{'Material':['Au','H2O'],'Density':[19.32,1.0],'SolDensity':[1.0,1.0],'Rmoles':[1.0,1.0],'R':[1.0,0.0]}}):
        """
        Calculates the Energy dependent form factor of multilayered nanoparticles with different materials

        x            : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array
        relement     : Resonant element of the nanoparticle. Default: 'Au'
        Energy       : Energy of X-rays in keV at which the form-factor is calculated. Default: None
        Np           : No. of points with which the size distribution will be computed. Default: 10
        NrDep        : Energy dependence of the non-resonant element. Default= 'False' (Energy independent), 'True' (Energy dependent)
        dist         : The probability distribution function for the radii of different interfaces in the nanoparticles. Default: Gaussian
        norm         : The density of the nanoparticles in nanoMolar (nanoMoles/Liter)
        norm_err     : Percentage of error on normalization to simulated energy dependent SAXS data
        sbkg         : Constant incoherent background for SAXS-term
        cbkg         : Constant incoherent background for cross-term
        abkg         : Constant incoherent background for Resonant-term
        error_factor : Error-factor to simulate the error-bars
        term         : 'SAXS-term' or 'Cross-term' or 'Resonant-term' or 'Total'
        D            : Hard Sphere Diameter
        phi          : Volume fraction of particles
        U            : The sticky-sphere interaction energy
        SF           : Type of structure factor. Default: 'None'
        Rsig         : Widths of the total radius of the nanoparticles. Default: 0.0
        mpar         : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('SolDensity' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        Radii ('R' in Angs)

        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.norm=norm
        self.norm_err=norm_err
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
        self.D=D
        self.phi=phi
        self.U=U
        self.__mpar__=mpar #If there is any multivalued parameter
        self.SF=SF
        self.term=term
        self.Rsig=Rsig
        self.choices={'dist':['Gaussian','LogNormal'],'NrDep':['True','False'],'SF':['None','Hard-Sphere', 'Sticky-Sphere'],
                      'term':['SAXS-term','Cross-term','Resonant-term','Total']} #If there are choices available for any fixed parameters
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
        self.params.add('D', value=self.D, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('phi', value=self.phi, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sbkg',value=self.sbkg,vary=0, min = -np.inf, max = np.inf, expr = None, brute_step = 0.1)
        self.params.add('cbkg', value=self.cbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('abkg', value=self.abkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('U', value=self.U, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('Rsig',value=self.Rsig,vary=0,min=0,max=np.inf,expr=None,brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key!='Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d'%(mkey, key,i),value=self.__mpar__[mkey][key][i],vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)


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
                dr = np.logspace(rmin, rmax, N, base=np.exp(1.0))
            fdist.x = dr
            rdist = fdist.y()
            sumdist = np.sum(rdist)
            rdist = rdist / sumdist
            return dr, rdist, totalR
        else:
            return [totalR], [1.0], totalR

    @lru_cache(maxsize=10)
    def new_sphere(self, q, R, Rsig, rho, eirho, adensity, dist='Gaussian',Np=10):
        q = np.array(q)
        dr, rdist, totalR = self.calc_Rdist(R, Rsig, dist, Np)
        form = np.zeros_like(q)
        eiform = np.zeros_like(q)
        aform = np.zeros_like(q)
        cform = np.zeros_like(q)
        pfac = (4 * np.pi * 2.818e-5 * 1.0e-8) ** 2
        for i in range(len(dr)):
            r = np.array(R) * (1 + (dr[i] - totalR) / totalR)
            ff, mff = ff_sphere_ml(q, r, rho)
            form = form + rdist[i] * ff
            eiff, meiff = ff_sphere_ml(q, r, eirho)
            eiform = eiform + rdist[i] * eiff
            aff, maff = ff_sphere_ml(q, r, adensity)
            aform = aform + rdist[i] * aff
            cform = cform + rdist[i] * (meiff * maff.conjugate()+meiff.conjugate()*maff)
        return pfac * form, pfac * eiform, pfac * aform, np.abs(pfac * cform)/2  # in cm^2

    @lru_cache(maxsize=2)
    def new_sphere_dict(self, q, R, Rsig, rho, eirho, adensity, dist='Gaussian',Np=10,key='SAXS-term'):
        form, eiform, aform, cform = self.new_sphere(q, R, Rsig, rho, eirho, adensity,dist=dist,Np=Np)
        if key == 'SAXS-term':
            return eiform
        elif key == 'Resonant-term':
            return aform
        elif key == 'Cross-term':
            return cform
        elif key == 'Total':
            return form

    def update_params(self):
        mkey=self.__mkeys__[0]
        key = 'Density'
        Nmpar=len(self.__mpar__[mkey][key])
        self.__density__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'SolDensity'
        self.__sol_density__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'Rmoles'
        self.__Rmoles__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'R'
        self.__R__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'Material'
        self.__material__ = [self.__mpar__[mkey][key][i] for i in range(Nmpar)]

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        scale = 1e27 / 6.022e23
        svol = 1.5 * 0.0172 ** 2 / 370 ** 2  # scattering volume in cm^3
        self.output_params = {'scaler_parameters': {}}
        self.update_params()
        rho, eirho, adensity, rhor, eirhor, adensityr = calc_rho(R=tuple(self.__R__),
                                                                      material=tuple(self.__material__),
                                                                      relement=self.relement,
                                                                      density=tuple(self.__density__),
                                                                      sol_density=tuple(self.__sol_density__),
                                                                      Energy=self.Energy, Rmoles=tuple(self.__Rmoles__),
                                                                      NrDep=self.NrDep)
        if type(self.x) == dict:
            sqf = {}
            for key in self.x.keys():
                sqf[key] = self.norm*1e-9 * 6.022e20 * self.new_sphere_dict(tuple(self.x[key]), tuple(self.__R__),
                                                                       self.Rsig, tuple(rho), tuple(eirho),
                                                                       tuple(adensity), key=key, dist=self.dist,Np=self.Np)  # in cm^-1
                if self.SF is None:
                    struct = np.ones_like(self.x[key])  # hard_sphere_sf(self.x[key], D = self.D, phi = 0.0)
                elif self.SF == 'Hard-Sphere':
                    struct = hard_sphere_sf(self.x[key], D=self.D, phi=self.phi)
                else:
                    struct = sticky_sphere_sf(self.x[key], D=self.D, phi=self.phi, U=self.U, delta=0.01)
                if key == 'SAXS-term':
                    sqf[key] = sqf[key] * struct + self.sbkg
                if key == 'Cross-term':
                    sqf[key] = sqf[key] * struct + self.cbkg
                if key == 'Resonant-term':
                    sqf[key] = sqf[key] * struct + self.abkg
            key1 = 'Total'
            total = self.norm*1e-9 * 6.022e20 * struct * self.new_sphere_dict(tuple(self.x[key]), tuple(self.__R__),
                                                                         self.Rsig, tuple(rho), tuple(eirho),
                                                                         tuple(adensity),
                                                                         key=key1,dist=self.dist,Np=self.Np) + self.sbkg  # in cm^-1
            if not self.__fit__:
                dr, rdist, totalR = self.calc_Rdist(tuple(self.__R__), self.Rsig, self.dist, self.Np)
                self.output_params['Distribution'] = {'x': dr, 'y': rdist}
                signal = total
                minsignal = np.min(signal)
                normsignal = signal / minsignal
                norm = np.random.normal(self.norm, scale = self.norm_err / 100.0)
                sqerr = np.random.normal(normsignal * norm, scale = self.error_factor)
                meta={'Energy':self.Energy}
                if self.Energy is not None:
                    self.output_params['simulated_w_err_%.4fkeV'%self.Energy] = {'x': self.x[key], 'y': sqerr*minsignal, 'yerr': np.sqrt(normsignal)*minsignal*self.error_factor,'meta':meta}
                else:
                    self.output_params['simulated_w_err'] = {'x': self.x[key], 'y': sqerr * minsignal, 'yerr': np.sqrt(normsignal) * minsignal}
                self.output_params['Total'] = {'x': self.x[key], 'y': total}
                for key in self.x.keys():
                    self.output_params[key] = {'x': self.x[key], 'y': sqf[key]}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1],
                                               'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1],
                                                 'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1] * scale,
                                                    'names': ['r (Angs)', 'Density (Molar)']}  # in Molar
                self.output_params['Structure_Factor'] = {'x': self.x[key], 'y': struct}
                xtmp, ytmp = create_steps(x=self.__R__[:-1], y=self.__Rmoles__[:-1])
                self.output_params['Rmoles_radial'] = {'x': xtmp, 'y': ytmp}
                xtmp, ytmp = create_steps(x=self.__R__[:-1], y=self.__density__[:-1])
                self.output_params['Density_radial'] = {'x': xtmp, 'y': ytmp}
        else:
            if self.SF is None:
                struct = np.ones_like(self.x)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x, D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x, D=self.D, phi=self.phi, U=self.U, delta=0.01)

            tsqf, eisqf, asqf, csqf = self.new_sphere(tuple(self.x), tuple(self.__R__), self.Rsig, tuple(rho),
                                                      tuple(eirho), tuple(adensity),dist=self.dist,Np=self.Np)
            sqf = self.norm*1e-9 * np.array(tsqf) * 6.022e20 * struct + self.sbkg  # in cm^-1
            if not self.__fit__: #Generate all the quantities below while not fitting
                asqf = self.norm*1e-9 * np.array(asqf) * 6.022e20 * struct + self.abkg  # in cm^-1
                eisqf = self.norm*1e-9 * np.array(eisqf) * 6.022e20 * struct + self.sbkg  # in cm^-1
                csqf = self.norm*1e-9 * np.array(csqf) * 6.022e20 * struct + self.cbkg  # in cm^-1
                # sqerr = np.sqrt(6.020e20*self.flux *self.norm*tsqf*struct*svol+self.sbkg)
                # sqwerr = (6.022e20*tsqf * svol * self.flux*self.norm*struct + self.sbkg + 2 * (0.5 - np.random.rand(len(tsqf))) * sqerr)
                signal = 6.022e20 * self.norm*1e-9 * np.array(tsqf) * struct + self.sbkg
                minsignal=np.min(signal)
                normsignal=signal/minsignal
                norm = np.random.normal(self.norm, scale = self.norm_err / 100.0)
                sqerr=np.random.normal(normsignal * norm,scale = self.error_factor)
                meta={'Energy':self.Energy}
                if self.Energy is not None:
                    self.output_params['simulated_w_err_%.4fkeV'%self.Energy] = {'x': self.x, 'y': sqerr*minsignal, 'yerr': np.sqrt(normsignal)*minsignal*self.error_factor,'meta':meta}
                else:
                    self.output_params['simulated_w_err'] = {'x': self.x, 'y': sqerr * minsignal, 'yerr': np.sqrt(normsignal) * minsignal*self.error_factor,'meta':meta}
                dr, rdist, totalR = self.calc_Rdist(tuple(self.__R__), self.Rsig, self.dist, self.Np)
                self.output_params['Distribution'] = {'x': dr, 'y': rdist}
                self.output_params['Total'] = {'x': self.x, 'y': signal}
                self.output_params['Resonant-term'] = {'x': self.x, 'y': asqf}
                self.output_params['SAXS-term'] = {'x': self.x, 'y': eisqf}
                self.output_params['Cross-term'] = {'x': self.x, 'y': csqf}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1],
                                               'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1],
                                                 'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1] * scale,
                                                    'names': ['r (Angs)', 'Density (Molar)']}  # in Molar
                self.output_params['Structure_Factor'] = {'x': self.x, 'y': struct}
                xtmp, ytmp = create_steps(x=self.__R__[:-1], y=self.__Rmoles__[:-1])
                self.output_params['Rmoles_radial'] = {'x': xtmp, 'y': ytmp}
                xtmp, ytmp = create_steps(x=self.__R__[:-1], y=self.__density__[:-1])
                self.output_params['Density_radial'] = {'x': xtmp, 'y': ytmp}
            sqf = self.output_params[self.term]['y']
        return sqf


if __name__=='__main__':
    x=np.logspace(-3,-0.8,500)
    fun=Sphere_Uniform(x=x)
    print(fun.y())
