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
from ff_ellipsoid import ff_ellipsoid_ml_asaxs
from Chemical_Formula import Chemical_Formula
from PeakFunctions import LogNormal, Gaussian
from Structure_Factors import hard_sphere_sf, sticky_sphere_sf
from utils import find_minmax, calc_rho, create_steps
from functools import lru_cache
import time


class Biphasic_Ellipsoid_Uniform: #Please put the class name same as the function name
    def __init__(self, x=0, Np=10, flux=1e13, term='Total', dist='Gaussian', Energy=None, relement='Au', Nalf=200,
                 NrDep='False', norm=1.0, Rsig=0.0, sbkg=0.0, cbkg=0.0, abkg=0.0, D=1.0, phi=0.1, U=-1.0,
                 SF='None', mpar={'Phase_1':{'Material': ['Au', 'H2O'],
                                             'Density': [19.32, 1.0],
                                             'VolFrac': [1.0, 1.0],
                                             'Rmoles': [1.0, 0.0],
                                             'R': [1.0, 0.0],
                                             'RzRatio':[1.0,1.0]},
                                  'Phase_2': {'Material': ['Au', 'H2O'],
                                              'Density': [19.32, 1.0],
                                              'VolFrac': [1.0, 1.0],
                                              'Rmoles': [1.0, 0.0],
                                              'R': [1.0, 0.0],
                                              'RzRatio': [1.0, 1.0]},
                                  'Solvent': {'Material': ['H2O', 'H2O'],
                                              'Density': [0.0, 1.0],
                                              'VolFrac': [1.0, 1.0],
                                              'Rmoles': [1.0, 0.0],
                                              'R': [1.0, 0.0],
                                              'RzRatio': [1.0, 1.0]
                                  }}):
        """
        Documentation
        Calculates the Energy dependent form factor of multilayered oblate nanoparticles with different materials and different phases

        x           : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array
        relement    : Resonant element of the nanoparticle. Default: 'Au'
        Energy      : Energy of X-rays in keV at which the form-factor is calculated. Default: None
        Np          : No. of points with which the size distribution will be computed. Default: 10
        NrDep       : Energy dependence of the non-resonant element. Default= 'False' (Energy independent), 'True' (Energy dependent)
        dist        : The probablity distribution fucntion for the radii of different interfaces in the nanoparticles. Default: Gaussian
        Nalf        : Number of azumuthal angle points for angular averaging
        norm        : The density of the nanoparticles in Molar (Moles/Liter)
        sbkg        : Constant incoherent background for SAXS-term
        cbkg        : Constant incoherent background for cross-term
        abkg        : Constant incoherent background for Resonant-term
        flux        : Total X-ray flux to calculate the errorbar to simulate the errorbar for the fitted data
        term        : 'SAXS-term' or 'Cross-term' or 'Resonant-term' or 'Total'
        D           : Hard Sphere Diameter
        phi         : Volume fraction of particles
        U           : The sticky-sphere interaction energy
        SF          : Type of structure factor. Default: 'None'
        Rsig        : Width of distribution of radii
        mpar        : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('SolDensity' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        Radii ('R' in Angs), and
                        Height to Radii ratio ('RzRatio' ratio)
        """
        if type(x) == list:
            self.x = np.array(x)
        else:
            self.x = x
        self.Nalf = Nalf
        self.norm = norm
        self.sbkg = sbkg
        self.cbkg = cbkg
        self.abkg = abkg
        self.dist = dist
        self.Np = Np
        self.Energy = Energy
        self.relement = relement
        self.NrDep = NrDep
        # self.rhosol=rhosol
        self.flux = flux
        self.D = D
        self.phi = phi
        self.U = U
        self.Rsig = Rsig
        self.__mpar__ = mpar  # If there is any multivalued parameter
        self.SF = SF
        self.term = term
        self.__Density__ = {}
        self.__VolFrac__ = {}
        self.__R__ = {}
        self.__Rmoles__ = {}
        self.__material__ = {}
        self.__RzRatio__= {}
        self.choices = {'dist': ['Gaussian', 'LogNormal'], 'NrDep': ['True', 'False'],
                        'SF': ['None', 'Hard-Sphere', 'Sticky-Sphere'],
                        'term': ['SAXS-term', 'Cross-term', 'Resonant-term',
                                 'Total']}  # If there are choices available for any fixed parameters
        self.__cf__ = Chemical_Formula()
        self.__fit__ = False
        self.output_params = {}
        self.output_params = {'scaler_parameters': {}}
        self.__mkeys__=list(self.__mpar__.keys())
        self.init_params()

    def init_params(self):
        """
        Define all the fitting parameters like
        self.params.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params = Parameters()
        self.params.add('norm', value=self.norm, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('D', value=self.D, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('phi', value=self.phi, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sbkg', value=self.sbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('cbkg', value=self.cbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('abkg', value=self.abkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('U', value=self.U, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('Rsig', value=self.Rsig, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        mkey1 = self.__mkeys__[0]
        for key in self.__mpar__[mkey1].keys():
            if key != 'Material':
                for i in range(len(self.__mpar__[mkey1][key])):
                    self.params.add('__%s_%s_%03d' % (mkey1, key, i), value=self.__mpar__[mkey1][key][i], vary=0,
                                    min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        for mkey in self.__mkeys__[1:]:
            for key in self.__mpar__[mkey].keys():
                if key != 'Material' and key != 'R':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
                elif key == 'R' or key=='RzRatio':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=-np.inf, max=np.inf
                                        , expr='__%s_%s_%03d' % (mkey1, key, i), brute_step=0.1)

    @lru_cache(maxsize=10)
    def calc_Rdist(self, R, Rsig, dist, N):
        R = np.array(R)
        totalR = np.sum(R[:-1])
        if Rsig > 0.001:
            fdist = eval(dist + '.' + dist + '(x=0.001, pos=totalR, wid=Rsig)')
            if dist == 'Gaussian':
                rmin, rmax = max(0.001, totalR - 5 * Rsig), totalR + 5 * Rsig
            else:
                rmin, rmax = max(0.001, np.exp(np.log(totalR) - 5 * Rsig)), np.exp(np.log(totalR) + 5 * Rsig)
            dr = np.linspace(rmin, rmax, N)
            fdist.x = dr
            rdist = fdist.y()
            sumdist = np.sum(rdist)
            rdist = rdist / sumdist
            return dr, rdist, totalR
        else:
            return [totalR], [1.0], totalR

    @lru_cache(maxsize=2)
    def ellipsoid(self, q, R, RzRatio, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, Nalf=1000):
        q = np.array(q)
        dr, rdist, totalR = self.calc_Rdist(R, Rsig, dist, Np)
        form = np.zeros_like(q)
        eiform = np.zeros_like(q)
        aform = np.zeros_like(q)
        cform = np.zeros_like(q)
        pfac = (2.818e-5 * 1.0e-8) ** 2
        for i in range(len(dr)):
            r = np.array(R) * (1 + (dr[i] - totalR) / totalR)
            fft,ffs,ffc,ffr = ff_ellipsoid_ml_asaxs(q, r, RzRatio, rho, eirho, adensity, Nalf)
            form = form + rdist[i] * fft
            eiform = eiform + rdist[i] * ffs
            aform = aform + rdist[i] * ffr
            cform = cform + rdist[i] * ffc
        return pfac * form, pfac * eiform, pfac * aform, np.abs(pfac * cform)  # in cm^2

    @lru_cache(maxsize=2)
    def ellipsoid_dict(self, q, R, RzRatio, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, Nalf=1000):
        form, eiform, aform, cform = self.ellipsoid(q, R, RzRatio, Rsig, rho, eirho, adensity, dist=dist, Np=Np, Nalf=Nalf)
        sqf={'Total':form,'SAXS-term':eiform,'Resonant-term':aform,'Cross-term':cform}
        return sqf

    def update_params(self):
        for mkey in self.__mkeys__:
            key = 'Density'
            Nmpar = len(self.__mpar__[mkey][key])
            self.__Density__[mkey] = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
            key = 'VolFrac'
            self.__VolFrac__[mkey] = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
            key = 'Rmoles'
            self.__Rmoles__[mkey] = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
            key = 'R'
            self.__R__[mkey] = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
            key = 'RzRatio'
            self.__RzRatio__[mkey] = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
            key = 'Material'
            self.__material__[mkey] = [self.__mpar__[mkey][key][i] for i in range(Nmpar)]
        for mkey in self.__mkeys__[1:]:
            key1='R'
            key2='RzRatio'
            for i in range(Nmpar):
                self.params['__%s_%s_%03d'%(mkey,key1,i)].set(expr='__%s_%s_%03d'%(self.__mkeys__[0],key1,i))
                self.params['__%s_%s_%03d' % (mkey, key2, i)].set(expr='__%s_%s_%03d' % (self.__mkeys__[0], key2, i))
        mkey='Solvent'
        key='VolFrac'
        for i in range(Nmpar):
            self.params['__%s_%s_%03d'%(mkey,key,i)].set(expr='1.0-__Phase_1_VolFrac_%03d-__Phase_2_VolFrac_%03d'%(i,i))


    def y(self):
        """
        Define the function in terms of x to return some value
        """
        svol = 1.5 * 0.0172 ** 2 / 370 ** 2  # scattering volume in cm^3
        self.update_params()
        mkey='Solvent'
        sol_density=tuple(np.ones_like(self.__Density__[mkey]))
        R=self.__R__[mkey]
        rho, eirho, adensity, rhor, eirhor, adensityr = calc_rho(R=tuple(R),
                                                                 material=tuple(self.__material__[mkey]),
                                                                 relement=self.relement,
                                                                 density=tuple(self.__Density__[mkey]),
                                                                 sol_density=sol_density,
                                                                 Energy=self.Energy,
                                                                 Rmoles=tuple(self.__Rmoles__[mkey]),
                                                                 NrDep=self.NrDep)
        for mkey in self.__mkeys__:
            if mkey!='Solvent':
                trho, teirho, tadensity, trhor, teirhor, tadensityr = calc_rho(R=tuple(self.__R__[mkey]),
                                                                           material=tuple(self.__material__[mkey]),
                                                                           relement=self.relement,
                                                                           density=tuple(self.__Density__[mkey]),
                                                                           sol_density=sol_density,
                                                                           Energy=self.Energy,
                                                                           Rmoles=tuple(self.__Rmoles__[mkey]),
                                                                           NrDep=self.NrDep)
                vf=np.array(self.__VolFrac__[mkey])
                rho=rho+vf*trho
                eirho=eirho+vf*teirho
                adensity=adensity+vf*tadensity

        major=np.sum(self.__R__[self.__mkeys__[0]])
        minor=np.sum(np.array(self.__R__[self.__mkeys__[0]])*np.array(self.__RzRatio__[self.__mkeys__[0]]))
        self.output_params['scaler_parameters']['Diameter (Angstroms)']=2*major
        self.output_params['scaler_parameters']['Height (Angstroms)']=2*minor
        if self.dist=='LogNormal':
            dstd=2 * np.sqrt((np.exp(self.Rsig**2) - 1) * np.exp(2 * np.log(major) + self.Rsig**2))
            hstd=2 * np.sqrt((np.exp(self.Rsig**2) - 1) * np.exp(2 * np.log(minor) + self.Rsig**2))
            self.output_params['scaler_parameters']['Diameter_Std (Angstroms)'] = dstd
            self.output_params['scaler_parameters']['Height_Std (Angstroms)'] = hstd
        else:
            self.output_params['scaler_parameters']['Diameter_Std (Angstroms)'] =2*self.Rsig
            self.output_params['scaler_parameters']['Height_Std (Angstroms)'] = 2 * self.Rsig*minor/major
        if type(self.x) == dict:
            sqf = {}
            key='SAXS-term'
            sqft=self.ellipsoid_dict(tuple(self.x[key]), tuple(self.__R__[self.__mkeys__[0]]),
                                                                       tuple(self.__RzRatio__[self.__mkeys__[0]]), self.Rsig,
                                                                       tuple(rho), tuple(eirho), tuple(adensity),
                                                                       dist=self.dist, Np=self.Np, Nalf=self.Nalf)
            if self.SF is None:
                struct = np.ones_like(self.x[key])  # hard_sphere_sf(self.x[key], D = self.D, phi = 0.0)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x[key], D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x[key], D=self.D, phi=self.phi, U=self.U, delta=0.01)
            for key in self.x.keys():
                if key == 'SAXS-term':
                    sqf[key] = self.norm * 6.022e20 *sqft[key] * struct + self.sbkg # in cm^-1
                if key == 'Cross-term':
                    sqf[key] = self.norm * 6.022e20 *sqft[key] * struct + self.cbkg # in cm^-1
                if key == 'Resonant-term':
                    sqf[key] = self.norm * 6.022e20 *sqft[key] * struct + self.abkg # in cm^-1
            key1='Total'
            total= self.norm * 6.022e20 *sqft[key1] * struct + self.sbkg
            if not self.__fit__:
                dr, rdist, totalR = self.calc_Rdist(tuple(self.__R__[self.__mkeys__[0]]), self.Rsig, self.dist, self.Np)
                self.output_params['Distribution'] = {'x': dr, 'y': rdist}
                self.output_params['Total'] = {'x': self.x[key], 'y': total}
                for key in self.x.keys():
                    self.output_params[key] = {'x': self.x[key], 'y': sqf[key]}
                xtmp, ytmp = create_steps(self.__R__[self.__mkeys__[0]],rho)
                self.output_params['rho_r'] = {'x': xtmp, 'y': ytmp}
                xtmp, ytmp = create_steps(self.__R__[self.__mkeys__[0]],eirho)
                self.output_params['eirho_r'] = {'x': xtmp, 'y': ytmp}
                xtmp, ytmp = create_steps(self.__R__[self.__mkeys__[0]], adensity)
                self.output_params['adensity_r'] = {'x': xtmp, 'y': ytmp}
                self.output_params['Structure_Factor'] = {'x': self.x[key], 'y': struct}
                # xtmp,ytmp=create_steps(x=self.__R__[][:-1],y=self.__Rmoles__[:-1])
                # self.output_params['Rmoles_radial']={'x':xtmp,'y':ytmp}
                # xtmp, ytmp = create_steps(x=self.__R__[:-1], y=self.__RzRatio__[:-1])
                # self.output_params['RzRatio_radial'] = {'x': xtmp, 'y': ytmp}
                # xtmp, ytmp = create_steps(x=self.__R__[:-1], y=self.__density__[:-1])
                # self.output_params['Density_radial'] = {'x': xtmp, 'y': ytmp}
        else:
            if self.SF is None:
                struct = np.ones_like(self.x)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x, D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x, D=self.D, phi=self.phi, U=self.U, delta=0.01)

            tsqf, eisqf, asqf, csqf = self.ellipsoid(tuple(self.x), tuple(self.__R__[self.__mkeys__[0]]),
                                                     tuple(self.__RzRatio__[self.__mkeys__[0]]), self.Rsig,
                                                     tuple(rho), tuple(eirho),
                                                      tuple(adensity), dist=self.dist, Np=self.Np, Nalf=self.Nalf)
            sqf = self.norm * np.array(tsqf) * 6.022e20 * struct + self.sbkg  # in cm^-1
            # if not self.__fit__: #Generate all the quantities below while not fitting
            asqf = self.norm * np.array(asqf) * 6.022e20 * struct + self.abkg  # in cm^-1
            eisqf = self.norm * np.array(eisqf) * 6.022e20 * struct + self.sbkg  # in cm^-1
            csqf = self.norm * np.array(csqf) * 6.022e20 * struct + self.cbkg  # in cm^-1
            sqerr = np.sqrt(6.022e20*self.norm*self.flux * tsqf * svol+self.sbkg)
            sqwerr = (6.022e20*self.norm*tsqf * svol * self.flux+self.sbkg + 2 * (0.5 - np.random.rand(len(tsqf))) * sqerr)
            dr, rdist, totalR = self.calc_Rdist(tuple(self.__R__[self.__mkeys__[0]]), self.Rsig, self.dist, self.Np)
            self.output_params['Distribution'] = {'x': dr, 'y': rdist}
            self.output_params['simulated_total_w_err'] = {'x': self.x, 'y': sqwerr, 'yerr': sqerr}
            self.output_params['Total'] = {'x': self.x, 'y': sqf}
            self.output_params['Resonant-term'] = {'x': self.x, 'y': asqf}
            self.output_params['SAXS-term'] = {'x': self.x, 'y': eisqf}
            self.output_params['Cross-term'] = {'x': self.x, 'y': csqf}
            xtmp, ytmp = create_steps(self.__R__[self.__mkeys__[0]], rho)
            self.output_params['rho_r'] = {'x': xtmp, 'y': ytmp}
            xtmp, ytmp = create_steps(self.__R__[self.__mkeys__[0]], eirho)
            self.output_params['eirho_r'] = {'x': xtmp, 'y': ytmp}
            xtmp, ytmp = create_steps(self.__R__[self.__mkeys__[0]], adensity)
            self.output_params['adensity_r'] = {'x': xtmp, 'y': ytmp}
            self.output_params['Structure_Factor'] = {'x': self.x, 'y': struct}
            # xtmp, ytmp = create_steps(x=self.__R__[:-1], y=self.__Rmoles__[:-1])
            # self.output_params['Rmoles_radial'] = {'x':xtmp , 'y': ytmp}
            # sqf = self.output_params[self.term]['y']
            # xtmp, ytmp = create_steps(x=self.__R__[:-1], y=self.__RzRatio__[:-1])
            # self.output_params['RzRatio_radial'] = {'x': xtmp, 'y': ytmp}
            # xtmp, ytmp = create_steps(x=self.__R__[:-1], y=self.__density__[:-1])
            # self.output_params['Density_radial'] = {'x': xtmp, 'y': ytmp}
        return sqf


if __name__=='__main__':
    x=np.arange(0.001,1.0,0.1)
    fun=Biphasic_Ellipsoid_Uniform(x=x)
    print(fun.y())
