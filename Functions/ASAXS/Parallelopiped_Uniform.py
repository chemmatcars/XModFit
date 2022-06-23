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
from Chemical_Formula import Chemical_Formula
from PeakFunctions import LogNormal, Gaussian
from Structure_Factors import hard_sphere_sf, sticky_sphere_sf
# from ff_cylinder import ff_cylinder_ml_asaxs
from utils import find_minmax, calc_rho, create_steps

from numba import njit, prange
from scipy.special import j1
import numba_scipy.special

@njit(parallel=True,cache=True, fastmath=True)
def parallelopiped_ml_asaxs(q, L, B, H, rho, eirho, adensity, Nphi, Npsi, HggtLB):
    if HggtLB:
        tfac=2*np.pi/q/H
        Nphi=1
        dphi=1.0
        Nqt=-1.0
    else:
        tfac=np.ones_like(q)
        dphi=np.pi/Nphi
        Nqt=1.0
    dphi=np.pi/Nphi
    dpsi=2.0*np.pi/Npsi
    fft = np.zeros_like(q)
    ffs = np.zeros_like(q)
    ffc = np.zeros_like(q)
    ffr = np.zeros_like(q)
    L=np.cumsum(L)
    B=np.cumsum(B)
    Nlayers=len(L)
    V = L[:-1]*B[:-1]*H
    drho=2.0*np.diff(np.array(rho))*V
    deirho=2.0*np.diff(np.array(eirho))*V
    dadensity=2.0*np.diff(np.array(adensity))*V
    dphidpsi=dphi*dpsi
    for i in prange(len(q)):
        for iphi in prange(0, Nphi+1):
            phi = iphi*dphi
            qh = q[i]*H*np.cos(phi)/2.0
            shfac = ((1.0 - Nqt) + (1.0 + Nqt) * np.sinc(qh / np.pi)) / 2.0
            sphi=((1.0-Nqt)+(1.0+Nqt)*np.sin(phi))/2.0
            qsphi=q[i]*sphi/2.0
            for ipsi in prange(0, Npsi+1):
                psi = ipsi*dpsi
                tft = np.complex(0.0, 0.0)
                tfs = 0.0
                tfr = 0.0
                sc=qsphi*np.cos(psi)
                ss=qsphi*np.sin(psi)
                for k in prange(Nlayers-1):
                    ql=L[k]*sc
                    qb=B[k]*ss
                    fac=np.sinc(ql/np.pi)*np.sinc(qb/np.pi)*shfac
                    tft += drho[k] * fac
                    tfs += deirho[k] * fac
                    tfr += dadensity[k] * fac
                fft[i] +=  np.abs(tft)**2* sphi
                ffs[i] +=  tfs**2* sphi
                ffc[i] +=  tfs*tfr*sphi
                ffr[i] +=  tfr**2*sphi
    fft*=dphidpsi
    ffs*=dphidpsi
    ffc*=dphidpsi
    ffr*=dphidpsi
    return fft*tfac,ffs*tfac,ffc*tfac,ffr*tfac

class Parallelopiped_Uniform: #Please put the class name same as the function name
    def __init__(self, x=0, Np=10, error_factor=1.0, dist='Gaussian', Energy=None, relement='Au', NrDep='False', L=1.0, B=1.0, H=1.0,
                 HggtLB=True, sig=0.0, norm=1.0, sbkg=0.0, cbkg=0.0, abkg=0.0, D=1.0, phi=0.1, U=-1.0, SF='None',Nphi=180,Npsi=360, term='Total',
                 mpar={'Layers': {'Material': ['Au', 'H2O'], 'Density': [19.32, 1.0], 'SolDensity': [1.0, 1.0],
                                  'Rmoles': [1.0, 1.0], 'Thickness': [0.0, 0.0]}}):
        """
        Documentation
        Calculates the Energy dependent form factor of multilayered cylinders with different materials

        x           : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array
        relement    : Resonant element of the nanoparticle. Default: 'Au'
        Energy      : Energy of X-rays in keV at which the form-factor is calculated. Default: None
        Np          : No. of points with which the size distribution will be computed. Default: 10
        L           : Length of the parallelelopiped in Angs
        B           : Breadth of the parallelopiped in Angs
        H           : Height of the parallelopiped in Angs
        HggtLB      : True if H>>L or B to use parallelopiped with infinite height
        NrDep       : Energy dependence of the non-resonant element. Default= 'False' (Energy independent), 'True' (Energy independent)
        dist        : The probablity distribution fucntion for the radii of different interfaces in the nanoparticles. Default: Gaussian
        sig         : Width of distribution or thicknesses of the layers of the cylinder
        Nphi        : Number of polar angle points for angular averaging
        Npsi        : Number of azimuthal angle for angular averaging
        norm        : The density of the nanoparticles in Molar (Moles/Liter)
        sbkg        : Constant incoherent background for SAXS-term
        cbkg        : Constant incoherent background for cross-term
        abkg        : Constant incoherent background for Resonant-term
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
                        Thicknesses ('Thickness' in Angs) of the layers starting with 0 for the first layer as the first layer thickness i provided by L,B, and H values
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.norm=norm
        self.sbkg=sbkg
        self.cbkg=cbkg
        self.abkg=abkg
        self.dist=dist
        self.sig=sig
        self.Np=Np
        self.L=L
        self.B=B
        self.H=H
        self.HggtLB=HggtLB
        self.Nphi=Nphi
        self.Npsi=Npsi
        self.Energy=Energy
        self.relement=relement
        self.NrDep=NrDep
        self.error_factor=error_factor
        self.D=D
        self.phi=phi
        self.U=U
        self.term=term
        self.__mpar__=mpar #If there is any multivalued parameter
        self.SF=SF
        self.choices={'dist':['Gaussian','LogNormal'],'NrDep':['True','False'],
                      'SF':['None','Hard-Sphere', 'Sticky-Sphere'],
                      'term': ['SAXS-term', 'Cross-term', 'Resonant-term',
                               'Total'],
                      'HggtLB':['True','False']
                      } #If there are choices available for any fixed parameters
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
        self.params.add('L', value=self.L, vary=0, min=1e-3, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('B', value=self.B, vary=0, min=1e-3, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('H', value=self.H, vary=0, min=1e-3, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sbkg',value=self.sbkg,vary=0, min = -np.inf, max = np.inf, expr = None, brute_step = 0.1)
        self.params.add('cbkg', value=self.cbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('abkg', value=self.abkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('U', value=self.U, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sig', value=self.sig, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        for mkey in self.__mpar__.keys():
            for key in self.__mpar__[mkey].keys():
                if key != 'Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=0.0,
                                        max=np.inf, expr=None, brute_step=0.1)
    @lru_cache(maxsize=10)
    def calc_LBdist(self, L, B, sig, dist, N):
        L = np.array(L)
        B = np.array(B)
        totalL = np.sum(L[:-1])
        totalB = np.sum(B[:-1])
        if sig > 0.001:
            fdist = eval(dist + '.' + dist + '(x=0.001, pos=totalL, wid=sig)')
            if dist=='Gaussian':
                Lmin, Lmax = max(0.001, totalL - 5 * sig), totalL + 5 * sig
                Bmin, Bmax = max(0.001, totalB - 5 * sig), totalB + 5 * sig
                dL = np.linspace(Lmin, Lmax, N)
                dB = np.linspace(Bmin, Bmax, N)
            else:
                Lmin, Lmax= max(-3, np.log(totalL) - 5*sig), np.log(totalL) + 5*sig
                Bmin, Bmax = max(-3, np.log(totalB) - 5 * sig), np.log(totalB) + 5 * sig
                dL = np.logspace(Lmin, Lmax, N, base=np.exp(1.0))
                dB = np.logspace(Bmin, Bmax, N, base=np.exp(1.0))
            fdist.x = dL
            dist = fdist.y()
            # sumdist = np.sum(Ldist)
            # Ldist = Ldist / sumdist
            return dL, totalL, dB, totalB, dist
        else:
            return [totalL], totalL, [totalB], totalB, [1.0]

    @lru_cache(maxsize=10)
    def parallelopiped(self, q, L, B, H, sig, rho, eirho, adensity, dist='Gaussian', Np=10, Nphi=200, Npsi=400, HggtLB=True):
        q = np.array(q)
        dL, totalL, dB, totalB, dist = self.calc_LBdist(L, B, sig, dist, Np)
        form = np.zeros_like(q)
        eiform = np.zeros_like(q)
        aform = np.zeros_like(q)
        cform = np.zeros_like(q)
        pfac = (2.818e-5 * 1.0e-8) ** 2
        sumL=np.sum(dist)
        for i in range(len(dL)):
            l = np.array(L) * (1 + (dL[i] - totalL) / totalL)
            b = np.array(B) * (1 + (dB[i] - totalB) / totalB)
            # fft, ffs, ffc, ffr = ff_cylinder_ml_asaxs(q, H, r, rho, eirho, adensity, Nalf)
            fft, ffs, ffc, ffr = parallelopiped_ml_asaxs(q, l, b, H, rho, eirho, adensity, Nphi, Npsi,HggtLB=HggtLB)
            form += dist[i] * fft/sumL
            eiform += dist[i] * ffs/sumL
            aform += dist[i] * ffr/sumL
            cform += dist[i] * ffc/sumL
        return pfac * form, pfac * eiform, pfac * aform, np.abs(pfac * cform)  # in cm^2

    @lru_cache(maxsize=10)
    def parallelopiped_dict(self, q, L, B, H, sig, rho, eirho, adensity, dist='Gaussian', Np=10, Nphi=200, Npsi=400, HggtLB=True):
        form, eiform, aform, cform = self.parallelopiped(q, L, B, H, sig, rho, eirho, adensity, dist=dist, Np=Np,
                                                    Nphi=Nphi, Npsi=Npsi,HggtLB=HggtLB)
        sqf = {'Total': form, 'SAXS-term': eiform, 'Resonant-term': aform, 'Cross-term': cform}
        return sqf

    def update_params(self):
        mkey = self.__mkeys__[0]
        key = 'Density'
        Nmpar = len(self.__mpar__[mkey][key])
        self.__density__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'SolDensity'
        self.__solDensity__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'Rmoles'
        self.__Rmoles__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'Thickness'
        self.__Thickness__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(Nmpar)]
        key = 'Material'
        self.__material__ = [self.__mpar__[mkey][key][i] for i in range(Nmpar)]

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        scale = 1e27 / 6.022e23
        svol = 1.5*0.0172**2/370**2  # scattering volume in cm^3
        self.update_params()
        self.__L__=np.array(self.__Thickness__)
        self.__B__=np.array(self.__Thickness__)
        self.__L__[0]=self.L
        self.__B__[0]=self.B
        rho, eirho, adensity, rhor, eirhor, adensityr = calc_rho(R=tuple(self.__L__), material=tuple(self.__material__),
                                                                 relement=self.relement,
                                                                 density=tuple(self.__density__),
                                                                 sol_density=tuple(self.__solDensity__),
                                                                 Energy=self.Energy, Rmoles=tuple(self.__Rmoles__),
                                                                 NrDep=self.NrDep)
        if type(self.x) == dict:
            sqf = {}
            key='SAXS-term'
            sqft=self.parallelopiped_dict(tuple(self.x[key]), tuple(self.__L__), tuple(self.__B__),
                                          self.H, self.sig,
                                          tuple(rho), tuple(eirho), tuple(adensity),
                                          dist = self.dist, Np = self.Np, Nphi = self.Nphi,
                                          Npsi = self.Npsi,HggtLB=self.HggtLB)
            if self.SF is None:
                struct = np.ones_like(self.x[key])  # hard_sphere_sf(self.x[key], D = self.D, phi = 0.0)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x[key], D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x[key], D=self.D, phi=self.phi, U=self.U, delta=0.01)
            for key in self.x.keys():
                if key == 'SAXS-term':
                    sqf[key] = self.norm * 1e-9 * 6.022e20 *sqft[key] * struct + self.sbkg # in cm^-1
                if key == 'Cross-term':
                    sqf[key] = self.norm * 1e-9 * 6.022e20 *sqft[key] * struct + self.cbkg # in cm^-1
                if key == 'Resonant-term':
                    sqf[key] = self.norm * 1e-9 * 6.022e20 *sqft[key] * struct + self.abkg # in cm^-1
            key1='Total'
            total= self.norm * 1e-9 * 6.022e20 *sqft[key1] * struct + self.sbkg
            if not self.__fit__:
                if self.sig>1e-5:
                    dL, totalL, dB, totalB, dist = self.calc_LBdist(tuple(self.__L__), tuple(self.__B__), self.sig, self.dist, self.Np)
                    self.output_params['L_Distribution'] = {'x': dL, 'y': dist}
                    self.output_params['B_Distribution'] = {'x': dB, 'y': dist}
                signal = total
                minsignal = np.min(signal)
                normsignal = signal / minsignal
                sqerr = np.random.normal(normsignal, scale=self.error_factor)
                meta = {'Energy': self.Energy}
                if self.Energy is not None:
                    self.output_params['simulated_w_err_%.4fkeV' % self.Energy] = {'x': self.x[key],
                                                                                   'y': sqerr * minsignal,
                                                                                   'yerr': np.sqrt(
                                                                                       normsignal) * minsignal * self.error_factor,
                                                                                   'meta': meta}
                else:
                    self.output_params['simulated_w_err'] = {'x': self.x[key], 'y': sqerr * minsignal,
                                                             'yerr': np.sqrt(normsignal) * minsignal}
                self.output_params['Total'] = {'x': self.x[key], 'y':total}
                for key in self.x.keys():
                    self.output_params[key] = {'x': self.x[key], 'y': sqf[key]}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1],
                                               'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1],
                                                 'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1] * scale,
                                                    'names': ['r (Angs)', 'Density (Molar)']}
                self.output_params['Structure_Factor'] = {'x': self.x[key], 'y': struct}
                xtmp,ytmp=create_steps(x=self.__L__[:-1],y=self.__Rmoles__[:-1])
                self.output_params['Rmoles_radial']={'x':xtmp,'y':ytmp}
                xtmp, ytmp = create_steps(x=self.__L__[:-1], y=self.__density__[:-1])
                self.output_params['Density_radial'] = {'x': xtmp, 'y': ytmp}
        else:
            if self.SF is None:
                struct = np.ones_like(self.x)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x, D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x, D=self.D, phi=self.phi, U=self.U, delta=0.01)

            tsqf, eisqf, asqf, csqf = self.parallelopiped(tuple(self.x), tuple(self.__L__), tuple(self.__B__), self.H, self.sig,
                                                          tuple(rho), tuple(eirho),tuple(adensity), dist=self.dist,
                                                          Np=self.Np, Nphi=self.Nphi, Npsi=self.Npsi,HggtLB=self.HggtLB)
            sqf = self.norm * 1e-9 * np.array(tsqf) * 6.022e20 * struct + self.sbkg  # in cm^-1
            # if not self.__fit__: #Generate all the quantities below while not fitting
            asqf = self.norm * 1e-9 * np.array(asqf) * 6.022e20 * struct + self.abkg  # in cm^-1
            eisqf = self.norm * 1e-9 * np.array(eisqf) * 6.022e20 * struct + self.sbkg  # in cm^-1
            csqf = self.norm * 1e-9 * np.array(csqf) * 6.022e20 * struct + self.cbkg  # in cm^-1
            # sqerr = np.sqrt(self.norm*6.022e20*self.flux * tsqf * svol*struct+self.sbkg)
            # sqwerr = (self.norm*6.022e20*tsqf * svol * struct*self.flux+self.sbkg + 2 * (0.5 - np.random.rand(len(tsqf))) * sqerr)
            # self.output_params['simulated_total_w_err'] = {'x': self.x, 'y': sqwerr, 'yerr': sqerr}
            signal = 6.022e20 * 1e-9 * self.norm * np.array(tsqf) * struct + self.sbkg
            minsignal = np.min(signal)
            normsignal = signal / minsignal
            sqerr = np.random.normal(normsignal, scale=self.error_factor)
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
            self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1],
                                           'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
            self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1],
                                             'names': ['r (Angs)', 'Electron Density (el/Angs^3)']}
            self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1] * scale,
                                                'names': ['r (Angs)', 'Density (Molar)']}  # in Molar
            self.output_params['Structure_Factor'] = {'x': self.x, 'y': struct}
            xtmp, ytmp = create_steps(x=self.__L__[:-1], y=self.__Rmoles__[:-1])
            self.output_params['Rmoles_radial'] = {'x':xtmp , 'y': ytmp}
            sqf = self.output_params[self.term]['y']
            xtmp, ytmp = create_steps(x=self.__L__[:-1], y=self.__density__[:-1])
            self.output_params['Density_radial'] = {'x': xtmp, 'y': ytmp}
            if self.sig>1e-5:
                dL, totalL, dB, totalB, dist = self.calc_LBdist(tuple(self.__L__), tuple(self.__B__), self.sig, self.dist, self.Np)
                self.output_params['L_Distribution'] = {'x': dL, 'y': dist}
                self.output_params['B_Distribution'] = {'x': dB, 'y': dist}
        return sqf



if __name__=='__main__':
    x=np.logspace(-3,0,200)
    fun=Parallelopiped_Uniform(x=x)
    print(fun.y())
