####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
import math
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines/'))
from functools import lru_cache
####Please do not remove lines above####

####Import your modules below if needed####
# from xr_ref import parratt_numba
from numba import njit, prange
@njit(parallel=True,cache=True)
def parratt_numba(q,lam,d,rho,beta):
    ref=np.ones_like(q)
    refc=np.ones_like(q)*complex(1.0,0.0)
    f1=16.0*np.pi*2.818e-5
    f2=-32.0*np.pi**2/lam**2
    Nl=len(d)
    for j in prange(len(q)):
        r=complex(0.0,0.0)
        for it in prange(1,Nl):
            i=Nl-it
            qc1=f1*(rho[i-1]-rho[0])
            qc2=f1*(rho[i]-rho[0])
            k1=np.sqrt(complex(q[j]**2-qc1,f2*beta[i-1]))
            k2=np.sqrt(complex(q[j]**2-qc2,f2*beta[i]))
            X=(k1-k2)/(k1+k2)
            fact1=complex(np.cos(k2.real*d[i]),np.sin(k2.real*d[i]))
            fact2=np.exp(-k2.imag*d[i])
            fact=fact1*fact2
            r=(X+r*fact)/(1.0+X*r*fact)
        ref[j]=np.abs(r)**2
        refc[j]=r
    return ref,r


class XLayers_Biphasic: #Please put the class name same as the function name
    def __init__(self,x=0.1,E=10.0,mpar={'Phase1':{'Layers':['top','bottom'],'d':[0.0,1.0],'rho':[0.0,0.333],'mu':[0.0,0.0],'sig':[0.0,3.0]},
                                         'Phase2':{'Layers':['top','bottom'],'d':[0.0,1.0],'rho':[0.0,0.333],'mu':[0.0,0.0],'sig':[0.0,3.0]}},
                 dz=0.5, rrf=True, fix_sig=False, qoff=0.0, yscale=1,cov1=0.5, bkg=0.0, coherrent=True, aveed=True):
        """
        Calculates X-ray reflectivity from a system of multiple layers using Parratt formalism

        x     	 : array of wave-vector transfer along z-direction
        E     	 : Energy of x-rays in invers units of x
        dz  :The thickness (Angstrom) of each layer for applying Parratt formalism
        rrf   	 : True for Frensnel normalized refelctivity and False for just reflectivity
        qoff  	 : q-offset to correct the zero q of the instrument
        cov1     : The coverage of Phase1 the value should be between 0 and 1
        yscale   : a scale factor for R or R/Rf
        bkg      : In-coherrent background
        coherrent: True or False for coherrent or incoherrent addition of reflectivities from different phases
        fix_sig  : True or False for constraining or not constraining all the roughness parameters to the roughness of the bare interface roughness
        mpar  	 : Dictionary of Phases where,
                   Layers: Layer description,
                   d: thickness of each layer in Angs,
                   rho:Electron density of each layer in el/Angs^3,
                   mu: Absorption length of each layer in 1/cm,
                   sig: roughness of interface separating each layer in Angs.
                   The upper and lower thickness should be always  fixed. The roughness of the topmost layer should be always kept 0.

        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.E=E
        self.__mpar__=mpar
        self.dz=dz
        self.rrf=rrf
        self.fix_sig=fix_sig
        self.qoff=qoff
        self.bkg=bkg
        self.yscale=yscale
        self.coherrent=coherrent
        self.aveed=aveed
        self.cov1=cov1
        self.choices={'rrf':[True,False],'fix_sig': [True,False],'coherrent':[True,False],'aveed':[True,False]}
        self.__d__={}
        self.__rho__={}
        self.__mu__={}
        self.__sig__={}
        self.__fit__=False
        self.__mkeys__ = list(self.__mpar__.keys())
        self.output_params = {'scaler_parameters': {}}
        self.init_params()


    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value=0,vary=0)
        """
        self.params=Parameters()
        self.params.add('qoff', self.qoff, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('yscale', self.yscale, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('cov1', self.cov1, vary=0, min=0, max=1, expr=None, brute_step=0.1)
        self.params.add('bkg', self.bkg, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        for mkey in self.__mpar__.keys():
            for key in self.__mpar__[mkey].keys():
                if key!='Layers':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey,key, i), value=self.__mpar__[mkey][key][i], vary=0, min=0, max=np.inf, expr=None, brute_step=0.05)

    @lru_cache(maxsize=10)
    def calcProfile(self,d,rho,mu,sig,phase,dz,zmin=None,zmax=None):
        """
        Calculates the electron and absorption density profiles
        """
        d = np.array(d)
        rho = np.array(rho)
        mu = np.array(mu)
        sig = np.array(sig)
        if self.fix_sig:
            for i in range(1,len(sig)):
                sig[i]=sig[1]
        n=len(d)
        if zmin is None and zmax is None:
            maxsig=max(np.abs(np.max(sig[1:])),3)
            Nlayers=int((np.sum(d[:-1])+10*maxsig)/dz)
            halfstep=(np.sum(d[:-1])+10*maxsig)/2/Nlayers
            __z__=np.linspace(-5*maxsig+halfstep,np.sum(d[:-1])+5*maxsig-halfstep,Nlayers)
            offset=0.0
        else:
            Nlayers=int((zmax-zmin)/dz)
            __z__=np.linspace(zmin,zmax,Nlayers+1)
            offset=np.sum(d[:-1])
        __d__=np.diff(__z__)
        __d__=np.append(__d__,[__d__[-1]])
        __rho__=self.sldCalFun(tuple(d),tuple(rho),tuple(sig), tuple(__z__),offset=offset)
        __mu__=self.sldCalFun(tuple(d),tuple(mu),tuple(sig), tuple(__z__),offset=offset)
        return n,__z__,__d__,__rho__,__mu__

    @lru_cache(maxsize=10)
    def sldCalFun(self,d,y,sigma,x,offset=0.0):
        wholesld=[]
        for j in range(len(x)):
            sld=0
            for i in range(len(d)-1):
                pos=np.sum(d[:i+1])
                sld=sld+math.erf((x[j]-pos+offset)/sigma[i+1]/math.sqrt(2))*(y[i+1]-y[i])
            wholesld.append(max((sld+y[0]+y[-1])/2,0))
        return np.array(wholesld)

    @lru_cache(maxsize=10)
    def stepFun(self, zmin, zmax, d, rho, mu):
        tdata = [[zmin, rho[0], mu[0]]]
        z = np.cumsum(d)
        for i, td in enumerate(d[:-1]):
            tdata.append([z[i], rho[i], mu[i]])
            tdata.append([z[i], rho[i + 1], mu[i + 1]])
        tdata.append([zmax, rho[-1], mu[-1]])
        tdata = np.array(tdata)
        return tdata[:, 0], tdata[:, 1], tdata[:, 2]

    @lru_cache(maxsize=10)
    def py_parratt(self,x,lam,d,rho,mu):
        return parratt_numba(np.array(x),lam,np.array(d),np.array(rho),np.array(mu))

    def update_parameters(self):
        for mkey in self.__mkeys__:
            # for key in self.__mpar__[mkey].keys():
            Nlayers=len(self.__mpar__[mkey]['d'])
            self.__d__[mkey]=tuple([self.params['__%s_%s_%03d'%(mkey,'d',i)].value for i in range(Nlayers)])
            self.__rho__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'rho', i)].value for i in range(Nlayers)])
            self.__mu__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'mu', i)].value for i in range(Nlayers)])
            self.__sig__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'sig', i)].value for i in range(Nlayers)])

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        x = self.x + self.qoff
        lam = 6.62607004e-34 * 2.99792458e8 * 1e10 / self.E / 1e3 / 1.60217662e-19
        if not self.__fit__:
            for mkey in self.__mpar__.keys():
                Nlayers = len(self.__mpar__[mkey]['sig'])
                for i in range(2,Nlayers):
                    if self.fix_sig:
                        self.params['__%s_%s_%03d' % (mkey, 'sig', i)].expr = '__%s_%s_%03d' % (mkey, 'sig', 1)
                    else:
                        self.params['__%s_%s_%03d' % (mkey, 'sig', i)].expr = None
        self.update_parameters()
        refq={}
        r2={}
        n={}
        z={}
        d={}
        rho={}
        mu={}
        mkeys = self.__mkeys__
        if not self.coherrent:
            for mkey in mkeys:
                n[mkey],z[mkey],d[mkey],rho[mkey],mu[mkey]=self.calcProfile(self.__d__[mkey],self.__rho__[mkey],
                                                                              self.__mu__[mkey],self.__sig__[mkey],mkey,
                                                                              self.dz)
                refq[mkey],r2[mkey]=self.py_parratt(tuple(x),lam,tuple(d[mkey]),tuple(rho[mkey]),tuple(mu[mkey]))
                if not self.__fit__:
                    self.output_params['%s_EDP' % mkey] = {'x': z[mkey]-np.sum(self.__d__[mkey][:-1]), 'y': rho[mkey]}
                    self.output_params['%s_dEDP' % mkey] = {'x': z[mkey][:-1] - np.sum(self.__d__[mkey][:-1]), 'y': np.diff(rho[mkey])}
                    self.output_params['%s_ADP' % mkey] = {'x': z[mkey]-np.sum(self.__d__[mkey][:-1]), 'y': mu[mkey]}
                    self.output_params['%s_dADP' % mkey] = {'x': z[mkey][:-1] - np.sum(self.__d__[mkey][:-1]), 'y': np.diff(mu[mkey])}
            trefq=self.cov1*refq[mkeys[0]]+(1.0-self.cov1)*refq[mkeys[1]]
            if self.rrf:
                rhos=(self.params['__%s_rho_000'%mkeys[0]].value,self.params['__%s_rho_%03d'%(mkeys[0],n[mkeys[0]]-1)].value)
                mus=(self.params['__%s_mu_000'%mkeys[0]].value,self.params['__%s_mu_%03d'%(mkeys[0],n[mkeys[0]]-1)].value)
                ref,r2=self.py_parratt(tuple(x-self.qoff),lam,(0.0,1.0),rhos,mus)
                trefq=trefq/ref
            return trefq * self.yscale + self.bkg
        else:
            if not self.aveed:
                for mkey in mkeys:
                    n[mkey], z[mkey], d[mkey], rho[mkey], mu[mkey] = self.calcProfile(self.__d__[mkey],
                                                                                        self.__rho__[mkey],
                                                                                        self.__mu__[mkey],
                                                                                        self.__sig__[mkey], mkey,
                                                                                        self.dz)
                    refq[mkey], r2[mkey] = self.py_parratt(tuple(x), lam, tuple(d[mkey]), tuple(rho[mkey]),
                                                           tuple(mu[mkey]))
                    if not self.__fit__:
                        tz, trho, tmu = self.stepFun(z[mkey][0], z[mkey][-1], self.__d__[mkey], self.__rho__[mkey],
                                                     self.__mu__[mkey])
                        self.output_params['%s_EDP' % mkey] = {'x': z[mkey] - np.sum(self.__d__[mkey][:-1]),
                                                               'y': rho[mkey], 'names':['z (Angs)','rho (el/Angs^3'],'plotType':'step'}
                        self.output_params['%s_dEDP' % mkey] = {'x': z[mkey][:-1] - np.sum(self.__d__[mkey][:-1]),
                                                               'y': np.diff(rho[mkey]), 'names': ['z (Angs)', 'rho (el/Angs^4'],
                                                               'plotType': 'step'}
                        self.output_params['%s_step_EDP' % mkey] = {'x': tz, 'y': trho,
                                                                                 'names': ['z (Angs)',
                                                                                           'rho (el/Angs^3)'],
                                                                                 'plotType': 'step'}
                        self.output_params['%s_ADP' % mkey] = {'x': z[mkey] - np.sum(self.__d__[mkey][:-1]),
                                                               'y': mu[mkey], 'names':['z (Angs)','mu (1/cm)'],'plotType':'step'}
                        self.output_params['%s_dADP' % mkey] = {'x': z[mkey][:-1] - np.sum(self.__d__[mkey][:-1]),
                                                               'y': np.diff(mu[mkey]), 'names': ['z (Angs)', 'mu (1/cm^2)'],
                                                               'plotType': 'step'}
                        self.output_params['%s_step_ADP' % mkey] = {'x': tz, 'y': tmu,
                                                                    'names': ['z (Angs)',
                                                                              'mu (1/cm)'],
                                                                    'plotType': 'step'}
                deld=np.sum(self.__d__[mkeys[0]][:-1])-np.sum(self.__d__[mkeys[1]][:-1])
                refq = abs(np.exp(-1j*self.x*deld)*self.cov1 * r2[mkeys[0]] + (1.0 - self.cov1) * r2[mkeys[1]])**2

            else:
                maxsig=max(max(self.__sig__[mkeys[0]][1:]),max(self.__sig__[mkeys[1]][1:]))
                zmin=-max(np.sum(self.__d__[mkeys[0]][:-1]),np.sum(self.__d__[mkeys[1]][:-1]))-5*maxsig
                zmax=5*maxsig
                Nlayers = int((zmax - zmin)/self.dz)
                for mkey in mkeys:
                    n[mkey], z[mkey], d[mkey], rho[mkey], mu[mkey] = self.calcProfile(self.__d__[mkey],
                                                                                        self.__rho__[mkey],
                                                                                        self.__mu__[mkey],
                                                                                        self.__sig__[mkey], mkey,
                                                                                        self.dz,zmin=zmin,zmax=zmax)
                    if not self.__fit__:
                        tz, trho, tmu = self.stepFun(z[mkey][0], z[mkey][-1], self.__d__[mkey], self.__rho__[mkey],
                                                     self.__mu__[mkey])
                        self.output_params['%s_EDP' % mkey] = {'x': z[mkey], 'y': rho[mkey], 'names':['z (Angs)','rho (el/Angs^3)'],'plotType':'step'}
                        self.output_params['%s_dEDP' % mkey] = {'x': z[mkey][:-1], 'y': np.diff(rho[mkey]) / self.dz,
                                                                             'names': ['z (Angs)',
                                                                                       'drho/dz (el/Angs^4)'],
                                                                             'plotType': 'step'}
                        self.output_params['%s_step_EDP' % mkey] = {'x': tz, 'y': trho,
                                                                    'names': ['z (Angs)',
                                                                              'rho (el/Angs^3)'],
                                                                    'plotType': 'step'}
                        self.output_params['%s_ADP' % mkey] = {'x': z[mkey], 'y': mu[mkey], 'names':['z (Angs)','mu (1/cm)'],'plotType':'step'}
                        self.output_params['%s_dADP' % mkey] = {'x': z[mkey][:-1], 'y': np.diff(mu[mkey]),
                                                               'names': ['z (Angs)', 'dmu/dz (1/cm^2)'], 'plotType': 'step'}
                        self.output_params['%s_step_ADP' % mkey] = {'x': tz, 'y': tmu,
                                                                    'names': ['z (Angs)',
                                                                              'mu (1/cm)'],
                                                                    'plotType': 'step'}
                trho=self.cov1*rho[mkeys[0]]+(1-self.cov1)*rho[mkeys[1]]
                tmu = self.cov1*mu[mkeys[0]]+(1-self.cov1)*mu[mkeys[1]]
                if not self.__fit__:
                    self.output_params['Total_EDP'] = {'x': z[mkey], 'y': trho, 'names':['z (Angs)','rho (el/Angs^3'],'plotType':'step'}
                    self.output_params['Total_dEDP'] = {'x': z[mkey], 'y': np.diff(trho), 'names': ['z (Angs)', 'rho (el/Angs^4'],
                                                       'plotType': 'step'}
                    self.output_params['Total_ADP'] = {'x': z[mkey], 'y': tmu, 'names':['z (Angs)','mu (1/cm)'],'plotType':'step'}
                    self.output_params['Total_dADP'] = {'x': z[mkey][:-1], 'y': np.diff(tmu), 'names': ['z (Angs)', 'dmu/dz (1/cm^2)'],
                                                       'plotType': 'step'}
                refq, r2 = self.py_parratt(tuple(x), lam, tuple(d[mkey]), tuple(trho), tuple(tmu))
            if self.rrf:
                rhos=(self.params['__%s_rho_000'%mkeys[0]].value,self.params['__%s_rho_%03d'%(mkeys[0],n[mkeys[0]]-1)].value)
                mus=(self.params['__%s_mu_000'%mkeys[0]].value,self.params['__%s_mu_%03d'%(mkeys[0],n[mkeys[0]]-1)].value)
                ref,r2=self.py_parratt(tuple(x-self.qoff),lam,(0.0,1.0),rhos,mus)
                refq=refq/ref
            return refq * self.yscale + self.bkg


if __name__=='__main__':
    x=np.linspace(0.001,1.0,100)
    fun=XLayers_Biphasic(x=x)
    print(fun.y())
