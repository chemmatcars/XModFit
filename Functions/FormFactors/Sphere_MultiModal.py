import numpy as np
from lmfit import Parameters
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
from utils import find_minmax
from PeakFunctions import Gaussian, LogNormal

from numba import jit

@jit(nopython=False)
def calc_dist(q,r,dist,sumdist):
    ffactor=np.ones_like(q)
    for i,q1 in enumerate(q):
        f = np.sum(16 * np.pi ** 2 * (np.sin(q1 * r) - q1 * r * np.cos(q1 * r)) ** 2 * dist / q1 ** 6)
        ffactor[i]=f / sumdist
    return ffactor


class Sphere_MultiModal:
    def __init__(self, x=0.001, dist='Gaussian', N=50, rhoc=1.0, rhosol=0.0, norm=1.0, bkg=0.0,
                 mpar={'Distributions':{'Dist':['Dist1'],'R':[1.0],'Rsig':[1.0],'Norm':[1.0]}}):
        """
        Calculates the form factor of a solid sphere with Multimodal size distribution
        x     	: Array of q-values in the same reciprocal unit as R and Rsig
        R     	: Mean radius of the solid spheres
        Rsig  	: Width of the distribution of solid spheres
        dist  	: Gaussian or LogNormal
        N     	: No. of points on which the distribution will be calculated
        integ   : The type of integration ('Normal' or 'MonteCarlo') Default: 'Normal'
        rhoc  	: Electron density of the particle
        rhosol	: Electron density of the solvent or surrounding environment
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.dist=dist
        self.rhoc=rhoc
        self.rhosol=rhosol
        self.norm=norm
        self.bkg=bkg
        self.N=N
        self.__mpar__=mpar
        self.__mkeys__=list(self.__mpar__.keys())
        self.choices={'dist':['Gaussian','LogNormal','Weibull']}
        self.init_params()
        self.output_params = {'scaler_parameters': {}}

    def init_params(self):
        self.params=Parameters()
        self.params.add('rhoc',value=self.rhoc,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('rhosol',value=self.rhosol,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('norm',value=self.norm,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('bkg',value=self.bkg,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key !='Dist':
                    if key == 'Rsig':
                        for i in range(len(self.__mpar__[mkey][key])):
                            self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                            min=0.001, max=np.inf, expr=None, brute_step=0.1)
                    else:
                        for i in range(len(self.__mpar__[mkey][key])):
                            self.params.add('__%s_%s_%03d'%(mkey,key,i),value=self.__mpar__[mkey][key][i],vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)

    def update_params(self):
        self.rhoc=self.params['rhoc'].value
        self.rhosol=self.params['rhosol'].value
        self.norm=self.params['norm'].value
        self.bkg=self.params['bkg'].value
        mkey = self.__mkeys__[0]
        key='R'
        self.__Nl__ = len(self.__mpar__[mkey][key])
        self.__R__ = np.array([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(self.__Nl__)])
        key='Rsig'
        self.__Rsig__ = np.array([self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(self.__Nl__)])
        key='Norm'
        self.__Norm__ = [self.params['__%s_%s_%03d' % (mkey, key, i)].value for i in range(self.__Nl__)]

    def y(self):
        self.update_params()
        rho=self.rhoc-self.rhosol
        if self.dist == 'Gaussian':
            rmin, rmax = max(0.0001, np.min(self.__R__-5*self.__Rsig__)),np.max(self.__R__+5*self.__Rsig__)
            r=np.linspace(rmin,rmax,self.N)
        elif self.dist == 'LogNormal':
            rmin, rmax = max(0.0001, np.min(np.exp(np.log(self.__R__) - 5 * self.__Rsig__))), np.max(np.exp(np.log(self.__R__) + 5 * self.__Rsig__))
            r = np.logspace(np.log10(rmin), np.log10(rmax), self.N)
        else:
            maxr=np.max(self.__R__)
            rmin,rmax= 0.0,maxr+maxr*maxr**(1.0/np.max(self.__Rsig__))
            r = np.linspace(rmin,rmax, self.N)
        dist=np.zeros_like(r)
        tdist=[]
        for i in range(self.__Nl__):
            if self.dist == 'Gaussian':
                gau=Gaussian.Gaussian(x = r, pos = self.__R__[i], wid = self.__Rsig__[i])
                gau.x=r
                tdist.append(self.__Norm__[i]*gau.y())
                dist = dist + tdist[i]
            elif self.dist == 'LogNormal':
                lgn=LogNormal.LogNormal(x = r, pos = self.__R__[i], wid = self.__Rsig__[i])
                lgn.x = r
                tdist.append(self.__Norm__[i]*lgn.y())
                dist = dist + tdist[i]
            else:
                twdist=(self.__Rsig__[i]/self.__R__[i])*(r/self.__R__[i])**(self.__Rsig__[i]-1.0)*np.exp(-(r/self.__R__[i])**self.__Rsig__[i])
                tdist.append(self.__Norm__[i]*twdist)
                dist = dist  + tdist[i]
        sumdist = np.sum(dist)
        ffactor=calc_dist(self.x,r,dist,sumdist)
        I_total=self.norm * rho ** 2 * ffactor + self.bkg
        if not self.__fit__:
            self.output_params['I_total'] = {'x': self.x,'y': I_total}
            self.output_params['Distribution'] = {'x': r, 'y': dist / sumdist}
            mean = np.sum(r * dist) / sumdist
            self.output_params['scaler_parameters']['Rmean'] = mean
            self.output_params['scaler_parameters']['Rwidth'] = np.sqrt(np.sum((r - mean) ** 2 * dist) / sumdist)
            for i in range(len(tdist)):
                self.output_params[self.__mpar__['Distributions']['Dist'][i]] = {'x': r, 'y': tdist[i] / sumdist}
                tffactor=calc_dist(self.x,r,tdist[i],sumdist)
                self.output_params['I_'+self.__mpar__['Distributions']['Dist'][i]]={'x':self.x,'y':self.norm*rho**2*tffactor+self.bkg}
        return I_total


if __name__=='__main__':
    x=np.linspace(0.001,1.0,500)
    fun=Sphere_MultiModal(x=x)
    print(fun.y())
