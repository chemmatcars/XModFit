import numpy as np
from lmfit import Parameters
import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines/'))
from utils import find_minmax
from functools import lru_cache

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


class ContinuousSphere:

    def __init__(self,x=0.001,Rsig=0.0,dist='Gaussian',N=50,norm=1.0,bkg=0.0,
                 mpar={'Model':{'Layers':['Layer 1','subphase'],'R':[10.0,1.0],'rho':[1.0,0.0]}}):
        """
        This calculates the form factor of a sphere with continous electron density gradient along the radial direction

        x			: single or array of q-values in the reciprocal unit as R
        mpar		: Layers: Layers, R: An array of radial locations of layers, rho: Electron density at the locations R
        Rsig		: Width of the distribution of all the radial locations
        N			: No. of points on which the distribution will be calculated
        dist		: 'Gaussian' or 'LogNormal'
        norm		: Normalization constant
        bkg		: Constant Bkg
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.Rsig=Rsig
        self.norm=norm
        self.dist=dist
        self.bkg=bkg
        self.N=N
        self.__mpar__=mpar
        self.__mkeys__ = list(self.__mpar__.keys())
        self.choices={'dist':['Gaussian','LogNormal']}
        self.filepaths = {}  # If a parameter is a filename with path
        self.output_params={'scaler_parameters':{}}
        self.init_params()

    def init_params(self):
        self.params=Parameters()
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key!='Layers':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d'%(mkey,key,i),value=self.__mpar__[mkey][key][i],vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('Rsig',value=self.Rsig,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('norm',value=self.norm,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('bkg',value=self.bkg,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)

    @lru_cache(maxsize=2)
    def calc_Rdist(self,R,Rsig,dist,N):
        totalR = np.sum(R[:-1])
        fdist = eval(dist + '.' + dist + '(x=0.001, pos=totalR, wid=self.Rsig)')
        rmin, rmax = find_minmax(fdist, totalR, Rsig)
        dr = np.linspace(rmin, rmax, N)
        fdist.x = dr
        rdist = fdist.y()
        sumdist = np.sum(rdist)
        rdist=rdist/sumdist
        self.output_params['Distribution'] = {'x': dr, 'y': rdist}
        return dr, rdist, totalR

    def update_parameters(self):
        mkey=self.__mkeys__[0]
        self.__R__ = [self.params['__%s_R_%03d' % (mkey,i)].value for i in range(len(self.__mpar__[mkey]['R']))]
        self.__rho__ = [self.params['__%s_rho_%03d' % (mkey, i)].value for i in range(len(self.__mpar__[mkey]['rho']))]

    def y(self):
        self.update_parameters()
        if self.Rsig<0.001:
            ff,aff=ff_sphere_ml(self.x,self.__R__,self.__rho__)
            return ff
            # return self.norm*self.csphere(R,rho)+self.bkg
        else:
            dr,rdist, totalR = self.calc_Rdist(tuple(self.__R__),self.Rsig,self.dist,self.N)
            res = np.zeros_like(self.x)
            for i in range(len(dr)):
                r = np.array(self.__R__) * (1 + (dr[i] - totalR) / totalR)
                ff,aff=ff_sphere_ml(self.x, r, self.__rho__)
                res = res + rdist[i] * ff
            return self.norm * res + self.bkg

if __name__=='__main__':
    x=np.logspace(-3,0,200)
    fun=ContinuousSphere(x=x)
    print(fun.y())





