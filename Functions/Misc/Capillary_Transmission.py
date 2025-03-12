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



class Capillary_Transmission: #Please put the class name same as the function name
    def __init__(self,x=0,Xh=0.0,Dh=2.0,Xch=0.0,Dc=1.0,tc=0.01,lc=0.1,ls=1,Db=0.2,norm=1.0,bkg=0.0,Npts=11,mpar={}):
        """
        Documentation
        Provides the transmission of X-ray through a capillary tube
        x           : Independent variable in the form of a scalar or an array
        Xh          : center of the hole in mm if the capillary is mounted within a hole
        Dh          : width of hole in mm if the capillary is mounted within a hole
        Xch         : Center of the capillary tube w.r.t the hole center
        Dc          : Inner diameter of the capillary tube in mm
        tc          : thickness of glass wall in mm
        lc          : absorption length of capillary tube wall in mm
        ls          : absorption length of sample inside the capillary tube
        Db          : width of the X-ray beam in mm assuming the beam profile to rectangular
        norm        : Normalization factor
        bkg         : background
        Npt         : No. of points to be used for beam profile convolution
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.Xh=Xh
        self.Dh=Dh
        self.Xch=Xch
        self.Dc=Dc
        self.tc=tc
        self.lc=lc
        self.Db=Db
        self.ls=ls
        self.norm=norm
        self.bkg=bkg
        self.Npts=Npts
        self.__mpar__=mpar #If there is any multivalued parameter
        self.choices={} #If there are choices available for any fixed parameters
        self.filepaths = {}  # If a parameter is a filename with path
        self.__fit__=False
        self.__mkeys__=list(self.__mpar__.keys())
        self.output_params={'scaler_parameters':{}}
        self.init_params()

    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params=Parameters()
        self.params.add('Dh', value=self.Dh, vary=0, min=1e-3, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('Xh', value=self.Xh, vary=0, min=1e-3, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('Dc', value=self.Dc, vary=0, min=0.0, max=np.inf, expr=None, brute_step=self.Dc/10)
        self.params.add('Xch',value=self.Xch,vary=0,min=-self.Dc/2,max=self.Dc/2,brute_step=self.Dc/10)
        self.params.add('tc',value=self.tc,vary=0,min=0.001,max=np.inf,expr=None,brute_step=self.tc/10)
        self.params.add('lc',value=self.lc,vary=0,min=1e-6,max=np.inf,expr=None,brute_step=self.lc/10)
        self.params.add('ls', value=self.ls, vary=0, min=1e-6, max=np.inf, expr=None, brute_step=self.ls/10)
        self.params.add('Db', value=self.Db, vary=0, min=1e-3, max=np.inf, expr=None, brute_step=self.Db/10)
        self.params.add('norm',value=self.norm, vary=0,min=0,max=np.inf,expr=None,brute_step=self.norm/10)
        self.params.add('bkg',value=self.bkg, vary=0,min=0,max=np.inf,expr=None,brute_step=1e-6)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key!='type':
                    for i in range(len(self.__mpar__[mkey][key])):
                            self.params.add('__%s_%s_%03d'%(mkey,key,i),value=self.__mpar__[mkey][key][i],vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)

    def update_parameters(self):
        """
        update all the multifit parameters
        """
        pass

    def hole(self,x,Dh,Xh):
        return np.heaviside(x-Xh+Dh/2.0,0.5)*(1.0-np.heaviside(x-Xh-Dh/2.0,0.5))
    def absorption(self,x,Dc,tc,Xc,ls,lc):
        """
        Xc= Xh+Xch, Absolute position of the capillary
        """
        R1 = Dc / 2
        R2 = R1 + tc
        xt=np.array(x)
        xtc = xt - Xc
        fac1 = np.where(np.abs(xtc) > R1, 1.0, np.exp(-2 * np.sqrt(R1 ** 2 - xtc ** 2) / ls) *
                        np.exp(-2 * (np.sqrt(R2 ** 2 - xtc ** 2) - np.sqrt(R1 ** 2 - xtc ** 2)) / lc))
        fac2 = np.where(np.logical_and(np.abs(xtc) > R1, np.abs(xtc) < R2), np.exp(-2 * np.sqrt(R2 ** 2 - xtc ** 2) / lc), 1.0)
        fac = fac1 * fac2
        return fac

    def beam_convovle_abs(self,x,Dc,tc,Xc,ls,lc,Db,Dh,Xh,N=11):
        """
        Xc= Xh+Xch, Absolute position of the capillary
        """
        gx=np.linspace(-Db/2,Db/2,N)
        if Db>1e-3:
            tmean=[]
            for tx in x:
                htemp = self.hole(tx-gx, Dh, Xh)
                tmean.append(np.mean(self.absorption(tx-gx,Dc,tc,Xc,ls,lc)*htemp))
            return np.array(tmean)
        else:
            htemp = self.hole(x, Dh, Xh)
            return self.absorption(tuple(x),Dc,tc,Xc,ls,lc,Dh,Xh)*htemp

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        self.update_parameters()
        fac=self.norm*self.beam_convovle_abs(self.x,self.Dc,self.tc,self.Xh+self.Xch,self.ls,self.lc,self.Db,
                                   self.Dh,self.Xh,N=self.Npts)+self.bkg
        if not self.__fit__:
            xt=np.arange(min(self.x),max(self.x),self.Db/self.Npts)
            htemp=self.hole(xt,self.Dh,self.Xh)
            cap=self.absorption(xt, self.Dc, self.tc,self.Xh+self.Xch,self.ls,self.lc)
            self.output_params['hole']={'x':xt,'y':htemp}
            self.output_params['capillary'] = {'x': xt, 'y': cap}
            self.output_params['hole_capillary']={'x': xt, 'y': cap*htemp}
        return fac


if __name__=='__main__':
    x=np.linspace(0.0,1.0,101)
    fun=Capillary_Transmission(x=x)
    print(fun.y())
