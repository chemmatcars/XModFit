####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_rountines'))
from functools import lru_cache
####Please do not remove lines above####

####Import your modules below if needed####
import copy
from xraydb import XrayDB
from itertools import combinations
import os
import mendeleev


class ForMol_Uniform: #Please put the class name same as the function name
    re=2.818e-5 #Classical electron radius in Angstroms
    No=6.023e23 #Avagadro's number
    def __init__(self,x=0,Energy=12.0,fname1='./Data/Pr4.xyz',eta1=1.0,fname2='None',eta2=0.0,sol=0.0,sig=0.0,
                 scale=1.0, norm=1, norm_err=0.01, bkg=0.0,mpar={},error_factor=1.0):
        """
        Calculates the form factor for two different kinds of  molecules in cm^-1 for which the XYZ coordinates of the all the atoms composing the molecules are known
        Reference: https://www.sciencedirect.com/science/article/pii/S2001037014600180?via%3Dihub
        x    	    : Scalar or array of reciprocal wave vectors
        Energy    	: Energy of the X-rays at which the scattering pattern is measured
        fname1	    : Name with path of the .xyz file containing X, Y, Z coordinates of all the atoms of the molecule of type 1
        eta1 	    : Fraction of molecule type 1
        fname2	    : Name with path of the .xyz file containing X, Y, Z coordinates of all the atoms of the moleucule of type 2
        eta2 	    : Fraction of molecule type 2
        sol         : Electron density of solvent in el/Angs^3
        sig  	    : Debye-waller factor
        scale       : scale factor for uniform scaling of the co-ordinates
        norm 	    : Normalization constant which can be the molar concentration of the particles
        norm_err    : Percentage of error on normalization to simulated energy dependent SAXS data
        bkg 	    : Background
        error_factor: Error-factor to simulate the error-bars
        """
        if type(x)==list:
            self.x = np.array(x)
        else:
            self.x = np.array([x])
        if os.path.exists(fname1):
            self.fname1 = fname1
        else:
            self.fname1 = None
        self.eta1 = eta1
        if os.path.exists(fname2):
            self.fname2 = fname2
        else:
            self.fname2 = None
        self.eta2 = eta2
        self.norm = norm
        self.norm_err = norm_err
        self.error_factor = error_factor
        self.bkg = bkg
        self.Energy = Energy
        self.sol = sol
        self.sig = sig
        self.scale = scale
        self.__mpar__ = mpar #If there is any multivalued parameter
        self.choices = {} #If there are choices available for any fixed parameters
        self.__fnames__ = [self.fname1,self.fname2]
        self.__xdb__ = XrayDB()
        #if self.fname1 is not None:
        #    self.__Natoms1__,self.__pos1__,self.__f11__=self.readXYZ(self.fname1)
        #if self.fname2 is not None:
        #    self.__Natoms2__,self.__pos2__,self.__f12__=self.readXYZ(self.fname2)
        self.output_params={'scaler_parameters':{}}


    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value=0,vary=0)
        """
        self.params=Parameters()
        self.params.add('eta1',value=self.eta1,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('eta2',value=self.eta2,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('norm',value=self.norm,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('bkg',value=self.bkg,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('sig',value=self.sig,vary=0,min=-np.inf,max=np.inf,expr=None,brute_step=0.1)
        self.params.add('scale', value=self.scale, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)

    @lru_cache(maxsize=10)
    def readXYZ(self,q,fname,energy=None):
        """
        Reads the xyz file to read the atomic positions and put it in dictionary
        """
        q=np.array(q)
        fh=open(fname,'r')
        lines=fh.readlines()
        Natoms=eval(lines[0])
        atoms={'Natoms':Natoms}
        # cen=np.zeros(3)
        atoms['elements']=[]
        atoms['ff']=[]
        atoms['vol']=[]
        for i in range(Natoms):
            line=lines[i+2].split()
            pos=np.array(list((map(eval,line[1:]))))
            element = line[0]
            if element not in atoms['elements']:
                atoms['elements'].append(element)
                f0 = self.__xdb__.f0(element, q)
                if energy is not None:
                    f1 = self.__xdb__.f1_chantler(element = element, energy=energy * 1e3, smoothing = 0)
                    f2 = self.__xdb__.f2_chantler(element = element, energy=energy * 1e3, smoothing = 0)
                    ff = f0 + f1 + 1.0j * f2
                else:
                    ff = f0 * (1.0 + 0j)
                atoms['ff'].append(ff)
                vol = 4 * 3.14159 * eval('mendeleev.%s.vdw_radius' % element)**3 / 3.0 / 1.0e6
                atoms['vol'].append(vol)
            atoms[i] = {'element': element, 'pos': pos}
        return atoms

    @lru_cache(maxsize=10)
    def get_pairs(self, q, fname, energy=None):
        atoms=self.readXYZ(q, fname, energy = energy)
        pairs=[]
        distances=[]
        for i in range(atoms['Natoms']):
            for j in range(atoms['Natoms']):
                pairs.append([atoms['elements'].index(atoms[i]['element']), atoms['elements'].index(atoms[j]['element'])])
                distances.append(np.sqrt(np.sum((atoms[i]['pos']-atoms[j]['pos'])**2)))
        return atoms['ff'], pairs, np.array(distances), atoms

    def calc_formol(self, q, atom_ff, atom_pairs, atom_dist, sol, atoms):
        """
        Calculates the form factor of a molecule with know atomic positions
        :param q: reciprocal lattice vector in inv-Angs
        :param atom_ff: Array of atomic form factors of all the elements where rows represent each of the elements and cols
                        represents different form factor at different q values
        :param atom_pairs: array of pairs
        :param atom_dist: list of distance between the pairs
        :param sol: solvent density in el/Angs^3
        :param atomic_volume: dictionary of atomic volumes of all the elements present in the system
        :return:
        """
        form = np.zeros_like(q)
        for k, atom in enumerate(atom_pairs):
            i, j = atom
            sinc = np.sinc(atom_dist[k]*q/np.pi)
            form = form + np.real((atom_ff[i]-sol*atoms['vol'][i])*(atom_ff[j]-sol*atoms['vol'][j]).conjugate()) * sinc
        return form

    @lru_cache(maxsize=10)
    def calc_form(self,q,fname, sol, scale, energy=None):
        atom_ff, atom_pairs, atom_dist, atoms = self.get_pairs(q, fname, energy=energy)
        return self.calc_formol(np.array(q),atom_ff,atom_pairs, scale*atom_dist, sol, atoms)

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        #Contribution from first molecule
        if self.fname1 is not None:
            form1 = self.re**2*1e-16*self.No*self.calc_form(tuple(self.x),self.fname1, self.sol, self.scale, energy=self.Energy)
        #Contribution from second molecule
        if self.fname2 is not None:
            form2 = self.re**2*1e-16*self.No*self.calc_form(tuple(self.x),self.fname2, self.sol, self.scale, energy=self.Energy)

        self.__fnames__=[self.fname1, self.fname2]

        if self.__fnames__[0] is not None and self.__fnames__[1] is not None:
            self.output_params[os.path.basename(self.fname1)+'_1']={'x':self.x,'y':self.norm*form1*np.exp(-self.x**2*self.sig**2),
                                                                    'names':['q','Intensity']}
            self.output_params[os.path.basename(self.fname2)+'_1']={'x':self.x,'y':self.norm*form2*np.exp(-self.x**2*self.sig**2),
                                                                    'names':['q','Intensity']}
            self.output_params['bkg']={'x':self.x,'y':self.bkg*np.ones_like(self.x)}
            total= (self.eta1*form1+self.eta2*form2)*self.norm*np.exp(-self.x**2*self.sig**2)*1e-3+self.bkg
        elif self.__fnames__[0] is not None and self.__fnames__[1] is None:
            self.output_params[os.path.basename(self.fname1)+'_1']={'x':self.x,'y':self.norm*self.eta1*form1*np.exp(-self.x**2*self.sig**2),
                                                                    'names':['q','Intensity']}
            self.output_params['bkg']={'x':self.x,'y':self.bkg*np.ones_like(self.x)}
            total= self.eta1*form1*self.norm*np.exp(-self.x**2*self.sig**2)*1e-3+self.bkg
        elif self.__fnames__[0] is None and self.__fnames__[1] is not None:
            self.output_params[os.path.basename(self.fname2)+'_1']={'x':self.x,'y':self.norm*self.eta2*form2*np.exp(-self.x**2*self.sig**2),
                                                                    'names':['q','Intensity']}
            self.output_params['bkg']={'x':self.x,'y':self.bkg*np.ones_like(self.x)}
            total= self.eta2*form2*self.norm*np.exp(-self.x**2*self.sig**2)*1e-3+self.bkg
        else:
            total= np.ones_like(self.x)
        signal = total[0:]
        minsignal = np.min(signal)
        normsignal = signal / minsignal
        norm = np.random.normal(self.norm, scale=self.norm_err / 100.0)
        sqerr = np.random.normal(normsignal * norm, scale=self.error_factor)
        meta = {'Energy': self.Energy}
        tkeys = list(self.output_params.keys())
        for key in tkeys:
            if 'simulated_w_err' in key:
                del self.output_params[key]
        if self.Energy is not None:
            self.output_params['simulated_w_err_%.3fkeV' % self.Energy] = {'x': self.x, 'y': sqerr * minsignal,
                                                                           'yerr': np.sqrt(normsignal) * minsignal * self.error_factor,
                                                                           'meta': meta}
        else:
            self.output_params['simulated_w_err'] = {'x': self.x, 'y': sqerr * minsignal,
                                                     'yerr': np.sqrt(normsignal) * minsignal * self.error_factor, 'meta': meta}
        return signal


if __name__=='__main__':
    x=np.logspace(-2,0.7,50)
    fun=ForMol_Uniform(x=x)
    fun.fname1='/media/sf_Mrinal_Bera/Documents/MA-Collab/XTal_data/P8W48.xyz'
    print(fun.y())
