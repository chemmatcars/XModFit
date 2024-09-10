####Please do not remove lines below####
import math
import os
import sys

import numpy as np
from lmfit import Parameters

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines/'))
from functools import lru_cache
"""from timeit import default_timer
from scipy import special"""

####Please do not remove lines above####

####Import your modules below if needed####
# from xr_ref import parratt
import Bio.PDB.Atom
from scipy.spatial.transform import Rotation as R
from mendeleev.fetch import fetch_table
from numba import njit


@njit(parallel=False,cache=True)
def parratt_numba(q,lam,d,rho,beta):
    ref=np.ones_like(q)
    refc=np.ones_like(q)*complex(1.0,0.0)
    f1=16.0*np.pi*2.818e-5
    f2=-32.0*np.pi**2/lam**2
    Nl=len(d)
    for j in range(len(q)):
        r=complex(0.0,0.0)
        for it in range(1,Nl):
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
    return ref, r

@njit(parallel=False, cache=True)
def protmembcomb(protmembpos, membpos, d, rho, mu, sig, protpos, protrho, layarea, cov):
    m = len(protmembpos)
    protmembrho = np.zeros(m)
    protmembsig = np.zeros(m)
    protmembmu = np.zeros(m)
    """Membrane position 0th index is 0 because the first layer is air and has d[0]=0 thickness, so membrane
           electron density needs to start at rho[1].
           Protein position 0th index is a real thickness, so protein electron density needs to start at protrho[0].
           The position comparison should be < for protein and <= for membrane."""
    k = 0
    l = 0
    for j in range(m - 1):
        if protmembpos[j] <= membpos[k] and k < (len(d) - 1):
            k += 1
        if protmembpos[j] < protpos[l] and l < (len(protpos) - 1):
            l += 1

        protmembsig[j] = sig[k]
        protmembmu[j] = mu[k]
        if protmembpos[j] > protpos[0] or protmembpos[j] < protpos[-1]:
            protmembrho[j] = rho[k]
        else:
            protmembrho[j] = (cov * protrho[l] + (1 + cov * (layarea[l] - 1)) * rho[k])

    protmembsig[-1] = sig[k]
    protmembmu[-1] = mu[k]
    protmembrho[-1] = rho[k]

    return protmembrho, protmembmu, protmembsig

@njit(parallel=False, cache=True)
def sldCalFun_numba(pos, y, sigma, x):
    wholesld = []
    for j in range(len(x)):
        sld = 0
        for i in range(len(pos) - 1):
            sld += math.erf((x[j] - pos[i]) / sigma[i + 1] / math.sqrt(2)) * (y[i + 1] - y[i])
        wholesld.append(max((sld + y[0] + y[-1]) / 2, 0))
    return wholesld

@njit(parallel=False, cache=True)
def SphereElecFrac(zrotcoord, radius, electrons, z_gridpos, zslice, ElecFracsum, Volfrac):
    for ind in range(len(radius)):
        vol = (4 / 3) * np.pi * (radius[ind] ** 3)

        botslice = z_gridpos - zrotcoord[ind] - 0.5 * zslice
        topslice = botslice + zslice

        zbotind = int(np.ceil((-radius[ind] - topslice[0]) / zslice))
        ztopind = int(np.floor((radius[ind] - botslice[0]) / zslice))

        inc = np.arange(zbotind, (ztopind + 1))

        d = np.maximum(botslice[inc], -radius[ind])
        h = np.minimum((topslice[inc] - d), (radius[ind] - botslice[inc]))

        ElecFracsum[inc] += electrons[ind] * (
                    np.pi * h * (np.square(radius[ind]) - np.square(d) - h * d - (1 / 3) * np.square(h))) / vol

        Volfrac[inc] += (np.pi * h * (np.square(radius[ind]) - np.square(d) - h * d - (1 / 3) * np.square(h)))

    return ElecFracsum, Volfrac

@njit(parallel=False, cache=True)
def Vol_gridcalc(xrotcoord, yrotcoord, zrotcoord, radius, slices, x_gridpos, y_gridpos, z_gridpos, edges, Vol_grid):
    for atomind in range(len(radius)):

        xtopind = int(np.ceil((xrotcoord[atomind] + radius[atomind] - edges[1] - 0.5 * slices[0]) / slices[0]))
        xbotind = int(np.floor((xrotcoord[atomind] - radius[atomind] - edges[1] - 0.5 * slices[0]) / slices[0]))
        ytopind = int(np.ceil((yrotcoord[atomind] + radius[atomind] - edges[3] - 0.5 * slices[1]) / slices[1]))
        ybotind = int(np.floor((yrotcoord[atomind] - radius[atomind] - edges[3] - 0.5 * slices[1]) / slices[1]))
        ztopind = int(np.ceil((zrotcoord[atomind] + radius[atomind] - edges[5] - 0.5 * slices[2]) / slices[2]))
        zbotind = int(np.floor((zrotcoord[atomind] - radius[atomind] - edges[5] - 0.5 * slices[2]) / slices[2]))

        radius2 = radius[atomind] ** 2

        for xind in range(xbotind, xtopind + 1):
            for yind in range(ybotind, ytopind + 1):
                for zind in range(zbotind, ztopind + 1):
                    atomdist = (x_gridpos[xind] - xrotcoord[atomind]) ** 2 + (
                                y_gridpos[yind] - yrotcoord[atomind]) ** 2 + (
                                           z_gridpos[zind] - zrotcoord[atomind]) ** 2
                    if atomdist <= radius2:
                        Vol_grid[xind, yind, zind] = 1

    return Vol_grid

@njit(parallel=False, cache=True)
def Area_gridcalc(xrotcoord, yrotcoord, radius, slices, x_gridpos, y_gridpos, edges, Area_grid):
    for ind in range(len(radius)):

        xtopind = int(np.ceil((xrotcoord[ind] + radius[ind] - edges[1] - 0.5 * slices[0]) / slices[0]))
        xbotind = int(np.floor((xrotcoord[ind] - radius[ind] - edges[1] - 0.5 * slices[0]) / slices[0]))
        ytopind = int(np.ceil((yrotcoord[ind] + radius[ind] - edges[3] - 0.5 * slices[1]) / slices[1]))
        ybotind = int(np.floor((yrotcoord[ind] - radius[ind] - edges[3] - 0.5 * slices[1]) / slices[1]))

        radius2 = radius[ind] ** 2

        for xind in range(xbotind, xtopind + 1):
            for yind in range(ybotind, ytopind + 1):
                atomdist = (x_gridpos[xind] - xrotcoord[ind]) ** 2 + (y_gridpos[yind] - yrotcoord[ind]) ** 2
                if atomdist <= radius2:
                    Area_grid[xind, yind] = 1

    return Area_grid

class XLayersPDB: #Please put the class name same as the function name
    def __init__(self, x=0.1, E=10.0, fname='./Data/tim3_hmmm_ext_all_bestframe.pdb', mpar={
        'Model': {'Layers': ['top', 'bottom'], 'd': [0.0, 1.0], 'rho': [0.0, 0.333], 'mu': [0.0, 0.0],
                   'sig': [0.0, 3.0]}},
                 dz=0.5, fslice=0.5, rrf=True, fix_sig=True, qoff=0.0, yscale=1, bkg=0.0,
                 theta=0.0, phi=0.0, protins=0.0, cov=0.1):
        """
        Calculates X-ray reflectivity from a system of multiple layers using Parratt formalism

        x      	 : array of wave-vector transfer along z-direction
        E      	 : Energy of x-rays in inverse units of x
        dz   	 : The thickness (Angstrom) of each layer for applying Parratt formalism
        fslice   : The thickness (Angstrom) of each layer of protein electron density
        rrf    	 : True for Frensnel normalized reflectivity and False for just reflectivity
        qoff   	 : q-offset to correct the zero q of the instrument
        yscale   : a scale factor for R or R/Rf
        bkg      : Incoherent background
        theta/phi: Euler angles of protein
        protins  : insertion of ED profile in Angs,
        cov      : coverage of ED profile in superposition with layers, number between 0 and 1.
        mpar  	 : Dictionary of Phases where,
                   Layers: Layer description,
                   d: thickness of each layer in Angs,
                   rho:Electron density of each layer in el/Angs^3,
                   mu: Absorption coefficient of each layer in 1/cm,
                   sig: roughness of interface separating each layer in Angs,
                   The upper and lower thickness should be always  fixed. The roughness of the topmost layer should be always kept 0.
        """
        if type(x) == list:
            self.x = np.array(x)
        else:
            self.x = x
        if os.path.exists(fname):
            self.fname=fname
        else:
            self.fname=None
        self.E = E
        self.__mpar__ = mpar
        self.dz = dz
        self.fslice = fslice
        self.rrf = rrf
        self.fix_sig = fix_sig
        self.qoff = qoff
        self.bkg = bkg
        self.yscale = yscale
        self.theta = theta
        self.phi = phi
        self.protins = protins
        self.cov = cov
        self.choices = {'rrf': [True, False], 'fix_sig': [True, False]}
        self.__d__ = {}
        self.__rho__ = {}
        self.__mu__ = {}
        self.__sig__ = {}
        self.__fit__ = False
        self.__mkeys__ = list(self.__mpar__.keys())
        self.__fix_sig_changed__ = 0
        self.init_params()
        self.output_params = {'scaler_parameters': {}}
        xcoord, ycoord, zcoord, electrons, radius, mass = self.pdbread(self.fname)
        self.__protvol__ = self.protvolcalc(tuple(xcoord), tuple(ycoord), tuple(zcoord), tuple(radius), fslice)
        self.__protpos__, self.__protrho__, self.__layarea__ = self.ED_profrot(tuple(xcoord), tuple(ycoord),
                                                                   tuple(zcoord), tuple(radius),
                                                                   tuple(electrons), self.theta, self.phi,
                                                                   self.fslice, self.__protvol__)

    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value=0,vary=0)
        """
        self.params = Parameters()
        self.params.add('qoff', self.qoff, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('yscale', self.yscale, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('bkg', self.bkg, vary=0, min=0, max=1, expr=None, brute_step=0.1)
        self.params.add('theta', self.theta, vary=1, min=-np.inf, max=np.inf, expr=None, brute_step=1)
        self.params.add('phi', self.phi, vary=1, min=-np.inf, max=np.inf, expr=None, brute_step=1)
        self.params.add('protins', self.protins, vary=1, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('cov', self.cov, vary=1, min=0, max=1, expr=None, brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey].keys():
                if key != 'Layers':
                    for i in range(len(self.__mpar__[mkey][key])):
                        # if key!='sig':
                        #     self.params.add('__%s_%s_%03d'%(mkey,key,i),value=self.__mpar__[mkey][key][i],vary=0,min=1e-10,max=np.inf,expr=None,brute_step=0.1)
                        # else:
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0,
                                        min=0, max=np.inf, expr=None, brute_step=0.05)

    @lru_cache(maxsize=2)
    def pdbread(self, fname):

        ptable = fetch_table('elements')[['symbol', 'atomic_number', 'vdw_radius', 'atomic_weight']]
        atominfo = ptable.set_index('symbol').T.to_dict('list')
        pdbcode = 'struct'

        pdbparser = Bio.PDB.PDBParser(QUIET=True)
        struct = pdbparser.get_structure(pdbcode, fname)

        radius = np.zeros(len(list(struct.get_atoms())))
        electrons = np.zeros(len(list(struct.get_atoms())))
        mass = np.zeros(len(list(struct.get_atoms())))
        coord = np.zeros((len(list(struct.get_atoms())), 3))

        ind = 0

        for atom in struct.get_atoms():
            coord[ind, :] = atom.coord
            if atom.name == 'CAL':
                electrons[ind] = atominfo['Ca'][0]
                radius[ind] = atominfo['Ca'][1] / 100
                mass[ind] = atominfo['Ca'][2]
            else:
                electrons[ind] = atominfo[atom.element][0]
                radius[ind] = atominfo[atom.element][1] / 100
                mass[ind] = atominfo[atom.element][2]
            ind += 1

        xcoord = np.squeeze(coord[:, 0])
        ycoord = np.squeeze(coord[:, 1])
        zcoord = np.squeeze(coord[:, 2])

        return xcoord, ycoord, zcoord, electrons, radius, mass

    @lru_cache(maxsize=3)
    def protvolcalc(self, xcoord, ycoord, zcoord, radius, fslice):
        xcoord = np.array(xcoord)
        ycoord = np.array(ycoord)
        zcoord = np.array(zcoord)
        radius = np.array(radius)

        xtop = np.max(xcoord + radius) + fslice
        xbot = np.min(xcoord - radius) - fslice
        ytop = np.max(ycoord + radius) + fslice
        ybot = np.min(ycoord - radius) - fslice
        ztop = np.max(zcoord + radius) + fslice
        zbot = np.min(zcoord - radius) - fslice

        edges = np.array([xtop, xbot, ytop, ybot, ztop, zbot])

        """Calculation of grid for volume contribution"""

        xgridnum = np.round((xtop - xbot) / fslice)
        ygridnum = np.round((ytop - ybot) / fslice)
        zgridnum = np.round((ztop - zbot) / fslice)

        xslice = (xtop - xbot) / xgridnum
        yslice = (ytop - ybot) / ygridnum
        zslice = (ztop - zbot) / zgridnum
        volslice = xslice * yslice * zslice

        slices = np.array([xslice, yslice, zslice, volslice])

        x_gridpos = xbot + xslice * (np.arange(1, xgridnum + 1) - 0.5)
        y_gridpos = ybot + yslice * (np.arange(1, ygridnum + 1) - 0.5)
        z_gridpos = zbot + zslice * (np.arange(1, zgridnum + 1) - 0.5)

        Vol_grid = np.zeros([int(xgridnum), int(ygridnum), int(zgridnum)], dtype=bool)
        Vol_grid = Vol_gridcalc(xcoord, ycoord, zcoord, radius, slices, x_gridpos, y_gridpos, z_gridpos, edges, Vol_grid)
        protvol = volslice * np.sum(np.count_nonzero(Vol_grid))

        return protvol

    @lru_cache(maxsize=2)
    def ED_profrot(self, xcoord, ycoord, zcoord, radius, electrons, theta, phi, fslice, protvol):
        """Move protein to center of mass"""

        coord = np.transpose(np.array([xcoord, ycoord, zcoord]))
        radius = np.array(radius)
        electrons = np.array(electrons)
        """CoM = np.sum(mass[:, np.newaxis]*coord, axis=0)/np.sum(mass)

        coordCoM = coord - CoM"""

        """Rotate coordinates"""

        rot = R.from_euler('zx', [[phi, theta]], degrees=True)

        rotcoord = rot.apply(coord)
        """Replace coord with coordCoM when done or make it an option"""

        """Calculation of edges of box that enclose protein structure"""

        xtop = np.max(rotcoord[:, 0] + radius) + fslice
        xbot = np.min(rotcoord[:, 0] - radius) - fslice
        ytop = np.max(rotcoord[:, 1] + radius) + fslice
        ybot = np.min(rotcoord[:, 1] - radius) - fslice
        ztop = np.max(rotcoord[:, 2] + radius) + fslice
        zbot = np.min(rotcoord[:, 2] - radius) - fslice

        edges = np.array([xtop, xbot, ytop, ybot, ztop, zbot])

        """Calculation of grid for volume contribution"""

        xgridnum = np.round((xtop - xbot) / fslice)
        ygridnum = np.round((ytop - ybot) / fslice)
        zgridnum = np.round((ztop - zbot) / fslice)

        xslice = (xtop - xbot) / xgridnum
        yslice = (ytop - ybot) / ygridnum
        zslice = (ztop - zbot) / zgridnum
        volslice = xslice * yslice * zslice

        slices = np.array([xslice, yslice, zslice, volslice])

        x_gridpos = xbot + xslice * (np.arange(1, xgridnum + 1) - 0.5)
        y_gridpos = ybot + yslice * (np.arange(1, ygridnum + 1) - 0.5)
        z_gridpos = zbot + zslice * (np.arange(1, zgridnum + 1) - 0.5)

        ElecFracsum = np.zeros(len(z_gridpos))
        Volfrac = np.zeros(len(z_gridpos))
        [SphereElec, Volfrac] = SphereElecFrac(rotcoord[:, 2], radius, electrons, z_gridpos, zslice, ElecFracsum,
                                               Volfrac)

        Area_grid = np.zeros([int(xgridnum), int(ygridnum)], dtype=bool)
        Area_grid = Area_gridcalc(rotcoord[:, 0], rotcoord[:, 1], radius, slices, x_gridpos, y_gridpos, edges, Area_grid)
        minareagrid = np.sum(np.count_nonzero(Area_grid))

        protpos = np.flipud(z_gridpos)
        protrho = np.flipud(SphereElec) / (volslice * minareagrid)
        layarea = (minareagrid * volslice - np.flipud(Volfrac) * (protvol / np.sum(Volfrac))) / (
                    minareagrid * volslice)

        return protpos, protrho, layarea

    @lru_cache(maxsize=2)
    def calcProfComb(self, d, rho, mu, sig, protins, cov, phase, dz, protpos, protrho, layarea):
        """
        Calculates the electron and absorption density superposition of user-provided
        files and the electron density profile in the uploaded file
        """
        d = np.array(d)
        rho = np.array(rho)
        mu = np.array(mu)
        sig = np.array(sig)
        n = len(d)
        protins = np.array(protins)
        cov = np.array(cov)
        protpos = np.array(protpos)
        protrho = np.array(protrho)
        layarea = np.array(layarea)
        protmu = np.zeros(protpos.shape)

        zprot = -(protpos - protpos[0] - np.sum(d[:-1]) + protins)
        dprot = np.diff(-protpos)
        protpos = np.cumsum(np.insert(-dprot, 0, -np.sum(d[:-1]) + protins))
        membpos = -1*np.cumsum(d)
        dprot = np.append(dprot, -protpos[-1] + protpos[-2])

        """Need to combine two arrays to create one array of thicknesses and associated
        electron density along with sig and mu arrays"""

        protmembpos = np.flip(np.unique(np.concatenate((membpos[:-1], protpos))), -1)
        protmembpos = np.append(protmembpos, protmembpos[-1] - dprot[-1])
        m = len(protmembpos)

        protmembrho, protmembmu, protmembsig = protmembcomb(protmembpos, membpos, d, rho, mu, sig,
                                                            protpos, protrho, layarea, cov)

        zprotmemb = -protmembpos
        """dprotmemb = np.append(np.diff(-1 * protmembpos), d[-1])"""
        protmembrho = np.insert(protmembrho, 0, rho[0])
        protmembmu = np.insert(protmembmu, 0, mu[0])
        protmembsig = np.insert(protmembsig, 0, sig[0])

        maxsig = max(np.abs(np.max(protmembsig[1:])), 3)
        Nlayers = int((protmembpos[0] - protmembpos[-1] + 10 * maxsig) / dz)
        halfstep = (np.sum(d[:-1]) + 10 * maxsig) / 2 / Nlayers
        __z__ = np.linspace(-protmembpos[0] - 5*maxsig + halfstep, -protmembpos[-1] + 5*maxsig - halfstep,
                            Nlayers)
        __d__ = np.diff(__z__)
        __d__ = np.append(__d__, [__d__[-1]])
        __rho__ = self.sldCalFun(tuple(-protmembpos), tuple(protmembrho), tuple(protmembsig), tuple(__z__))
        __mu__ = self.sldCalFun(tuple(-protmembpos), tuple(protmembmu), tuple(protmembsig), tuple(__z__))

        return n, __z__, __d__, __rho__, __mu__, zprot, protmu, zprotmemb, protmembrho, protmembmu


    @lru_cache(maxsize=10)
    def sldCalFun(self, pos, y, sigma, x):
        """wholesld = []"""
        pos = np.array(pos)
        y = np.array(y)
        x = np.array(x)
        sigma = np.array(sigma)

        """posterm = np.tile(np.append(pos, 2 * pos[-1] - pos[-2]), [len(x), 1])
        sigmaterm = np.tile(np.reciprocal(np.append(sigma[1:], sigma[-1])),[len(x), 1])
        yterm = np.tile(np.append(y[1:], y[-1]) - y, [len(x), 1])
        xterm = np.transpose(np.tile(x, [len(y), 1]))

        sldvec = (special.erf((xterm - posterm) * sigmaterm / math.sqrt(2)) * yterm)
        wholesld = (np.sum(sldvec[:, :-1], axis=1) + y[0] + y[-1]) / 2
        wholesld[wholesld < 0] = 0
        wholesld = tuple(wholesld)"""

        """wholesld = []
        for j in range(len(x)):
            sld = 0
            for i in range(len(pos) - 1):
                sld += math.erf((x[j] - pos[i]) / sigma[i + 1] / math.sqrt(2)) * (y[i + 1] - y[i])
            wholesld.append(max((sld + y[0] + y[-1]) / 2, 0))
            sldvec = (special.erf((x[j] - np.append(pos, 2 * pos[-1] - pos[-2])) *
                                  np.reciprocal(np.append(sigma[1:], sigma[-1])) /
                                  math.sqrt(2)) * (np.append(y[1:], y[-1]) - y))
            sld = np.sum(sldvec[:-1])"""

        wholesld = tuple(sldCalFun_numba(pos, y, sigma, x))
        return wholesld

    @lru_cache(maxsize=10)
    def stepFun(self, zmin, zmax, d, rho, mu):
        tdata=[[zmin, rho[0], mu[0]]]
        z=np.cumsum(d)
        for i, td in enumerate(d[:-1]):
            tdata.append([z[i], rho[i], mu[i]])
            tdata.append([z[i], rho[i+1], mu[i+1]])
        tdata.append([zmax, rho[-1], mu[-1]])
        tdata=np.array(tdata)
        return tdata[:, 0], tdata[:, 1], tdata[:, 2]

    @lru_cache(maxsize=10)
    def protstepFun(self, zmin, zmax, z, rho, mu):
        tdata = [[zmin, rho[0], mu[0]]]
        for i, td in enumerate(z[:-1]):
            tdata.append([z[i], rho[i], mu[i]])
            tdata.append([z[i], rho[i+1], mu[i+1]])
        tdata.append([zmax, rho[-1], mu[-1]])
        tdata = np.array(tdata)
        return tdata[:, 0], tdata[:, 1], tdata[:, 2]

    @lru_cache(maxsize=10)
    def py_parratt(self, x, lam, d, rho, mu):
        return parratt_numba(np.array(x), lam, np.array(d), np.array(rho), np.array(mu))
# return parratt(np.array(x), lam, np.array(d), np.array(rho), np.array(mu))

    def update_parameters(self):
        for mkey in self.__mpar__.keys():
            # for key in self.__mpar__[mkey].keys():
            Nlayers = len(self.__mpar__[mkey]['d'])
            self.__d__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'd', i)].value for i in range(Nlayers)])
            self.__rho__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'rho', i)].value for i in range(Nlayers)])
            self.__mu__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'mu', i)].value for i in range(Nlayers)])
            sig=np.array([self.params['__%s_%s_%03d' % (mkey, 'sig', i)].value for i in range(Nlayers)])
            sig=np.where(sig<1e-5,1e-5,sig)
            self.__sig__[mkey] = tuple(sig)

    def y(self):
        """
        Define the function in terms of x to return some value
        """
        x = self.x + self.qoff
        lam = 6.62607004e-34 * 2.99792458e8 * 1e10 / self.E / 1e3 / 1.60217662e-19
        phitemp = self.params['phi'].value % 360
        thetatemp = self.params['theta'].value % 360
        if thetatemp > 180:
            self.params['theta'].value = 360 - thetatemp
            self.params['phi'].value = (phitemp + 180) % 360
        else:
            self.params['phi'].value = phitemp

        if not self.__fit__:
            for mkey in self.__mpar__.keys():
                Nlayers = len(self.__mpar__[mkey]['sig'])
                if self.fix_sig:
                    for i in range(2,Nlayers):
                        self.params['__%s_%s_%03d'%(mkey,'sig',i)].expr='__%s_%s_%03d'%(mkey,'sig',1)
                    self.__fix_sig_changed__=1
                if not self.fix_sig and self.__fix_sig_changed__>0:
                    for i in range(2,Nlayers):
                        self.params['__%s_%s_%03d' % (mkey, 'sig', i)].expr = None
                    self.__fix_sig_changed__=0
        xcoord, ycoord, zcoord, electrons, radius, mass = self.pdbread(self.fname)
        self.__protvol__ = self.protvolcalc(tuple(xcoord), tuple(ycoord), tuple(zcoord), tuple(radius), self.fslice)
        self.__protpos__, self.__protrho__, self.__layarea__ = self.ED_profrot(tuple(xcoord), tuple(ycoord),
                                                                               tuple(zcoord), tuple(radius),
                                                                               tuple(electrons), self.theta, self.phi,
                                                                               self.fslice, self.__protvol__)
        self.update_parameters()
        mkey=list(self.__mpar__.keys())[0]
        n, z, d, rho, mu, zprot, protmu, zprotmemb, protmembrho, protmembmu = self.calcProfComb(self.__d__[mkey], self.__rho__[mkey],
                                            self.__mu__[mkey], self.__sig__[mkey],
                                            self.protins, self.cov, mkey, self.dz,
                                            tuple(self.__protpos__), tuple(self.__protrho__), tuple(self.__layarea__))
        if not self.__fit__:
            tz, trho, tmu = self.stepFun(z[0], z[-1], self.__d__[mkey], self.__rho__[mkey], self.__mu__[mkey])
            pz, prho, pmu = self.protstepFun(zprot[0], zprot[-1], tuple(zprot + (zprot[-1] - zprot[-2])), tuple(self.__protrho__), tuple(protmu))
            ptz, ptrho, ptmu = self.protstepFun(z[0], z[-1], tuple(zprotmemb), tuple(protmembrho), tuple(protmembmu))
            self.output_params['%s_EDP' % self.__mkeys__[0]] = {'x': z, 'y': rho, 'names':['z (Angs)','rho (el/Angs^3)'],'plotType':'step'}
            self.output_params['%s_dEDP' % self.__mkeys__[0]] = {'x': z[:-1], 'y': np.diff(rho)/self.dz,
                                                                'names': ['z (Angs)', 'drho/dz (el/Angs^4)'],'plotType':'step'}
            self.output_params['%s_step_layEDP' % self.__mkeys__[0]]={'x': tz, 'y': trho, 'names':['z (Angs)','rho (el/Angs^3)'],'plotType':'step'}
            self.output_params['%s_step_protEDP' % self.__mkeys__[0]] = {'x': pz, 'y':prho, 'names': ['z (Angs)', 'rho (el/Angs^3)'], 'plotType': 'step'}
            self.output_params['%s_step_EDP' % self.__mkeys__[0]] = {'x': ptz, 'y': ptrho, 'names': ['z (Angs)', 'rho (el/Angs^3)'], 'plotType': 'step'}
            self.output_params['%s_ADP' % self.__mkeys__[0]] = {'x': z, 'y': mu, 'names':['z (Angs)','mu (1/cm)'],'plotType':'step'}
            self.output_params['%s_dADP' % self.__mkeys__[0]] = {'x': z[:-1], 'y': np.diff(mu)/self.dz, 'names': ['z (Angs)', 'dmu/dz (1/cm^2)'],'plotType':'step'}
            self.output_params['%s_step_ADP' % self.__mkeys__[0]] = {'x': tz, 'y': tmu, 'names':['z (Angs)','mu (1/cm)'], 'plotType':'step'}
        refq, r2 = self.py_parratt(tuple(x), lam, tuple(d), tuple(rho), tuple(mu))
        if self.rrf:
            rhos = (self.params['__%s_rho_000'%(mkey)].value, self.params['__%s_rho_%03d' % (mkey,n - 1)].value)
            mus = (0,0)
            ref, r2 = self.py_parratt(tuple(x - self.qoff), lam, (0.0, 1.0), rhos, mus)
            refq = refq / ref
        return refq * self.yscale+self.bkg


if __name__=='__main__':
    x=np.linspace(0.001,1.0,200)
    fun=XLayersPDB(x=x)
    print(fun.y())
