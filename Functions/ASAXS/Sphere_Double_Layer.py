####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
from functools import lru_cache

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_routines'))
####Please do not remove lines above####

####Import your modules below if needed####
from FormFactors.Sphere import Sphere
from ff_sphere import ff_sphere_ml
from Chemical_Formula import Chemical_Formula
from PeakFunctions import LogNormal, Gaussian
from utils import find_minmax, calc_rho
import time
from functools import lru_cache



class Sphere_Double_Layer: #Please put the class name same as the function name
    def __init__(self, x=0, Np=20, flux=1e13, dist='Gaussian', Energy=None, relement='Au', NrDep=False, norm=1.0e-4,
                 sbkg=0.0, cbkg=0.0, abkg=0.0, nearIon='Rb', farIon='Cl', ionDensity=0.0, stThickness=1.0,
                 stDensity=0.0, dbLength=1.0, dbDensity=0.0,Ndb=20,Rsig=0.0,D=0.0,phi=0.1,U=-1.0,SF=None,term='Total',
                 mpar={'Layers':{'Material': ['Au', 'H2O'], 'Density': [19.32, 1.0], 'SolDensity': [1.0, 1.0],
                                      'Rmoles': [1.0, 0.0], 'R': [1.0, 0.0]}}):
        """
        Documentation
        Calculates the Energy dependent form factor of multilayered nanoparticles with different materials

        x           : Reciprocal wave-vector 'Q' inv-Angs in the form of a scalar or an array
        relement    : Resonant element of the nanoparticle. Default: 'Au'
        Energy      : Energy of X-rays in keV at which the form-factor is calculated. Default: None
        Np          : No. of points with which the size distribution will be computed. Default: 10
        NrDep       : Energy dependence of the non-resonant element. Default= 'False' (Energy independent), 'True' (Energy dependent)
        dist        : The probablity distribution fucntion for the radii of different interfaces in the nanoparticles. Default: Gaussian
        norm        : The density of the nanoparticles in Molar (Moles/Liter)
        sbkg        : Constant incoherent background for SAXS-term
        cbkg        : Constant incoherent background for cross-term
        abkg        : Constant incoherent background for Resonant-term
        nearIon     : The ionic layer closer to the particle
        farIon      : The ionic layer farther from the particle
        ionDensity  : The bulk density of the ions in Moles per liter
        stThickness : Thickness of the stern layer
        stDensity   : Density of the ions in the stern layer in Moles per liter
        dbLength    : The ratio of decay length and the stern layer thickness
        dbDensity   : The ratio of maximum density of the debye layer w.r.t the stern layer density
        Ndb         : Number of layers used to represent the double layer region
        flux        : Total X-ray flux to calculate the errorbar to simulate the errorbar for the fitted data
        Rsig        : Width of the overall particle size distribution
        D           : Hard Sphere Diameter
        phi         : Volume fraction of particles
        U           : The sticky-sphere interaction energy
        SF          : Type of structure factor. Default: 'None'
        term        : 'Total' or 'SAXS-term' or 'Resonant-term' or 'Cross-term'
        mpar        : Multi-parameter which defines the following including the solvent/bulk medium which is the last one. Default: 'H2O'
                        Material ('Materials' using chemical formula),
                        Density ('Density' in gm/cubic-cms),
                        Density of solvent ('SolDensity' in gm/cubic-cms) of the particular layer
                        Mole-fraction ('Rmoles') of resonant element in the material)
                        Radii ('R' in Angs)
        """
        if type(x) == list:
            self.x = np.array(x)
        else:
            self.x = x
        self.norm = norm
        self.sbkg = sbkg
        self.cbkg = cbkg
        self.abkg = abkg
        self.dist = dist
        self.nearIon=nearIon
        self.farIon=farIon
        self.ionDensity=ionDensity
        self.stThickness=stThickness
        self.stDensity=stDensity
        self.dbLength=dbLength
        self.dbDensity=dbDensity
        self.Ndb=Ndb
        self.Np = Np
        self.Energy = Energy
        self.relement = relement
        self.NrDep = NrDep
        self.Rsig=Rsig
        self.SF=SF
        self.D=D
        self.U=U
        self.phi=phi
        self.term=term
        # self.rhosol=rhosol
        self.flux = flux
        self.__mpar__ = mpar  # If there is any multivalued parameter
        self.choices = {'dist': ['Gaussian', 'LogNormal'],'SF':['None','Hard-Sphere', 'Sticky-Sphere'],
                        'term':['SAXS-term','Cross-term','Resonant-term','Total'], 'NrDep': [True, False],'term':{'Total','SAXS-term','Cross-term','Resonant-term'}}  # If there are choices available for any fixed parameters
        self.__cf__ = Chemical_Formula()
        self.__fit__ = False
        self.output_params={'scaler_parameters':{}}
        self.__mkeys__=list(self.__mpar__.keys())
        self.init_params()

    def init_params(self):
        """
        Define all the fitting parameters like
        self.params.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params = Parameters()
        self.params.add('norm', value=self.norm, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('stThickness', value=self.stThickness, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('stDensity',value=self.stDensity, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('dbLength', value=self.dbLength, vary=0, min=1.0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('dbDensity', value=self.dbDensity, vary=0, min=0, max=1, expr=None, brute_step=0.1)
        self.params.add('sbkg', value=self.sbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('cbkg', value=self.cbkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('abkg', value=self.abkg, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('Rsig', value=self.Rsig, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('D', value=self.D, vary=0, min=0.0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('phi', value=self.phi, vary=0, min=0.0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('U', value=self.U, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        for mkey in self.__mkeys__:
            for key in self.__mpar__[mkey]:
                if key != 'Material':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey, key, i), value=self.__mpar__[mkey][key][i], vary=0, min=-np.inf,
                                        max=np.inf, expr=None, brute_step=0.1)


    @lru_cache(maxsize=10)
    def solrho(self, Rp=100.0, Rc=12.5, strho=1.0, tst=2.0, lrho=0.5, lexp=10.0, rhosol=0.0, R=(1.0),
                                material=('H2O'),density=(1.0),sol_density=(1.0),relement='Au',Rmoles=(1.0)):
        """
        Calculates the electron density for the bulk distribution of ions following double layer distribution surrounding a spherical particle

        Rp       :: Radius of the sphere in Angstroms enclosing the spherical particle
        Rc       :: Radial distance in Angstroms after which the solvent contribution starts
        strho    :: Concentration of the ions of interest in the stern layer in Molar
        tst      :: Thickness of stern layer in Angstroms
        lrho     :: The maximum concentration of the diffuse layer in Molars
        lexp     :: The decay length of the diffuse layer assuming exponential decay
        rhosol   :: The surrounding bulk density
        R        :: Tuple of radii of the layers of the multilayered particle
        material :: Tuple of material of the layers of the multilayered particle
        density  :: Tuple of density of layers of the multilayered of particle
        sol_density :: Tuple of densities of the layers if the particle layers are dissolved in a solvent
        relement  :: Resonant element
        Rmoles    :: The mole of the resonant element in each of the layers of the particle
        """
        R1=Rc
        R2=Rc+tst
        #integral=np.sum([r1**2*np.exp(-(r1-R2)/lexp) for r1 in np.linspace(R2,Rp,1001)])*(Rp-R2)/1000
        #integral=lexp*(R2**2*np.exp(-R2/lexp)-Rp**2*np.exp(-Rp/lexp))+2*lexp**2*(R2*np.exp(-R2/lexp)-Rp*np.exp(-Rp/lexp))+2*lexp**3*(np.exp(-Rp/lexp)-np.exp(-R2/lexp))
        r1=0
        tnear=0.0
        tfar=0.0
        nion = self.__cf__.parse(self.nearIon)
        mwnion = self.__cf__.molecular_weight()
        fion = self.__cf__.parse(self.farIon)
        mwfion = self.__cf__.molecular_weight()
        salt=self.nearIon+self.farIon
        salt_formula=self.__cf__.parse(salt)
        saltmw=self.__cf__.molecular_weight()
        salt_mole_ratio=self.__cf__.element_mole_ratio()
        density=np.array(density)
        for i, tR in enumerate(R):
            r2=r1+tR
            mat = material[i].split(':')
            if len(mat) == 2:
                solute, solvent = mat
                element_adjust = None
                if '*' in solute:
                    m = solute.split('*')[0]
                    f = self.__cf__.parse(m)
                    element_adjust = self.__cf__.elements()[-1]
                solute_formula = self.__cf__.parse(solute)
                fac = 1.0
                if relement in solute_formula.keys():
                    if element_adjust is not None:
                        self.__cf__.formula_dict[relement] = 0.0
                        t1 = self.__cf__.molar_mass()
                        self.__cf__.formula_dict[element_adjust] = self.__cf__.element_mole_ratio()[element_adjust] - Rmoles[i]
                        self.__cf__.formula_dict[relement] = Rmoles[i]
                        t2 = self.__cf__.molar_mass()
                        if t1 > 0:
                            fac = t2 / t1
                density[i] = fac * density[i]
                solute_elements = self.__cf__.elements()
                solute_mw = self.__cf__.molecular_weight()
                solute_mv = self.__cf__.molar_volume()
                solute_mole_ratio = self.__cf__.element_mole_ratio()

                solvent_formula = self.__cf__.parse(solvent)
                solvent_elements = self.__cf__.elements()
                solvent_mw = self.__cf__.molecular_weight()
                solvent_mole_ratio = self.__cf__.element_mole_ratio()

                solvent_moles = sol_density[i]*(1-solute_mv*density[i]/solute_mw)/ solvent_mw
                solute_moles = density[i] / solute_mw
                total_moles = solvent_moles + solute_moles
                solvent_mole_fraction = solvent_moles / total_moles
                solute_mole_fraction = solute_moles / total_moles
                comb_material = ''
                for ele in solute_mole_ratio.keys():
                    comb_material += '%s%.10f' % (ele, solute_mole_ratio[ele] * solute_mole_fraction)
                for ele in solvent_mole_ratio.keys():
                    comb_material += '%s%.10f' % (ele, solvent_mole_ratio[ele] * solvent_mole_fraction)
                tdensity = density[i] + sol_density[i] * (1 - solute_mv * density[i] / solute_mw)
                # self.output_params['scaler_parameters']['density[%s]' % material[i]]=tdensity
            else:
                element_adjust = None
                if '*' in material[i]:
                    m = material[i].split('*')[0]
                    f = self.__cf__.parse(m)
                    element_adjust = self.__cf__.elements()[-1]
                formula = self.__cf__.parse(material[i])
                fac = 1.0
                if relement in formula.keys():
                    self.__cf__.formula_dict[relement] = 0.0
                    t1 = self.__cf__.molar_mass()
                    if element_adjust is not None:
                        self.__cf__.formula_dict[element_adjust] = self.__cf__.element_mole_ratio()[element_adjust]-Rmoles[i]
                    self.__cf__.formula_dict[relement] = Rmoles[i]
                    t2 = self.__cf__.molar_mass()
                    if t1 > 0:
                        fac = t2 / t1
                mole_ratio = self.__cf__.element_mole_ratio()
                comb_material = ''
                for ele in mole_ratio.keys():
                    comb_material += '%s%.10f' % (ele, mole_ratio[ele])
                # comb_material=material[i]
                tdensity = density[i]
                # self.output_params['scaler_parameters']['density[%s]' % material[i]] = tdensity
            formula = self.__cf__.parse(comb_material)
            molwt = self.__cf__.molecular_weight()
            elements = self.__cf__.elements()
            mole_ratio = self.__cf__.element_mole_ratio()
            if self.nearIon in material[i]:
                if len(nion) > 1:
                    nmratio = 1.0
                else:
                    nmratio = mole_ratio[self.nearIon]
                tnear=tnear+(r2**3-r1**3)*tdensity*1e3*nmratio/molwt#mwnion
            elif self.farIon in material[i]:
                if len(fion) > 1:
                    fmratio = 1.0
                else:
                    fmratio = mole_ratio[self.farIon]
                tfar=tfar+(r2**3-r1**3)*tdensity*1e3*fmratio/molwt#mwfion
            r1=r2+0.0
        integral=(R2**2*lexp+2*R2*lexp**2+2*lexp**3)-(Rp**2*lexp+2*Rp*lexp**2+2*lexp**3)*np.exp((R2-Rp)/lexp)
        if 3*lrho*integral>=rhosol*(Rp**3-R1**3)-strho*(R2**3-R1**3)-tnear:
            near=0.00
            lrho=(rhosol*(Rp**3-R1**3)-strho*(R2**3-R1**3)-tnear)/3/integral
        else:
            near=(rhosol*(Rp**3-R1**3)-strho*(R2**3-R1**3)-3*lrho*integral-tnear)/(Rp**3-R2**3-3*integral)
        far=(rhosol*(Rp ** 3 - R1 ** 3)-tfar)/(Rp **3-R2**3 -3*lrho*integral)
        self.output_params['scaler_parameters']['tnear']=tnear
        self.output_params['scaler_parameters']['tfar']=tfar
        self.output_params['scaler_parameters']['rho_near']=near
        self.output_params['scaler_parameters']['rho_far']=far
        return near, far # in Moles/Liter

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
    def new_sphere(self, q, R, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10):
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
            cform = cform + rdist[i] * (meiff.conjugate() * maff+meiff*maff.conjugate())
        return pfac * form, pfac * eiform, pfac * aform, np.abs(pfac * cform)/2.0  # in cm^2

    @lru_cache(maxsize=2)
    def new_sphere_dict(self, q, R, Rsig, rho, eirho, adensity, dist='Gaussian', Np=10, key='SAXS-term'):
        form, eiform, aform, cform = self.new_sphere(q, R, Rsig, rho, eirho, adensity, dist=dist, Np=Np)
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
        scale=1e27/6.022e23
        self.update_params()
        tR=self.__R__[:-1]
        tnmaterial=self.__material__[:-1]
        tfmaterial =self.__material__[:-1]
        tndensity=self.__density__[:-1]
        tfdensity = self.__density__[:-1]
        tsoldensity = self.__sol_density__[:-1]
        tRmoles=self.__Rmoles__[:-1]
        Rp = (3 / (4 * np.pi * self.norm * 6.022e23)) ** (1.0 / 3.0) * 1e9
        Rc=np.sum(tR)
        near, far = self.solrho(Rp=Rp, Rc=Rc, strho=self.stDensity, tst=self.stThickness, lrho=self.dbDensity*self.stDensity,
                                lexp=self.dbLength*self.stThickness, rhosol=self.ionDensity, R=tuple(tR),
                                material=tuple(tnmaterial), density=tuple(tndensity), sol_density=tuple(tsoldensity),
                                relement=self.relement,Rmoles=tuple(tRmoles))
        dbr=[self.stThickness]
        dbr=dbr+[(Rp-Rc-self.stThickness)/self.Ndb for i in range(self.Ndb)]
        nden=[self.stDensity]
        fden=[0.0]
        nden=np.append(nden,self.dbDensity*self.stDensity*np.exp(-np.cumsum(dbr[1:])/self.dbLength/self.stThickness)+near)
        fden=np.append(fden,far*(1-np.exp(-np.cumsum(dbr[1:])/self.dbLength/self.stThickness)))
        self.output_params['scaler_parameters']['Rp']=Rp
        nmf=self.__cf__.parse(self.nearIon)
        nmw=self.__cf__.molecular_weight()
        fmf=self.__cf__.parse(self.farIon)
        fmw=self.__cf__.molecular_weight()
        for i in range(len(dbr)):
            tR.append(dbr[i])
            tsoldensity.append(self.__sol_density__[-1])

            tndensity.append(2*nden[i]*nmw/1000) # converting int gms/cubic-cms
            tfdensity.append(2*fden[i]*fmw/1000) # converting int gms/cubic-cms
            tnmaterial.append('%s:%s'%(self.nearIon,self.__material__[-1]))
            tfmaterial.append('%s:%s' % (self.farIon,self.__material__[-1]))
            if self.relement in self.nearIon:
                tRmoles.append(1.0)
            elif self.relement in self.farIon:
                tRmoles.append(1.0)
            else:
                tRmoles.append(1.0)
        rhon, eirhon, adensityn, rhorn, eirhorn, adensityrn = calc_rho(R=tuple(tR), material=tuple(tnmaterial),relement=self.relement,
                                                                            density=tuple(tndensity), sol_density=tuple(tsoldensity),
                                                                            Energy=self.Energy, Rmoles=tuple(tRmoles), NrDep=self.NrDep)
        rhof, eirhof, adensityf, rhorf, eirhorf, adensityrf = calc_rho(R=tuple(tR), material=tuple(tfmaterial),relement=self.relement,
                                                                            density=tuple(tfdensity), sol_density=tuple(tsoldensity),
                                                                            Energy=self.Energy, Rmoles=tuple(tRmoles), NrDep=self.NrDep)
        rho, eirho, adensity = (rhon+rhof)/2, (eirhon+eirhof)/2, (adensityn+adensityf)/2
        rhor,eirhor, adensityr = rhorn, eirhorn, adensityrn
        rhor[:,1]=(rhor[:,1]+rhorf[:,1])/2
        eirhor[:, 1] = (eirhor[:, 1] + eirhorf[:, 1])/2
        adensityr[:,1]=(adensityr[:,1]+ adensityrf[:,1])/2

        if type(self.x) == dict:
            sqf = {}
            for key in self.x.keys():
                sqf[key] = self.norm * 6.022e20 * self.new_sphere_dict(tuple(self.x[key]), tuple(tR),
                                                                       self.Rsig, tuple(rho), tuple(eirho),
                                                                       tuple(adensity), key=key, dist=self.dist,
                                                                       Np=self.Np)  # in cm^-1
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
            total = self.norm * 6.022e20 * struct * self.new_sphere_dict(tuple(self.x[key]), tuple(tR),
                                                                         self.Rsig, tuple(rho), tuple(eirho),
                                                                         tuple(adensity),
                                                                         key=key1, dist=self.dist,
                                                                         Np=self.Np) + self.sbkg  # in cm^-1
            if not self.__fit__:
                dr, rdist, totalR = self.calc_Rdist(tuple(self.__R__), self.Rsig, self.dist, self.Np)
                self.output_params['Distribution'] = {'x': dr, 'y': rdist}
                self.output_params['Simulated_total_wo_err'] = {'x': self.x[key], 'y': total}
                self.output_params['Total'] = {'x': self.x[key], 'y': total}
                for key in self.x.keys():
                    self.output_params[key] = {'x': self.x[key], 'y': sqf[key]}
                self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
                self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
                self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1]*scale,'names':['r (Angs)','Density (Molar)']}# in Molar
                self.output_params['rhon_r'] = {'x': rhorn[:, 0], 'y': rhorn[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
                self.output_params['eirhon_r'] = {'x': eirhorn[:, 0], 'y': eirhorn[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
                self.output_params['adensityn_r'] = {'x': adensityrn[:, 0], 'y': adensityrn[:, 1]*scale,'names':['r (Angs)','Density (Molar)']}# in Molar
                self.output_params['rhof_r'] = {'x': rhorf[:, 0], 'y': rhorf[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
                self.output_params['eirhof_r'] = {'x': eirhorf[:, 0], 'y': eirhorf[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
                self.output_params['adensityf_r'] = {'x': adensityrf[:, 0], 'y': adensityrf[:, 1]*scale,'names':['r (Angs)','Density (Molar)']}# in Molar
                self.output_params['Structure_Factor']={'x':self.x[key],'y':struct}
        else:
            if self.SF is None:
                struct = np.ones_like(self.x)
            elif self.SF == 'Hard-Sphere':
                struct = hard_sphere_sf(self.x, D=self.D, phi=self.phi)
            else:
                struct = sticky_sphere_sf(self.x, D=self.D, phi=self.phi, U=self.U, delta=0.01)

            tsqf, eisqf, asqf, csqf = self.new_sphere(tuple(self.x), tuple(tR), self.Rsig, tuple(rho),
                                                      tuple(eirho), tuple(adensity), dist=self.dist, Np=self.Np)
            sqf = self.norm * np.array(tsqf) * 6.022e20 * struct + self.sbkg  # in cm^-1
            # if not self.__fit__:  # Generate all the quantities below while not fitting
            dr, rdist, totalR = self.calc_Rdist(tuple(self.__R__), self.Rsig, self.dist, self.Np)
            self.output_params['Distribution'] = {'x': dr, 'y': rdist}
            asqf = self.norm * np.array(asqf) * 6.022e20 + self.abkg  # in cm^-1
            eisqf = self.norm * np.array(eisqf) * 6.022e20 + self.sbkg  # in cm^-1
            csqf = self.norm * np.array(csqf) * 6.022e20 + self.cbkg  # in cm^-1
            svol = 0.2 ** 2 * 1.5 * 1e-3  # scattering volume in cm^3
            sqerr = np.sqrt(self.flux * sqf * svol)
            sqwerr = (sqf * svol * self.flux + 2 * (0.5 - np.random.rand(len(sqf))) * sqerr)
            dr, rdist, totalR = self.calc_Rdist(tuple(self.__R__), self.Rsig, self.dist, self.Np)
            self.output_params['Distribution'] = {'x': dr, 'y': rdist}
            self.output_params['simulated_total_w_err'] = {'x': self.x, 'y': sqwerr, 'yerr': sqerr}
            self.output_params['simulated_total_wo_err'] = {'x': self.x, 'y': sqf * svol * self.flux}
            self.output_params['Total'] = {'x': self.x, 'y': sqf}
            self.output_params['SAXS-term'] = {'x': self.x, 'y': eisqf}
            self.output_params['Cross-term'] = {'x': self.x, 'y': csqf}
            self.output_params['Resontant-term'] = {'x': self.x, 'y': asqf}
            self.output_params['rho_r'] = {'x': rhor[:, 0], 'y': rhor[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
            self.output_params['eirho_r'] = {'x': eirhor[:, 0], 'y': eirhor[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
            self.output_params['adensity_r'] = {'x': adensityr[:, 0], 'y': adensityr[:, 1]*scale, 'names':['r (Angs)','Density (Molar)']} # in Molar
            self.output_params['rhon_r'] = {'x': rhorn[:, 0], 'y': rhorn[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
            self.output_params['eirhon_r'] = {'x': eirhorn[:, 0], 'y': eirhorn[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
            self.output_params['adensityn_r'] = {'x': adensityrn[:, 0], 'y': adensityrn[:, 1]*scale,'names':['r (Angs)','Density (Molar)']} # in Molar
            self.output_params['rhof_r'] = {'x': rhorf[:, 0], 'y': rhorf[:, 1],'names':['r (Angs)','Electron Density (el/Angs^3)']}
            self.output_params['eirhof_r'] = {'x': eirhorf[:, 0], 'y': eirhorf[:, 1],'names':['r(Angs)','Electron Density (el/Angs^3)']}
            self.output_params['adensityf_r'] = {'x': adensityrf[:, 0], 'y': adensityrf[:, 1]*scale,'names':['r (Angs)','Density (Molar)']} # in Molar
            self.output_params['Structure_Factor'] = {'x': self.x, 'y': struct}
            sqf=self.output_params[self.term]['y']
        return sqf


if __name__ == '__main__':
    x = np.arange(0.001, 1.0, 0.001)
    fun = Sphere_Double_Layer(x=x)
    print(fun.y())
