####Please do not remove lines below####
from lmfit import Parameters
import numpy as np
import sys
import os
import scipy.constants
import re
import cmath
from xraydb import XrayDB
xdb = XrayDB()
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))
sys.path.append(os.path.abspath('./Fortran_rountines'))
####Please do not remove lines above####

####Import your modules below if needed####
from numba import jit

@jit(nopython=True)
def fluCalFun(x, topdel, topbet, botdel, botbet, flutopdel, flutopbet, flubotdel, flubotbet, flutopmu, flubotmu,
              topmu, k0, conbot, contop, beamwid, detlen, gl2, qz, samplesize, qoff, shoff, gl2off, detoff, Rc, sur_cov,
              x_axis, beam_profile):
    if x_axis == 'Qz':  # qz scan
        alpha = (x + qoff) / 2 / k0  # get incident angle
        delx = -shoff / alpha + gl2off  # get the footprint shift due to "sh" off and "gl2" off in unit of /AA
    else:  # sh scan
        alpha = (qz + qoff) / 2 / k0 * np.ones_like(x)
        totshoff = shoff + x + gl2 * alpha  # total footprint shift from the center of the sample in unit of /AA
        delx = -totshoff / alpha + gl2off
    if beam_profile == 'Uniform':
        fprint = beamwid / alpha  # get the footprint in unit of /AA for the rectangular beam
    else:
        fprint = 6.0 * beamwid / alpha  # get the footprint in unit of /AA for the Gaussian beam

    flu = []
    top_con = []
    bot_con = []
    sur_con = []
    for i in range(len(x)):
        steps = int(fprint[i] / 1e5)  # use 0.1 mm as the step size
        stepsize = fprint[i] / steps
        # print(alpha[i], fprint[i], steps, stepsize)
        x0 = np.linspace(-fprint[i] / 2 + delx[i] + stepsize / 2, fprint[i] / 2 + delx[i] - stepsize / 2, steps)  # get the position of single ray hitting z=0 relative to the center of sample with the step size "steps"
        if beam_profile == 'Uniform':
            weight = 1 / beamwid * np.ones_like(x0)  # get the beam intensity weight at given x0 for the rectangular beam
            # weight = 1 * np.ones_like(x0)
        else:
            weight = 1 / (beamwid * np.sqrt(2 * np.pi)) * np.exp(-(x0 - delx[i]) ** 2 / (2 * beamwid ** 2))  # get the beam intensity weight for the Gaussian beam

        xprimetotal, zprimetotal = getxz(x0, Rc, alpha[i])  # get x' and z', where the ray hits the interface from a given x0 and alpha, including the 'nan', where the ray misses the interface
        xprime = xprimetotal[np.where(np.isnan(xprimetotal) == False)]  # get x' and z' for the ray hitting the interface
        zprime = zprimetotal[np.where(np.isnan(xprimetotal) == False)]
        x0hit = x0[np.where(np.isnan(xprimetotal) == False)]  # get x0 for the ray hitting the interface
        x0miss = x0[np.where(np.isnan(xprimetotal) == True)]  # get x0 for the ray missing the interface, which has intensity contribution on the top phase
        weighthit = weight[np.where(np.isnan(xprimetotal) == False)]  # get weight array for the ray hitting the interface
        weightmiss = weight[np.where(np.isnan(xprimetotal) == True)]  # get weight array for the ray missing the interface

        alphaprime = alpha[i] - xprime / Rc  # a' is the real incident angle at x' position
        x1 = xprime - zprime / (2 * alphaprime - alpha[i])  # the position of the reflected beam hitting z=0

        # print(x1)
        # print(xprime, zprime)

        pend, trans, refl = frsnllCal(topdel, topbet, botdel, botbet, k0, alphaprime)  # get penetration depth, the frsnll transmissivity, and the reflectivity from given a'

        # print(trans)
        mu_i = topmu / alpha[i] + flutopmu  # the effective absorption coefficient of the incident beam
        mu_r = -topmu / (2 * alphaprime - alpha[i]) + flutopmu  # the effective absorption coefficient of the reflected beam
        mu_t = alphaprime / (alpha[i] * pend) + flubotmu  # the effective absorption coefficient of the transimitted beam

        z_i1 = (x0hit - detlen / 2 - detoff) * alpha[i]  # the position of the incident/transmitted beam hit the downstream edge of the detecting area
        z_i2 = (x0hit + detlen / 2 - detoff) * alpha[i]  # the position of the incident/transmitted beam hit the upstream edge of the detecting area
        z_r1 = -(x1 - detlen / 2 - detoff) * (2 * alphaprime - alpha[i])  # the position of the reflected beam hit the downstream edge of the detecting area
        z_r2 = -(x1 + detlen / 2 - detoff) * (2 * alphaprime - alpha[i])  # the position of the reflected beam hit the upstream edge of the detecting area

        z_imed = np.median([z_i1, z_i2, zprime], axis=0)  # get median value of z_i1, z_i2, and zprime
        z_rmed = np.median([z_r1, z_r2, zprime], axis=0)  # get median value of z_r1, z_r2, and zprime

        # print(z_i1)
        # print(z_imed)
        # print(mu_t*z_imed, '/n')
        # print(mu_t*z_i1)
        # print(z_imed-z_i1)
        # print('x0hit', len(x0hit[np.where(np.isnan(x0hit)==True)]))
        # print('z_i2', len(z_i2[np.where(np.isnan(z_i2) == True)]))
        # print('z_imed', len(z_imed[np.where(np.isnan(z_imed) == True)]))
        # print('z_r1', len(z_r1[np.where(np.isnan(z_r1) == True)]))
        # print('z_rmed', len(z_rmed[np.where(np.isnan(z_rmed) == True)]))
        # print('mu_r', len(mu_r[np.where(np.isnan(mu_r) == True)]))
        # print('x1', len(x1[np.where(np.isnan(x1) == True)]))

        # print('temp1', np.where(np.isnan(np.exp(-topmu * x1)) == True))

        # temp = np.exp(-topmu * x0hit) * (np.exp(mu_i * z_i2) - np.exp(mu_i * z_imed))

        # print(mu_r)
        # print('x1',x1)
        # print('alpha', alphaprime)
        # print(np.exp(-topmu * x1))
        # print(np.exp(mu_r * z_r1) - np.exp(mu_r * z_rmed))
        # print(1.0 / mu_r * refl * (np.exp(mu_r * z_r1 - topmu * x1) - np.exp(mu_r * z_rmed - topmu * x1)))
        # int_top = 1.0 / mu_i * np.exp(-topmu * x0hit) * (np.exp(mu_i * z_i2) - np.exp(mu_i * z_imed)) + 1.0 / mu_r * refl * np.exp(-topmu * x1) * (np.exp(mu_r * z_r1) - np.exp(mu_r * z_rmed))
        int_top = 1.0 / mu_i * np.exp(-topmu * x0hit) * (np.exp(mu_i * z_i2) - np.exp(mu_i * z_imed)) + np.abs(
            1.0 / mu_r * refl * (np.exp(mu_r * z_r1 - topmu * x1) - np.exp(mu_r * z_rmed - topmu * x1)))
        int_bot = 1.0 / mu_t * trans * np.exp(-topmu * xprime) * np.exp(-alphaprime * zprime / alpha[i] / pend) * (
                    np.exp(mu_t * z_imed) - np.exp(mu_t * z_i1))
        int_sur = np.where((xprime < detlen / 2 + detoff) & (xprime > -detlen / 2 + detoff),
                           trans * np.exp(-topmu * xprime), 0)

        # print(int_bot)
        # print(int_sur)

        tot_top = stepsize * contop * np.sum(
            np.where(xprime > - samplesize / 2, int_top * weighthit, 0)) * scipy.constants.Avogadro / 1e27
        tot_bot = stepsize * conbot * np.sum(
            np.where(xprime > - samplesize / 2, int_bot * weighthit, 0)) * scipy.constants.Avogadro / 1e27
        tot_sur = stepsize * np.sum(int_sur * weighthit) * sur_cov

        if len(x0miss) != 0:  # for the ray missing the interface, (all of them from the downsteam side)
            # print('missing the interface at x = ', x[i])
            z_i1miss = (x0miss - detlen / 2 - detoff) * alpha[
                i]  # the position of the incident beam missing the interface hits the downstream edge of the detecting area
            z_i2miss = (x0miss + detlen / 2 - detoff) * alpha[
                i]  # the position of the incident beam missing the interface hits the upstream edge of the detecting area
            int_topmiss = 1.0 / mu_i * np.exp(-topmu * x0miss) * (np.exp(mu_i * z_i2miss) - np.exp(mu_i * z_i1miss))
            tot_top = tot_top + stepsize * contop * np.sum(int_topmiss * weightmiss) * scipy.constants.Avogadro / 1e27
        # print(np.where(xprime > -self.samplesize * 1e7 / 2, int_bot * weight, 0))
        # print(tot_bot)
        # print(tot_sur)
        int_tot = tot_top + tot_bot + tot_sur
        # print(int_tot)
        top_con.append(tot_top)
        bot_con.append(tot_bot)
        sur_con.append(tot_sur)
        flu.append(int_tot)
    # print(flu)
    return flu, top_con, bot_con, sur_con

@jit(nopython=True)
def getxz(x0, curvature, a0):
    a0 = a0 * np.ones_like(x0)
    a = a0 * a0 + 1
    b = -(2 * curvature * a0 + 2 * x0 * a0 * a0)
    c = 2 * curvature * a0 * x0 + x0 * x0 * a0 * a0
    if curvature > 0:
        x = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    else:
        x = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    z = a0 * (x0 - x)
    return x, z

@jit(nopython=True)
def frsnllCal(dett, bett, detb, betb, k0, alpha):
    f1 = np.sqrt((alpha * alpha+1j*2*bett))
    #fmax = np.sqrt(complex(alpha * alpha - 2 * (detb - dett), 2 * betb))
    fmax = np.sqrt((alpha * alpha - 2 * (detb - dett) + 1j * 2 * betb))
    eff_d = 1 / (2 * k0 * fmax.imag)
    trans = 4 * abs(f1 / (f1 + fmax)) * abs(f1 / (f1 + fmax))
    frsnll=abs((f1-fmax)/(f1+fmax))*abs((f1-fmax)/(f1+fmax))
    return eff_d, trans, frsnll




class XFNTR_ADV_F: #Please put the class name same as the function name
    def __init__(self,x=0.1,E=20.0, mpar={'Top':{'Components':['Sr'],'Concentration':[0.0],'Radius':[1.25]}, 'Bottom':{'Components':['SrCl2'],'Concentration':[50.0],'Radius':[2.388]}},
                 topchem='C12H26', topden=0.75, botchem='H2O', botden=0.997, element='Sr', line='Ka1', beam_profile='Uniform',
                 x_axis = 'Qz', beamwid = 0.02, detlen=12.7, gl2=1238.0, qz=0.01, samplesize=76, qoff=0.0, shoff=0.0, gl2off=0.0, detoff=0.0, yscale=1, int_bg=0, Rc=300, sur_cov=1):
        """
        Calculates X-ray reflectivity from a system of multiple layers using Parratt formalism

        x     	: array of wave-vector transfer along z-direction
        E     	: Energy of x-rays in units of keV
        topchem : chemical formula for the top phase
        topden  : mass density for the top phase in the unit of g/ml
        botchem : chemical formula for the bottom phase
        botden  : mass density for the bottom phase in the unit of g/ml
        ele:    : target element, e.g., 'Sr'
        line:   : emission line, e.g., 'Ka1'
        vslit   : vertical slits size in unit of mm
        detlen  : detector size projected on the surface in the unit of mm
        qoff  	: q-offset to correct the zero q of the instrument
        yscale  : a scale factor for the fluorescence intensity
        int_bg  : the background fluorescence intensity from the secondary scattering from the primary beam, should be zero for air/water interface
        Rc : the radius of the interfacial curvature in unit of meter; 0 means it's flat
        sur_cov : the surface coverage of target element in unit of per \AA^-2
        mpar:   : components with unknown concentrations(mM) and radius (\AA) in the top and bottom phase
        """
        if type(x)==list:
            self.x=np.array(x)
        else:
            self.x=x
        self.E=E
        self.__mpar__ = mpar
        self.topchem = topchem
        self.topden = topden
        self.botchem = botchem
        self.botden = botden
        self.element = element
        self.line = line
        self.beam_profile = beam_profile
        self.x_axis = x_axis
        self.beamwid = beamwid
        self.detlen = detlen
        self.gl2 = gl2
        self.qz = qz
        self.samplesize = samplesize
        self.qoff = qoff
        self.shoff = shoff
        self.gl2off = gl2off
        self.detoff = detoff
        self.yscale = yscale
        self.int_bg = int_bg
        self.Rc = Rc
        self.sur_cov = sur_cov
        #self.ion_depth = ion_depth
        elelist = xdb.atomic_symbols
        linelist = list(xdb.xray_lines(98).keys())
        self.choices={'element':elelist,'line': linelist, 'beam_profile':['Uniform', 'Gaussian'], 'x_axis':['Qz', 'sh']} #If there are choices available for any fixed parameters
        self.output_params = {}
        self.init_params()
        self.__fit__=False
        self.__avoganum__ = scipy.constants.Avogadro
        self.__eleradius__ = scipy.constants.physical_constants['classical electron radius'][0]*1e10 #classic electron radius in \AA
        self.output_params = {'scaler_parameters': {}}

    def init_params(self):
        """
        Define all the fitting parameters like
        self.param.add('sig',value = 0, vary = 0, min = -np.inf, max = np.inf, expr = None, brute_step = None)
        """
        self.params = Parameters()
        self.params.add('qoff', self.qoff, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('shoff', self.shoff, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('gl2off', self.gl2off, vary=0, min=-np.inf, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('yscale', self.yscale, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('int_bg', self.int_bg, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('Rc', self.Rc, vary=0, min=10, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('sur_cov', self.sur_cov, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)
        self.params.add('detoff', self.detoff, vary=0, min=0, max=np.inf, expr=None, brute_step=0.1)

        for mkey in self.__mpar__.keys():
            for key in self.__mpar__[mkey].keys():
                if key!='Components':
                    for i in range(len(self.__mpar__[mkey][key])):
                        self.params.add('__%s_%s_%03d' % (mkey,key, i), value=self.__mpar__[mkey][key][i], vary=0, min=0, max=np.inf, expr=None, brute_step=0.05)

    def parseFormula(self, chemfor):
        a = re.findall(r'[A-Z][a-z]?|[0-9]+[.][0-9]+|[0-9]+', chemfor)
        if not a[-1].replace('.', '').isdigit():
            a.append('1')
        formula = {}
        i = 1
        while i <= len(a):
            if not a[i].replace('.', '').isdigit():
                a.insert(i, '1')
            if a[i - 1] in formula.keys():
                formula[a[i - 1]] = float(a[i]) + formula[a[i - 1]]
            else:
                formula[a[i - 1]] = float(a[i])
            i += 2
        return formula

    def formStr(self, chemfor):
        string=''
        for i in range(len(chemfor)):
            key=list(chemfor.keys())[i]
            if chemfor[key]>0:
                string=string+key+str('{0:.3f}'.format(chemfor[key]).rstrip('0').rstrip('.'))
        return string

    def getDelBet(self, energy, chemfor, massden):
        k0 = 2 * np.pi * energy / 12.3984
        formula=self.parseFormula(chemfor)
        #print(energy,chemfor,massden)
        molarmass = np.sum([xdb.molar_mass(key) * formula[key] for key in formula.keys()])
        massratio = {}
        for key in formula.keys():
            massratio[key] = xdb.molar_mass(key) * formula[key] / molarmass
        molarele = np.sum([xdb.atomic_number(key) * formula[key] for key in formula.keys()])
        eleden = massden / molarmass * molarele * self.__avoganum__ / 1e24
        tot_mu = np.sum([xdb.mu_elam(key, energy * 1000) * massratio[key] * massden for key in massratio.keys()])  #in unit of cm
        #print(eleden, 10000/tot_mu)
        return self.__eleradius__*2*np.pi/k0/k0*eleden, tot_mu/2/k0/1e8

    def getBulkCon(self, element, chemfor, massden):
        formula = self.parseFormula(chemfor)
        molarmass = np.sum([xdb.molar_mass(key) * formula[key] for key in formula.keys()])
        try:
            return massden/molarmass*formula[element]*1000  #return the bulk concentration in unit of M
        except:
            return 0


    def updatePhaseInfo(self, solchem, solden, phase):
        solformula = self.parseFormula(solchem)  #  solvent formula
        chemfor = []  # list of chemical in that phase
        concen = []   # list of concentrations for each row in that phase
        radius = []   # list of radius for each row in that phase
        volume = 0    # total volume of chemical in that phase
       # Nlayers = len(self.__mpar__[mkey]['d'])
       # self.__d__[mkey] = tuple([self.params['__%s_%s_%03d' % (mkey, 'd', i)].value for i in range(Nlayers)])
        for i in range(len(self.__mpar__[phase]['Components'])):
            chemfor.append(self.parseFormula(str(self.__mpar__[phase]['Components'][i])))
            concen.append(float(self.params['__%s_%s_%03d' % (phase, 'Concentration', i)].value))
            radius.append(float(self.params['__%s_%s_%03d' % (phase, 'Radius', i)].value))
        totalformula = {}
        for i in range(len(concen)):
            volume = volume + concen[i] * pow(radius[i], 3)
            for j in range(len(chemfor[i])):  # merge components at all rows.
                key = list(chemfor[i].keys())[j]
                if key in totalformula:
                    totalformula[key] = totalformula[key] + chemfor[i][key] * concen[i]
                else:
                    totalformula[key] = chemfor[i][key] * concen[i]

        totalvolume = volume / 1000 * self.__avoganum__ * 1e-27 * 4 / 3 * np.pi  # total volume of all components in unit of liter
        solvolume = 1 - totalvolume  # solvent volume in unit of liter

        solmolarmass = np.sum(
            [xdb.molar_mass(key) * solformula[key] for key in solformula.keys()])  # molar mass for the solvent
        solmolar = solvolume * solden / solmolarmass * 1e6  # solvent molarity in mM

        for key in solformula.keys():  # add solvent into the totalformula
            if key in totalformula:
                totalformula[key] = totalformula[key] + solformula[key] * solmolar
            else:
                totalformula[key] = solformula[key] * solmolar

        chemstr = self.formStr(totalformula)
        molarmass = np.sum([xdb.molar_mass(key) * totalformula[key] for key in totalformula.keys()]) / 1e6
        return chemstr, molarmass






    def y(self):
        """
        Define the function in terms of x to return some value
        """

        self.newtopchem, self.newtopden = self.updatePhaseInfo(self.topchem, self.topden, 'Top')   # update the top phase chemical formula and density
        self.newbotchem, self.newbotden = self.updatePhaseInfo(self.botchem, self.botden, 'Bottom') # update the bottom phase chemical formula and density
        k0 = 2 * np.pi * self.E / 12.3984  # wave vector
        conbot = self.getBulkCon(self.element, self.newbotchem,self.newbotden)  # get the bottom bulk concentration of the target element
        contop = self.getBulkCon(self.element, self.newtopchem,self.newtopden)  # get the top bulk concentration of the target element

        #print(conbot, contop)

        fluene = xdb.xray_lines(self.element)[self.line].energy / 1000  # get the fluorescence energy in KeV

        topdel, topbet = self.getDelBet(self.E, self.newtopchem, self.newtopden)  # get top del & bet for the incident beam
        botdel, botbet = self.getDelBet(self.E, self.newbotchem, self.newbotden)  # get bottom del & bet for the incident beam
        flutopdel, flutopbet = self.getDelBet(fluene, self.newtopchem, self.newtopden)  # get top del & bet for the fluorescent beam
        flubotdel, flubotbet = self.getDelBet(fluene, self.newbotchem, self.newbotden)  # get bottom del & bet for the fluorescent beam

        flutopmu = flutopbet * 2 * (2 * np.pi * fluene / 12.3984)  # get the absorption coefficient for the fluorescent beam in the top phase  mu_eo  (\AA^-1)
        flubotmu = flubotbet * 2 * (2 * np.pi * fluene / 12.3984)  # get the absorption coefficient for the fluorescent beam in the bottom phase mu_ew (\AA^-1)
        topmu = topbet * 2 * k0  # get the absorption coefficient in top phase in \AA^-1 for the incident beam mu_io

        x = self.x
        beamwid = self.beamwid * 1e7
        detlen = self.detlen * 1e7
        gl2 = self.gl2 * 1e7
        qz = self.qz
        samplesize = self.samplesize * 1e7
        qoff = self.qoff
        shoff = self.shoff * 1e7
        gl2off = self.gl2off * 1e7
        detoff = self.detoff * 1e7
        Rc = self.Rc * 1e10
        sur_cov = self.sur_cov
        x_axis = self.x_axis
        beam_profile = self.beam_profile

        flu, top, bot, sur = fluCalFun(x, topdel, topbet, botdel, botbet, flutopdel, flutopbet, flubotdel, flubotbet, flutopmu, flubotmu, topmu, k0, conbot, contop, beamwid, detlen, gl2, qz, samplesize, qoff, shoff, gl2off, detoff, Rc, sur_cov,
              x_axis, beam_profile)
        #print(flu)
        #print(bot)
        if not self.__fit__:
            #flu, top, bot, sur = self.fluCalFun(x)
            self.output_params['Top phase contribution'] = {'x': x, 'y': top * self.yscale}
            self.output_params['Bottom phase contribution'] = {'x': x, 'y': bot * self.yscale}
            self.output_params['Interface contribution'] = {'x': x, 'y': sur * self.yscale}
        return flu * self.yscale + self.int_bg

        #return self.x


if __name__=='__main__':
    x=np.linspace(0.006,0.015,100)
    fun=XFNTR_ADV_F(x=x)
    print(fun.y())