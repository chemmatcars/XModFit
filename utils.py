import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import inv
from functools import lru_cache
from Chemical_Formula import Chemical_Formula

cf=Chemical_Formula(formula='Au')

def calc_pr(q,Iq):
    """
    Calculates Moore's autocorrelation function
    """
    dq=np.mean(np.diff(q))
    rmax=2*np.pi/q[0]
    rmin=2*np.pi/q[-1]
    pr=[]
    for r in np.arange(0,rmax,1.0):
        pr.append([r,r*np.sum(q*Iq*np.sin(q*r))*dq/2/np.pi**2])
    return array(pr)

def Bint(n,s,d):
    """
    Calculates sine integrals
    """
    return np.pi*n*d*(-1)**(n+1)*np.sin(2*np.pi*d*s)/((np.pi*n)**2-(2*np.pi*d*s)**2)

def calc_prm(q1,Iq1,dIq1=None,qmin=None,qmax=None,Nq=None,Nr=101,dmax=100.0):
    """
    Calculates autocorrelation by Moore's method (J Appl. Cryst. 13, 168 (1980))
    """
    f1=interp1d(q1,Iq1)
    f2=interp1d(q1,dIq1)
    if qmin is None:
        qmin=q1[0]
    if qmax is None:
        qmax=q1[-1]
    if Nq is None:
        Nq=len(q1)
    q=np.linspace(qmin,qmax,Nq)
    Iq=f1(q)
    dIq=f2(q)
    s=q/2.0/np.pi
    U=s*Iq
    nmin=int(2*s[0]*dmax)+1
    nmax=int(2*s[-1]*dmax)+1
    if dIq is None:
        dIq=np.ones_like(Iq)
    #Calculation of Matrix of product of sine Integrals
    Cmat=np.array([[np.sum(Bint(n,s,dmax)*Bint(m,s,dmax)/s**2/dIq**2) for n in range(nmin,nmax+1)] 
                   for m in range(nmin,nmax+1)])
    ymat=np.array([np.sum(Bint(n,s,dmax)*U/s**2/dIq**2) for n in range(nmin,nmax+1)])
    InvCmat=inv(Cmat)
    a=np.dot(ymat,InvCmat)/4.0
    r=np.linspace(0.001,dmax,Nr)
    pr=8*np.pi*r*np.array([np.sum(a*np.sin(np.pi*ri*np.array(range(nmin,nmax+1))/dmax)) for ri in r])
    N,M=np.meshgrid(np.arange(1,InvCmat.shape[0]+1),np.arange(1,InvCmat.shape[0]+1))
    dpr=2*np.pi*r*np.sqrt(np.array([np.sum(np.sin(np.pi*N*ri/dmax)*np.sin(np.pi*M*ri/dmax)*InvCmat) for ri in r]))
    Iqc=4*np.array([np.sum(a*Bint(np.array(range(nmin,nmax+1)),s1,dmax)) for s1 in s])/s
    #chi=np.sum((Iq-Iqc)**2/dIq**2)
    #chi_r=chi/(len(q)-len(a))
    return r,pr,dpr,q,Iqc

def find_minmax(fun,pos=1.0,wid=1.0,accuracy=1e-6):
    """
    Find the minimum and maximum values of x for which a peak like function 'fun(x,pos,wid)' has a relatve value compare to the peak value more than the 'accuracy'.
    fun           : Peak like function class 
    pos           : Peak position
    wid           : width of the peak
    """
    xmin=1e-10
    xmax=pos+5*wid
    N=int(1.0/accuracy)
    fun.x=np.linspace(xmin,xmax,N)
    fun.pos=pos
    fun.wid=wid
    fval=fun.y()
    frange=np.where(fval/np.max(fval/100)>accuracy)[0]
    while len(frange)>1 and np.abs(fun.x[frange[len(frange)-1]]-fun.x[-1])<1e-6:
        xmax=xmax*3
        fun.x=np.linspace(xmin,xmax,N*3)        
        fval=fun.y()
        frange=np.where(fval/np.max(fval)/100>accuracy)[0]
    return fun.x[frange[0]],fun.x[frange[-1]]


@lru_cache(maxsize=10)
def calc_rho(R=(1.0, 0.0), material=('Au', 'H2O'), relement='Au', density=(19.3, 1.0), sol_density=(1.0, 1.0),
             Rmoles=(1.0, 0.0), Energy=None, NrDep='True'):
    """
    Calculates the complex electron density of core-shell type multilayered particles in el/Angstroms^3

    R           :: list of Radii and subsequent shell thicknesses in Angstroms of the nanoparticle system
    material    :: list of material of all the shells starting from the core to outside
    relement    :: Resonant element
    density     :: list of density of all the materials in gm/cm^3 starting from the inner core to outside
    Rmoles      :: mole-fraction of the resonant element in the materials
    sol_density :: density of solvent in which the particle layers are dissolved
    NrDep       :: True or False for using or not using energy dependent scattering factors for non-resonant elements
    Energy      :: Energy in keV
    """
    density = list(density)
    if len(material) == len(density):
        Nl = len(material)
        rho = []
        adensity = []  # Density of anomalous element
        eirho = []  # Energy independent electron density
        r = 0.0
        rhor = []
        eirhor = []
        adensityr = []
        for i in range(Nl):
            mat = material[i].split(':')
            if len(mat) == 2:
                solute, solvent = mat
                element_adjust = None
                if '*' in solute:
                    m = solute.split('*')[0]
                    f = cf.parse(m)
                    element_adjust = cf.elements()[-1]
                solute_formula = cf.parse(solute)
                fac=1.0
                if relement in solute_formula.keys():
                    cf.formula_dict[relement] = 0.0
                    t1 = cf.molar_mass()
                    if element_adjust is not None:
                        cf.formula_dict[element_adjust] = cf.element_mole_ratio()[element_adjust] - Rmoles[i]
                    cf.formula_dict[relement] = Rmoles[i]
                    t2 = cf.molar_mass()
                    if t1 > 0:
                        fac = t2 / t1
                density[i] = fac * density[i]
                solute_elements = cf.elements()
                solute_mw = cf.molecular_weight()
                solute_mv = cf.molar_volume()
                solute_mole_ratio = cf.element_mole_ratio()

                solvent_formula = cf.parse(solvent)
                solvent_elements = cf.elements()
                solvent_mw = cf.molecular_weight()
                solvent_mole_ratio = cf.element_mole_ratio()

                solvent_moles = sol_density[i] * (1 - solute_mv * density[i] / solute_mw) / solvent_mw
                solute_moles = density[i] / solute_mw
                total_moles = solvent_moles + solute_moles
                solvent_mole_fraction = solvent_moles / total_moles
                solute_mole_fraction = solute_moles / total_moles
                comb_material = ''
                for ele in solute_mole_ratio.keys():
                    comb_material += '%s%.10f' % (ele, solute_mole_ratio[ele] * solute_mole_fraction)
                for ele in solvent_mole_ratio.keys():
                    comb_material += '%s%.10f' % (ele, solvent_mole_ratio[ele] * solvent_mole_fraction)
                density[i] = density[i] + sol_density[i] * (1 - solute_mv * density[i] / solute_mw)
                # self.output_params['scaler_parameters']['density[%s]' % material[i]]=tdensity
            else:
                element_adjust = None
                if '*' in material[i]:
                    m = material[i].split('*')[0]
                    f = cf.parse(m)
                    element_adjust = cf.elements()[-1]
                formula = cf.parse(material[i])
                fac = 1.0
                if relement in formula.keys():
                    cf.formula_dict[relement] = 0.0
                    t1 = cf.molar_mass()
                    if element_adjust is not None:
                        cf.formula_dict[element_adjust] = cf.element_mole_ratio()[
                                                                       element_adjust] - Rmoles[i]
                    cf.formula_dict[relement] = Rmoles[i]
                    t2 = cf.molar_mass()
                    if t1 > 0:
                        fac = t2 / t1
                mole_ratio = cf.element_mole_ratio()
                comb_material = ''
                for ele in mole_ratio.keys():
                    comb_material += '%s%.10f' % (ele, mole_ratio[ele])
                density[i] = fac * density[i]
            tdensity = density[i]
            formula = cf.parse(comb_material)
            molwt = cf.molecular_weight()
            elements = cf.elements()
            mole_ratio = cf.element_mole_ratio()
            # numbers=np.array(chemical_formula.get_element_numbers(material[i]))
            moles = [mole_ratio[ele] for ele in elements]
            nelectrons = 0.0
            felectrons = complex(0.0, 0.0)
            aden = 0.0
            for j in range(len(elements)):
                f0 = cf.xdb.f0(elements[j], 0.0)[0]
                nelectrons = nelectrons + moles[j] * f0
                if Energy is not None:
                    if elements[j] != relement:
                        if NrDep:
                            f1 = cf.xdb.f1_chantler(element=elements[j], energy=Energy * 1e3, smoothing=0)
                            f2 = cf.xdb.f2_chantler(element=elements[j], energy=Energy * 1e3, smoothing=0)
                            felectrons = felectrons + moles[j] * complex(f1, f2)
                    else:
                        f1 = cf.xdb.f1_chantler(element=elements[j], energy=Energy * 1e3, smoothing=0)
                        f2 = cf.xdb.f2_chantler(element=elements[j], energy=Energy * 1e3, smoothing=0)
                        felectrons = felectrons + moles[j] * complex(f1, f2)
                if elements[j] == relement:
                    aden += 0.6023 * moles[j] * tdensity / molwt
            adensity.append(aden)
            eirho.append(0.6023 * (nelectrons) * tdensity / molwt)
            rho.append(0.6023 * (nelectrons + felectrons) * tdensity / molwt)
            rhor.append([r, np.real(rho[-1])])
            eirhor.append([r, np.real(eirho[-1])])
            adensityr.append([r, np.real(adensity[-1])])
            r = r + R[i]
            rhor.append([r, np.real(rho[-1])])
            eirhor.append([r, np.real(eirho[-1])])
            adensityr.append([r, np.real(adensity[-1])])
        rhor, eirhor, adensityr = np.array(rhor), np.array(eirhor), np.array(adensityr)
        rhor[-1, 0] = rhor[-1, 0] + R[-2]
        eirhor[-1, 0] = eirhor[-1, 0] + R[-2]
        adensityr[-1, 0] = adensityr[-1, 0] + R[-2]
        return np.array(rho), np.array(eirho), np.array(adensity), np.array(rhor), np.array(eirhor), np.array(adensityr)

def create_steps(x=[1],y=[1]):
    res=np.array([[0.0,0.0]])
    r1=res[0,0]
    for i,r in enumerate(x):
        r2=r1+r
        res=np.vstack((res,np.array([[r1,y[i]],[r2,y[i]]])))
        r1=r2
    res=np.vstack((res,np.array([[r1,0],[r1+x[-1],0]])))
    return res[:,0],res[:,1]
