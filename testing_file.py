from common_functions import *

import numpy as np

def static_eq_calc(params):
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    phitild = phi0+phi1
    mutild = mu0 + mu1
    C_star = phitild*nu1/(eps*cp-phitild)
    gam = nu0-cbar+(cmax/lam)*C_star
    print(gam, gam**2, 4*cbar*nu0, np.sqrt(gam**2+4*cbar*nu0))
    R_star = (-gam + np.sqrt(gam**2+4*cbar*nu0))/2
    P_star = (epsn * C_star*R_star*cmax/(R_star+nu0)-mutild*C_star)/(cp*C_star/(C_star+nu1))

    print(cp*C_star/(C_star+nu1), epsn * C_star*R_star*cmax/(R_star+nu0))
    return np.array([R_star, C_star, P_star])


mass_vector = np.array([1, 1, 100])

def parameter_calculator_mass(mass_vector):
    alpha = 15
    b = 330/12
    v = 0.2 #/12
    maximum_consumption_rate = alpha * mass_vector[1:]**(0.75)

    ci = v*maximum_consumption_rate
    #ci[-1] = ci[-1]*0.1
    #print(maximum_consumption_rate)
    r0  = 0.1
    nu = alpha/b*mass_vector[1:]**(0)
    #print(ci)
    return ci, nu, maximum_consumption_rate, r0


cost_of_living, nu, growth_max, lam = parameter_calculator_mass(mass_vector)
print(nu, "nu", growth_max)
base = 10 #3 #*mass_vector[0]**(-0.25) #0.01
phi0= cost_of_living[1]/2 #/2#*5
phi1 = cost_of_living[1]/2 #*2#/2#*5
eps = 0.7
epsn = 0.7

cmax, cp = growth_max
mu0 = 0 # cost_of_living[0] #/3 #*2#/2#*5 #*60 #/2
mu1 = cost_of_living[0]#/3 #*2 #*2#/2#*5 #*2 #*10 #/2
nu0 = nu[0] #nu
nu1 = nu[1] #nu

params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
          'resource': base, 'lam':lam, 'nu0':nu0, 'nu1': nu1}

#print(nash_eq_find(static_eq_calc(params_ext), params_ext))

value = np.array([0.79, 0.5, 0.07])
def f(x):
    return opt_taun_analytical(value, opt_taup_find(value, x, params_ext), 10, 0.7, 0.54545454)[0] - x

NE = optm.root(f, x0 = 0.5)
print(NE.x, opt_taup_find(value, NE.x, params_ext)[0])
#def instantaneous_intertrophic_flux(state, params):

