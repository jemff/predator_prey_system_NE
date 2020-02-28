import numpy as np


mass_vector = np.array([0.01, 1, 10])

def parameter_calculator(mass_vector):
    X0 = 10 **(-4) #kg**(0.75) m^(-3)
    seconds_per_month = 2419200
    D = 3
    no_specs = 2
    no_specs = no_specs
    a0 = 10**(-1.77) #m^3 s^(-1) kg^(-1.05)
    V0 = 0.33 #ms^(-1) kg^pv)
    pd = 0.2 #Unitless
    d0 = 1.62 #m kg^(2pd)
    pv = 0.25 #Unitless
    Rp = 0.1  #Preferred ratio body-size
    h0 = 3 #10 ** 4 #This seems reasonable since the value range is very wide.  # kg**(beta-1) s
    r0 = 1.71 * 10**(-6) * seconds_per_month #kg**(1-beta) / month
    L = 1  # kg^(1-beta)s^(-1)
    beta = 0.75 #Metabolic scaling constant
    K0 = 100 #kg^beta m^(-3)
    Cbar0 = 0.01 #Kg biomass pr m^3
    efficiency = 0.7  # Arbitrary
    loss_rate = 4.15 * 10 ** (-8) # kg**(beta) m**(-3)

    alpha = a0*mass_vector**(pv+2*pd*(D-1))
    search_rate = np.pi * V0 * d0 ** 2 \
                  * (mass_vector[1:]) ** (pv + 2 * pd* (D - 1)) \
                  * (mass_vector[0:2] ** (2 * pd))

    inner_term = Rp*mass_vector[1:]/mass_vector[0:2]  # Simplified
    second_term = (1 + (np.log10(inner_term)) ** 2) ** (-0.2)
    outer_term = 1 / (1 + 0.25 * np.exp(-mass_vector[1:] ** (0.33)))

    attack_probability = outer_term * second_term

    encounter_rate = search_rate * attack_probability
    handling_time = h0 * mass_vector ** (-beta)
    ci = loss_rate * mass_vector ** (beta - 1)

    nu = 1/handling_time[1:] * 1/encounter_rate

    return ci[1:]*seconds_per_month, nu, 1/handling_time[1:], r0

cost_of_living, nu, growth_max, lam = parameter_calculator(mass_vector)


base = 0.01
cbar = base
phi0_base = cost_of_living[1]/2
phi1 = cost_of_living[1]/2

cmax, cp = growth_max
mu0 = cost_of_living[0]/2
mu1 = cost_of_living[0]/2
nu0, nu1 = nu

eps = 0.7
epsn = 0.7
phi0 = phi0_base
