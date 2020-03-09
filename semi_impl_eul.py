import numpy as np
import scipy as scp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from io import StringIO
from scipy import optimize as optm
import scipy.integrate
from multiprocessing import Pool
from mpl_toolkits.axes_grid1 import make_axes_locatable

#mass_vector = np.array([1, 100, 1000])

import numpy as np


mass_vector = np.array([1, 10, 1000])

def parameter_calculator_too_complicated(mass_vector):
    X0 = 10 **(-4) #kg**(0.75) m^(-3)
    seconds_per_month = 6 #2419200
    D = 3
    no_specs = 2
    no_specs = no_specs
    a0 = 10**(-1.77) #m^3 s^(-1) kg^(-1.05)
    V0 = 0.33 #ms^(-1) kg^pv)
    pd = 0.2 #Unitless
    d0 = 1.62 #m kg^(2pd)
    pv = 0.25 #Unitless
    Rp = 0.1  #Preferred ratio body-size
    h0 = 0.01 #10 ** 4 #This seems reasonable since the value range is very wide.  # kg**(beta-1) m
    r0 = 10#1.71 * 10**(-6) * seconds_per_month #kg**(1-beta) / month
    L = 1  # kg^(1-beta)s^(-1)
    beta = 0.75 #Metabolic scaling constant
    K0 = 100 #kg^beta m^(-3)
    Cbar0 = 0.01 #Kg biomass pr m^3
    efficiency = 0.7  # Arbitrary
    loss_rate = 4.15 * 10 ** (-2) # kg**(beta) m**(-3) #We just multiplied by 1 million because of reasons We s

    alpha = a0*mass_vector**(pv+2*pd*(D-1))
    search_rate = np.pi * V0 * d0 ** 2 \
                  * (mass_vector[1:]) ** (2/(3)) \
                  * (mass_vector[0:2] ** (2/3))*seconds_per_month

    inner_term = Rp*1/mass_vector[0:2]  # Simplified
    second_term = (1 + (np.log10(inner_term)) ** 2) ** (-0.2)
    outer_term = 1 / (1 + 0.25 * np.exp(-mass_vector[1:] ** (0.33)))

    attack_probability = outer_term * second_term

    encounter_rate = search_rate * attack_probability * 0.3

    # 28*0.2 * (mass_vector[1:])**(beta-1)

    handling_time = h0 * mass_vector ** (-beta) #/seconds_per_month
    ci = 3.5 * loss_rate * (mass_vector[1:] ** (beta)) *seconds_per_month
    maximum_consumption_rate = 7 * loss_rate * (mass_vector[1:] ** (beta)) *seconds_per_month

    nu = maximum_consumption_rate * 1/encounter_rate
    print(nu, maximum_consumption_rate,  r0*mass_vector[0]**(beta), ci, encounter_rate)
    return ci, nu, maximum_consumption_rate, r0*mass_vector[0]**(beta)


mass_vector = np.array([0.01, 1, 100])


def parameter_calculator(mass_vector):
    alpha = 0.55*17.5/12
    b = 330/12
    v = 0.1 #/12
    maximum_consumption_rate = alpha * mass_vector[1:]**(0.75)

    ci = v*maximum_consumption_rate
    #ci[-1] = ci[-1]*0.1
    #print(maximum_consumption_rate)
    r0  = 1
    nu = alpha/b
    #print(ci)
    return ci, nu, maximum_consumption_rate, r0


def heatmap_plotter(data, title, image_name, ext):
    plt.figure()
    plt.title(title)
    #    plt.colorbar(res_nums, fraction=0.046, pad=0.04)
    plt.xlabel("Cbar, g/m^3")
    plt.ylabel("phi0, g/(m^3 * week)")

    ax = plt.gca()
    im = ax.imshow(data, cmap='Reds', extent =ext)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    plt.savefig(image_name+".png", bbox_inches='tight')


def num_derr(f, x, h):
    x_m = float(np.copy(x))
    x_p = float(np.copy(x))
    x_m -= h
    x_p += h
    derr = (f(x_p) - f(x_m))/(2*h)
#    print(derr)
    return derr


def strat_finder(y, params, opt_prey = True, opt_pred = True, taun_old = 1):
    C, N, P = y[0], y[1], y[2]
    taun = 1
    taup = 1
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    if opt_prey is True and opt_pred is True:
        taun = min(max(opt_taun_find(y, params, taun_old), 0), 1)
        taup = min(max(opt_taup_find(y, taun, params), 0), 1)

    elif opt_prey is True and opt_pred is False:

        taun_fitness_II = lambda s_prey: epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * 1 * s_prey * P / (1 * s_prey * N + nu1) - mu0 * s_prey ** 2 - mu1

        p_term = lambda s_prey: (N * s_prey + nu1)

        taun_fitness_II_d = lambda s_prey: epsn * (cmax * nu0) * C / ((s_prey * C + nu0) ** 2) - 2*s_prey*mu0 \
                                           - (cp * nu1) * P * ((s_prey + 1)) / (p_term(s_prey)) ** 2


        linsp = np.linspace(0.0001, 1, 100)
        comparison_numbs = (taun_fitness_II_d(linsp))
        if len(np.where(comparison_numbs > 0)[0]) is 0 or len(np.where(comparison_numbs < 0)[0]) is 0:
            t0 = taun_fitness_II(0)
            t1 = taun_fitness_II(1)
            if t0 > t1:
                max_cands = 0
            else:
                max_cands = 1

        else:
            #print(comparison_numbs[np.where(comparison_numbs < 0)[0]])
            #print(taun_fitness_II_d(comparison_numbs[np.where(comparison_numbs < 0)[0]]))
            maxi_mill = linsp[np.where(comparison_numbs > 0)[0][0]]
            mini_mill = linsp[np.where(comparison_numbs < 0)[0][0]]
            #print(taun_fitness_II_d(0.0001), taun_fitness_II_d(maxi_mill))
            max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[mini_mill, maxi_mill], method='brentq').root

        taun = min(max(max_cands, 0.0001), 1)
    elif opt_pred is True and opt_prey is False:
        taup = min(max(opt_taup_find(y, 1, params),0),1)

    return taun, taup


def opt_taup_find_old_g(y, s_prey, params):
    #print(params)
    k = s_prey * y[1] / params['nu1']
    c = params['cp']/params['nu1']*params['eps'] * s_prey * y[1] / params['phi0']
    x = 1 / 3 * (2 ** (2 / 3) / (
                3 * np.sqrt(3) * np.sqrt(27 * c ** 2 * k ** 8 + 8 * c * k ** 7) + 27 * c * k ** 4 + 4 * k ** 3) ** (1 / 3)
                 + (3 * np.sqrt(3) * np.sqrt(27 * c ** 2 * k ** 8 + 8 * c * k ** 7) + 27 * c * k ** 4 + 4 * k ** 3) ** (
                             1 / 3) / (2 ** (2 / 3) * k ** 2) - 2 / k) #Why was WA not included!?!?
    return x


def opt_taup_find(y,s_prey,params):
    s = 100
    b = 330/12
    k_1 = b*s**(3/4)*params['eps']*s_prey*y[1]/(2*params['phi0']) #k3 is phi0, k1 is b eps taun N s^{3/4} aka c_p
    k_2 = 1/params['nu1']*s_prey*y[1]
#    x = 1 / 3 * ((2 ** (1 / 3) * k_1) / (27 * k_1 ** 2 * k_2 ** 4 + 2 * k_1 ** 3 * k_2 ** 3 + 3 * np.sqrt(3) * np.sqrt(
#        27 * k_1 ** 4 * k_2 ** 8 + 4 * k_1 ** 5 * k_2 ** 7)) ** (1 / 3)
#                 - 2 / k_2 +
#                 (27 * k_1 ** 2 * k_2 ** 4 + 2 * k_1 ** 3 * k_2 ** 3 + 3 * np.sqrt(3) * np.sqrt(
#                         27 * k_1 ** 4 * k_2 ** 8 + 4 * k_1 ** 5 * k_2 ** 7)) ** (1 / 3) /
#                 (2 ** (1 / 3) * k_2 ** 2 * k_1)) #See notebook
    x = 1/3*((27 * k_1 * k_2 ** 4 + 2 * k_2 ** 3 + 3 * np.sqrt(3) * np.sqrt(
        27 * k_1 ** 2 * k_2 ** 8 + 4 * k_1 * k_2 ** 7)) ** (1 / 3) / (2 ** (1 / 3) * k_2 ** 2) - 2 / k_2 + 2 ** (
                          1 / 3) / (27 * k_1 * k_2 ** 4 + 2 * k_2 ** 3 + 3 * np.sqrt(3) * np.sqrt(
        27 * k_1 ** 2 * k_2 ** 8 + 4 * k_1 * k_2 ** 7)) ** (1 / 3))

#    if max(x.shape) > 1: Add this back again
#        x[x>1] = 1
#    else:
#        if x[0] > 1:
#            x[0] = 1
    #print(k_1, params['phi0'], "terms in opt_taup")
    return x

def opt_taun_find(y, params, dummy):
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    taun_fitness_II = lambda s_prey: epsn * cmax * s_prey * C / (s_prey * C + nu0) - cp * taup(s_prey) * s_prey * P / (
                    taup(s_prey) * s_prey * N + nu1) - mu0 * s_prey**2 - mu1
    p_term = lambda s_prey : (N*s_prey*taup(s_prey)+nu1)

    taup = lambda s_prey: opt_taup_find(y, s_prey, params) #cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey)) -cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey))

    taup_prime = lambda s_prey: (opt_taup_find(y, s_prey+0.00001, params)-opt_taup_find(y, s_prey-0.00001, params))/(2*0.00001) #cp*(1/(N*s_prey**2) - 1/2*np.sqrt(eps/(phi0*N))*s_prey**(-3/2))

    taun_fitness_II_d = lambda s_prey: epsn*(cmax*nu0)*C/((s_prey*C+nu0)**2) - 2*s_prey*mu0 \
                                            - (nu1*cp)*P*((s_prey*taup_prime(s_prey)+taup(s_prey))/(p_term(s_prey))**2)

    linsp = np.linspace(0.0001 , 1, 100)
    #print(
    comparison_numbs = taun_fitness_II_d(linsp)
    #print(comparison_numbs)
    if len(np.where(comparison_numbs > 0)[0]) is 0 or len(np.where(comparison_numbs < 0)[0]) is 0:
        t0 = taun_fitness_II(0.0001)
        t1 = taun_fitness_II(1)
        if t0 > t1:
            max_cands = 0.0001
        else:
            max_cands = 1

    else:
        maxi_mill = linsp[np.where(comparison_numbs > 0)[0][-1]]
        mini_mill = linsp[np.where(comparison_numbs < 0)[0][-1]]
        max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[mini_mill, maxi_mill], method='brentq').root

   # print(num_derr(taun_fitness_II, max_cands, 0.0001), taun_fitness_II_d(max_cands), max_cands)
    print(np.max(taun_fitness_II(linsp)), taun_fitness_II(max_cands))
    return max_cands #max_cands_two[np.argmax(taun_fitness_II(max_cands_two))]


def optimal_behavior_trajectories(t, y, params, seasons = False, taun=1, taup=1):
    C = y[0]
    N = y[1]
    P = y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()
    if seasons is True:
        Cdot = lam*(cbar+0.5*cbar*np.cos(t*np.pi/6) - C) - cmax*N*taun*C/(taun*C+nu0)  # t is one month
    else:
        Cdot = lam*(cbar - C) - cmax*N*taun*C/(taun*C+nu0)
    flux_c_to_n = N*taun*C/(taun*C+nu0)
    flux_n_to_p = N*taup * taun*P*cp*1/(taup*taun*N + nu1)  # Now these values are calculated twice..
    Ndot = N*(epsn*cmax*taun*C/(taun*C+nu0) - taup * taun*P*cp*1/(taup*taun*N + nu1) - mu0*taun**2 - mu1)
    Pdot = P*(cp*eps*taup*taun*N/(N*taup*taun + nu1) - phi0*taup**2 - phi1)
    return np.array([Cdot, Ndot, Pdot, flux_c_to_n, flux_n_to_p])


def semi_implicit_euler(t_final, y0, step_size, f, params, opt_prey = True, opt_pred = True):
    solution = np.zeros((y0.shape[0], int(t_final / step_size)))
    flux = np.zeros((2,int(t_final / step_size)))
    strat = np.zeros((2,int(t_final / step_size)))
    t = np.zeros(int(t_final / step_size))
    solution[:, 0] = y0
    strat[0,0] = 0.5
    for i in range(1, int(t_final / step_size)):
        taun_best, taup_best = strat_finder(solution[:,i-1], params, opt_prey = opt_prey, opt_pred = opt_pred, taun_old = strat[0,i-1])
        strat[:, i] = taun_best, taup_best
        t[i] = t[i-1] + step_size
        fwd = f(t[i], solution[:, i - 1], taun_best, taup_best)
        solution[:, i] = solution[:, i-1] + step_size*fwd[0:3]
        flux[:, i] = fwd[3:]
    return t, solution, flux, strat



#base = 10
its = 0
step_size = 0.5*2.5
step_size_phi = 0.0025*2.5 #0.00125
#phi0_base = 0.2

#cmax = 2
#mu0 = 0.2
#mu1 = 0.2
#eps = 0.7
#epsn = 0.7
#cp = 2 m,i
#phi0 = 0.4 #phi0_base
#phi1 = 0.2
#lam = 0.5

cost_of_living, nu, growth_max, lam = parameter_calculator(mass_vector)

base = 50 #*mass_vector[0]**(-0.25) #0.01
cbar = base
phi0_base = cost_of_living[1]/2
phi1 = cost_of_living[1]/2
phi0 = phi0_base

cmax, cp = growth_max
mu0 = cost_of_living[0]*2 #*60 #/2
mu1 = cost_of_living[0]*2 #*10 #/2
nu0 = nu #nu
nu1 = nu #nu

eps = 0.7
epsn = 0.7
#phi0 = 0.4 #phi0_base

opt_prey = True
opt_pred = True
#Currently the affinity is set to 1!!

params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
          'resource': base, 'lam':lam, 'nu0':nu0, 'nu1': nu1}

#params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
#          'resource': base, 'lam':lam}


def dynamic_pred_prey(phi0_dyn, step_size=step_size, its=its, params=params_ext):
#    solution_storer = np.zeros
    flux_and_strat_storer = []
    params['phi0'] = phi0_dyn

    t_end = 10

    init = np.array([0.8, 0.5, 0.5])
    time_b, sol_basic, flux_bas, strat_bas = semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
            optimal_behavior_trajectories(t, y, params, taun=tn, taup=tp), params, opt_prey = False, opt_pred = False)
    base_case = np.array([sol_basic[0, -1], sol_basic[1, -1], sol_basic[2, -1]])


    tim, sol, flux, strats = semi_implicit_euler(t_end, base_case, 0.001, lambda t, y, tn, tp:
    optimal_behavior_trajectories(t, y, params, taun=tn, taup=tp), params)

    strats = np.zeros((its, 2))
    fluxes = np.zeros((its, 2))
    pops = np.zeros((its, 3))
    t_end = 5

    for i in range(0, its):
        params['resource'] = base+step_size*i
        init = np.array([sol[0,-1], sol[1,-1], sol[2,-1]])
        tim, sol, flux, strat = semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
            optimal_behavior_trajectories(t, y, params, taun=tn, taup=tp), params_ext, opt_prey=True, opt_pred=True)



        tim_OG, sol_OG, flux_OG, strat_OG = semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
            optimal_behavior_trajectories(t, y, params, taun=tn, taup=tp), params_ext, opt_prey=False, opt_pred=False)
        pops[i] = sol[:,-1] #np.sum((sol-sol_OG)*0.001, axis = 1) #sol[:,-1] - sol_OG[:,-1]
        strats[i] = np.maximum(strat[:, -1], strat[:, -2])
#        if strats[i, 0] is 0 or strats[i, 1] is 0:
#            print(strat)
        fluxes[i] = np.sum((flux-flux_OG)*0.001, axis = 1)
        #print(fluxes[i], pops[i], phi0_dyn, base+step_size*i)
        print(i)
    return np.hstack([strats, pops, fluxes])



if its > 0:
    data_init = list(np.linspace(phi0_base, phi0_base + its * step_size_phi, its))
    agents = 8
    with Pool(processes = agents) as pool:
        results = pool.map(dynamic_pred_prey, data_init, 1)

    print(np.array(results).shape)
    #st = dynamic_pred_prey(phi0_base, step_size=step_size, its=its, params=params_ext)
    #print(st)

    results = np.array(results)

    ran = [base, base+its*step_size, 100*phi0_base, 100*(phi0_base+its*step_size_phi)]

    heatmap_plotter(results[:, :, 3], 'Prey g/m^3', "prey_conc", ran)
    heatmap_plotter(results[:, :, 4], 'Predators g/m^3', "pred_conc", ran)
    heatmap_plotter(results[:, :, 0], 'Prey foraging intensity', "prey_for", ran)
    heatmap_plotter(results[:, :, 1], 'Predator foraging intensity', "pred_for", ran)


    plt.imshow(results[:, :, 0], cmap='viridis')
    plt.title('Prey strategy')
    plt.colorbar()
    plt.show()

    plt.imshow(results[:, :, 1], cmap='viridis')
    plt.title('Predator strategy')
    plt.colorbar()
    plt.show()

    plt.imshow(results[:, :, 3], cmap='viridis')
    plt.title('Prey pop')
    plt.colorbar()
    plt.show()

    plt.imshow(results[:, :, 4], cmap='viridis')
    plt.title('Predator pop')
    plt.colorbar()
    plt.show()

    plt.imshow(results[:, :, 5], cmap='viridis')
    plt.title('Flux, 0 to 1')
    plt.colorbar()
    plt.show()

    plt.imshow(results[:, :, 6], cmap='viridis')
    plt.title('Flux, 1 to 2')
    plt.colorbar()
    plt.show()


print(mu0, mu1, cmax, nu0, phi0, phi1, cp, nu1)
t_end = 15


init = np.array([base, base, base]) #np.array([5.753812957581866, 5.490194692112937, 1.626801718856221])#
tim, sol, flux, strat = semi_implicit_euler(t_end, init, 0.0001, lambda t, y, tn, tp:
optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=True, opt_pred=True)
tim, sol_2, flux_2, strat_2 = semi_implicit_euler(t_end, init, 0.0001, lambda t, y, tn, tp:
optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=False, opt_pred=False)
tim, sol_3, flux_3, strat_3 = semi_implicit_euler(t_end, init, 0.0001, lambda t, y, tn, tp:
optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=True, opt_pred=False)
tim, sol_4, flux_4, strat_4 = semi_implicit_euler(t_end, init, 0.0001, lambda t, y, tn, tp:
optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=False, opt_pred=True)

print(sol_2)
C, N, P =  C, N, P = sol[:,-1]

print(C, N, P, "CNP1", strat_finder(sol[:,-1], params_ext))
print(sol_3[:,-1], sol_4[:,-1], "Other optimal combinations, population levels")

y = np.array([C, N, P])
numbs = np.linspace(0,1,500)
taun_fitness_II = lambda s_prey: \
    epsn * cmax * s_prey * C / (s_prey * C + nu0) - cp * opt_taup_find(y, s_prey, params_ext) * s_prey * P / (
                opt_taup_find(y, s_prey, params_ext) * s_prey * N + nu1) - mu0 * s_prey**2 - mu1


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)


plt.scatter(numbs, (taun_fitness_II(numbs)))
plt.show()

C, N, P = sol[:,-2]
print(C, N, P, "CNP2", strat_finder(sol[:,-2], params_ext))
y = np.array([C, N, P])
numbs = np.linspace(0,1,500)
taun_fitness_II = lambda s_prey: \
    epsn * cmax * s_prey * C / (s_prey * C + nu0) - cp * opt_taup_find(y, s_prey, params_ext) * s_prey * P / (
                opt_taup_find(y, s_prey, params_ext) * s_prey * N + nu1) - mu0 * s_prey**2 - mu1

plt.scatter(numbs, (taun_fitness_II(numbs)))
plt.show()

plt.scatter(numbs, opt_taup_find(y, numbs, params_ext))
print(opt_taup_find(y, numbs, params_ext), "taup", params_ext)
plt.show()

plt.figure()
#plt.plot(tim, sol[0,:], label = 'Resource biomass')
plt.plot(tim, sol[1,:], label = 'Dynamic prey biomass', color = 'Green')
plt.plot(tim, sol[2,:], label = 'Dynamic predator biomass', color = 'Red')
plt.plot(tim, sol_2[1,:], label = 'Static prey biomass', linestyle = '-.', color = 'Green')
plt.plot(tim, sol_2[2,:], label = 'Static predator biomass', linestyle = '-.', color = 'Red')

plt.xlabel("Months")
plt.ylabel("g/m^3")
plt.legend(loc = 'lower left')

plt.savefig("Indsvingning.png")

plt.figure()


#locs = np.where(strat[0, 1:] < 0.15)[0]
#locs = np.where(strat[0, 1:] < 0.3)[0]
#strat[0, locs] = 0.7902903906643381
#strat[1, locs] = 0.6747319288363259 #due to numerical fluctuations... The method is of too low an order and the new way to find max of taun is not stable.


plt.plot(tim[1:], strat[0,1:], 'x', label = "Prey foraging intensity",alpha=0.5)
plt.plot(tim[1:], strat[1,1:], 'x', label = "Predator foraging intensity", alpha=0.5)
plt.xlabel("Months")
plt.ylabel("Intensity")
plt.legend(loc = 'center left')
plt.savefig("Indsvingning_strat.png")

print(strat[0,1:])
