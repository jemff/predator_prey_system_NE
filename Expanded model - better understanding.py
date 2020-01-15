#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as scp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from io import StringIO
from scipy import optimize as optm
import scipy.integrate
from multiprocessing import Pool


taup = lambda s_prey, N : cp*(np.sqrt(eps/(phi0*s_prey*N)) - 1/(N*s_prey))


#prey_strats = np.linspace(0.01,1,198)

#prey_numbers = np.linspace(0.1,30,198)

#X, Y = np.meshgrid(prey_strats, prey_numbers)

#Z = taup(X, Y)

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 50)

taun_fitness_II = lambda s_prey, C, P, N, : \
    epsn*cmax*s_prey*C/(s_prey*C+cmax) - cp * taup(s_prey, N) * s_prey*P/(taup(s_prey, N)*s_prey*N + cp) \
    - mu0*s_prey - mu1


bnds = (0, 1)

def optimal_behavior_trajectories(t, y, cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, opt_prey = True, opt_pred=True, seasons = False, taun_old = 0.5):
    C = y[0]
    N = y[1]
    P = y[2]
    taun = 1
    taup_opt = 1

    bnds = (0, 1)

    if opt_prey is True and opt_pred is True:
        taun = min(max(optm.minimize(lambda test: -taun_fitness_II(test, C, P, N), np.array([taun_old]), method = 'L-BFGS-B',
                                     bounds = [bnds]).x[0], 0.00001), 1)
        taup_opt = min(max(taup(taun, N),0),1)
    elif opt_prey is True and opt_pred is False:
        #print(optm.minimize(lambda s_prey: -(2*s_prey*C/(s_prey*C+cmax) - taup_opt * s_prey*P/(taup_opt*s_prey*N + cp) - mu0*s_prey - mu1), 0.5).x[0])
        taun = min(max(optm.minimize(lambda s_prey: -(cmax*epsn*s_prey*C/(s_prey*C+cmax) - cp*taup_opt * s_prey*P/(taup_opt*s_prey*N + cp) - mu0*s_prey - mu1), 0.5).x[0],0),1)
    elif opt_pred is True and opt_prey is False:
        taup_opt = min(max(taup(taun, N),0),1)
    if seasons is True:
        Cdot = lam*(cbar+0.5*cbar*np.cos(t*np.pi/180) - C) - N*taun*C/(taun*C+cmax) #t is one month
    else:
        Cdot = lam*(cbar - C) - cmax*N*taun*C/(taun*C+cmax)
    flux_c_to_n = N*taun*C/(taun*C+cmax)
    flux_n_to_p = N*taup_opt * taun*P*cp*1/(taup_opt*taun*N + cp) #Now these values are calculated twice..
    Ndot = N*(epsn*cmax*taun*C/(taun*C+cmax) - taup_opt * taun*P*cp*1/(taup_opt*taun*N + cp) - mu0*taun - mu1)
    Pdot = P*(cp*eps*taup_opt*taun*N/(N*taup_opt*taun + cp) - phi0*taup_opt - phi1)
    return np.array([Cdot, Ndot, Pdot, taun, taup_opt, flux_c_to_n, flux_n_to_p])




def opt_taun_find(y, params):
    C, N, P = y[0], y[1], y[2]


    taun_fitness_II = lambda s_prey: \
        epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * taup(s_prey, N) * s_prey * P / (
                    taup(s_prey, N) * s_prey * N + cp) \
        - mu0 * s_prey - mu1
    taun_fitness_II_d = lambda s_prey: epsn*cmax**2*C/(s_prey*C+cmax)**2 \
                                       - (N**2*cp*(eps/(phi0*N))**(1/2)*s_prey**(5/2))**(-1) -mu0

    max_cands = optm.find_roots(taun_fitness_II_d).root
    loc = np.where(taun_fitness_II(max_cands) == np.max(taun_fitness_II(max_cands)))[0]

    return max_cands[loc]


def strat_finder(y, params, opt_prey = True, opt_pred = True):
    C, N, P = y[0], y[1], y[2]
    if opt_prey is True and opt_pred is True:
        roots = optm.find_roots()
        taun_fitness_II(roots)
        taun = min(max(opt_taun_find(y, params)), 1)
        taup_opt = min(max(taup(taun, N),0),1)


    elif opt_prey is True and opt_pred is False:
        taun = min(max(optm.minimize(lambda s_prey: -(cmax*epsn*s_prey*C/(s_prey*C+cmax) - cp*taup_opt * s_prey*P/(taup_opt*s_prey*N + cp) - mu0*s_prey - mu1), 0.5).x[0],0),1)
    elif opt_pred is True and opt_prey is False:
        taup_opt = min(max(taup(taun, N),0),1)

    return taun, taup_opt

def optimal_behavior_trajectories_basic(t, y, cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, prey_opt, pred_opt, seasons = False, taun_old = 0.5):
    C = y[0]
    N = y[1]
    P = y[2]

    taun = 1
    taup_opt = 1
    if seasons is True:
        Cdot = lam*(cbar+0.5*cbar*np.cos(t*np.pi/180) - C) - 2*N*taun*C/(taun*C+cmax)*10 #t is one month
    else:
        Cdot = lam*(cbar - C) - cmax*N*taun*C/(taun*C+cmax)
    flux_c_to_n = N*taun*C/(taun*C+cmax)
    flux_n_to_p = N*taup_opt * taun*P*cp*1/(taup_opt*taun*N + cp) #Now these values are calculated twice..
    Ndot = N*(epsn*cmax*taun*C/(taun*C+cmax) - taup_opt * taun*P*cp*1/(taup_opt*taun*N + cp) - mu0*taun - mu1)
    Pdot = P*(cp*eps*taup_opt*taun*N/(N*taup_opt*taun + cp) - phi0*taup_opt - phi1)
    return np.array([Cdot, Ndot, Pdot, taun, taup_opt, flux_c_to_n, flux_n_to_p])


def runge_kutta_step_3_order_autonomous(y, step_size, f, taun_old):
    k1 = step_size * f(y, taun_old)[0:3]
    k2 = step_size * f(y + 1 / 3 * k1, taun_old)[0:3]
    k3 = step_size * f(y + 2 / 3 * k2, taun_old)[0:3]

    return y + 1 / 4 * (k1 + 3 * k3), f(y, taun_old)[3:]


def rk23_w_flux(t_final, y0, step_size, f):
    solution = np.zeros((y0.shape[0], int(t_final / step_size)))
    flux_and_strat = np.zeros((4,int(t_final / step_size)))
    flux_and_strat[0,0] = 0.5 #initial guess for strategy
    t = np.zeros(int(t_final / step_size))
    solution[:, 0] = y0
    for i in range(1, int(t_final / step_size)):
        taun_old = flux_and_strat[0,i-1]
        solution[:, i], flux_and_strat[:,i] = runge_kutta_step_3_order_autonomous(solution[:, i - 1],
                                                                                  step_size, f, taun_old)
        # print(runge_kutta_step_4_order_autonomous(solution[:,i-1], step_size, f))
        t[i] = t[i - 1] + step_size

    return t, solution, flux_and_strat


def semi_implicit_euler(t, y, step_size, f):
    solution = np.zeros((y0.shape[0], int(t_final / step_size)))
    flux = np.zeros((2,int(t_final / step_size)))
    strat = np.zeros((2,int(t_final / step_size)))
    t = np.zeros(int(t_final / step_size))
    solution[:, 0] = y0
    for i in range(1, int(t_final / step_size)):
        taun_best, taup_best = strat_finder(solution[:,i-1])
        solution[:, i] = solution[:,i-1] + step_size*f(solution[:, i - 1], taun_best, taup_best)

    return t, solution, flux, strat

base = 5
its = 640
step_size = 0.15
step_size_phi = 0.05
cbar = base
phi0_base = 0.1

cmax = 3
mu0 = 0.3
mu1 = 0.3
eps = 0.5
epsn = 0.5
cp = 3
phi0 = phi0_base
phi1 = 0.2
lam = 0.5

opt_prey = True
opt_pred = True
#Currently the affinity is set to 1!!

params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps' : eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
          'resource': base, 'lam':lam, 'opt_prey':opt_prey, 'opt_pred': opt_pred}

#initial_conditions_5 = [cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, opt_prey, opt_pred]

#t_start = 0
#t_end = 60

#init = np.array([0.8, 0.5, 0.5])
#time_b, sol_basic, flux_and_strat_bas = rk23_w_flux(t_end, init, 0.01, lambda y:
#optimal_behavior_trajectories_basic(0, y, *initial_conditions_5))

#init_0 = np.array([sol_basic[0, -1], sol_basic[1, -1], sol_basic[2, -1]])


def dynamic_pred_prey(phi0_dyn, step_size=step_size, its=its, params=params_ext):
#    solution_storer = np.zeros
    flux_and_strat_storer = []
    params['phi0'] = phi0_dyn

    t_end = 100

    init = np.array([0.8, 0.5, 0.5])
    time_b, sol_basic, flux_and_strat_bas = rk23_w_flux(t_end, init, 0.01, lambda y, tno:
    optimal_behavior_trajectories_basic(0, y, *list(params.values()), taun_old = tno))

    base_case = np.array([sol_basic[0, -1], sol_basic[1, -1], sol_basic[2, -1]])

    strats_fluxes = np.zeros((its, 4))
    #strats = np.zeros(its, 2)
    for i in range(0, its):
        params['resource'] = base+step_size*i
        initial_conditions = list(params.values())
        if i is 0:
            init = base_case
        else:
            init = np.array([sol[0,-1], sol[1,-1], sol[2,-1]])

        tim, sol, flux_and_strat = rk23_w_flux(t_end, init, 0.01, lambda y, tno:
            optimal_behavior_trajectories(0, y, *initial_conditions, taun_old=tno)) #Consider solving semi-implicitly.

        strats_fluxes[i, 0] = flux_and_strat[0, -1] #taun
        strats_fluxes[i, 1] = flux_and_strat[1,-1] #taup
        strats_fluxes[i, 2] = np.sum(flux_and_strat[-2] * 0.01) #flux n
        strats_fluxes[i, 3] =  np.sum(flux_and_strat[-1] * 0.01) #flux p
        #flux_and_strat_storer.append(flux_and_strat)

    return strats_fluxes
#        solution_storer.append(sol)

data_init = list(np.linspace(phi0_base, its*phi0_base, its))
agents = 8
with Pool(processes = agents) as pool:
    results = pool.map(dynamic_pred_prey, data_init, 1)


results = np.array(results)
plt.imshow(results[:,:, 0], cmap='viridis')
plt.colorbar()
plt.show()

plt.imshow(results[:,:, 1], cmap='viridis')
plt.colorbar()
plt.show()

#for j in range(its):
#    cmax = 3
#    mu0 = 0.3
#    mu1 = 0.3
#    eps = 0.5
#    epsn = 0.5
#    cp = 3
#    phi0 = phi0_base + j*step_size_phi
#    phi1 = 0.2
#    lam = 0.5

#    opt_prey = True
#    opt_pred = True

#    initial_conditions_5 =  [cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, opt_prey, opt_pred]

#    t_start = 0
#    t_end = 60

#    init = np.array([0.8, 0.5, 0.5])
#    time_b, sol_basic, flux_and_strat_bas = rk23_w_flux(t_end, init, 0.01, lambda y :
#                                        optimal_behavior_trajectories_basic(0, y, *initial_conditions_5))

#    init_0 = np.array([sol_basic[0,-1], sol_basic[1,-1], sol_basic[2,-1]])



#    static_store = []
#    static_store_flux = []

#    for i in range(0,its):
#        initial_conditions_5 =  [cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, base+i*step_size, lam, opt_prey, opt_pred]
#        if i is 0:
#            init = init_0
#        else:
#            init = np.array([sol_basic[0,-1], sol_basic[1,-1], sol_basic[2,-1]])
#        time_b, sol_basic, flux_and_strat_bas = rk23_w_flux(t_end, init, 0.01, lambda y:
#        optimal_behavior_trajectories_basic(0, y, *initial_conditions_5))

#        static_store_flux.append(flux_and_strat_bas)
#        static_store.append(sol_basic)


# In[158]:


#    solution_storer = []
#    flux_and_strat_storer = []

#    for i in range(0, its):
#        initial_conditions_5 =  [cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, base+step_size*i, lam, opt_prey, opt_pred]
#        if i is 0:
#            init = init_0
#        else:
#            init = np.array([sol[0,-1], sol[1,-1], sol[2,-1]])

#        tim, sol, flux_and_strat = rk23_w_flux(t_end, init, 0.01, lambda y:
#            optimal_behavior_trajectories(0, y, *initial_conditions_5)) #Consider solving semi-implicitly.

#        flux_and_strat_storer.append(flux_and_strat)
#        solution_storer.append(sol)


#    flux_diff_n = np.zeros(len(solution_storer))
#    flux_diff_p = np.zeros(len(solution_storer))
#    taup_vec = np.zeros(len(solution_storer))
#    taun_vec = np.zeros(len(solution_storer))


#    resource = np.zeros(len(solution_storer))
#    for i in range(len(solution_storer)):
#        flux_diff_n[i] = np.sum(0.01 * static_store_flux[i][-2]) - np.sum(flux_and_strat_storer[i][-2] * 0.01)
#        flux_diff_p[i] = np.sum(0.01 * static_store_flux[i][-1]) - np.sum(flux_and_strat_storer[i][-1] * 0.01)
#        taup_vec[i] = flux_and_strat_storer[i][1,-1]
#        taun_vec[i] = flux_and_strat_storer[i][0,-1]
#        resource[i] = base+i*step_size

#plt.plot(resource, flux_diff_n, 'x', label = 'Flux diff n')
#plt.plot(resource, flux_diff_p, 'x', label = 'Flux diff p')
#plt.legend(loc = 'lower left')
#plt.show()



#plt.plot(resource, taup_vec, 'x', label = 'taup')
#plt.plot(resource, taun_vec, 'x', label = 'taun')
#plt.legend(loc = 'lower left')
#plt.show()
