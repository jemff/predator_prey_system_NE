import numpy as np
import scipy as scp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from io import StringIO
from scipy import optimize as optm
import scipy.integrate
from multiprocessing import Pool


def strat_finder(y, params, opt_prey = True, opt_pred = True):
    C, N, P = y[0], y[1], y[2]
    taun = 1
    taup = 1
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()

    if opt_prey is True and opt_pred is True:
        taun = min(max(opt_taun_find(y, params), 0), 1)
        taup = min(max(opt_taup_find(y, taun, params), 0), 1)

    elif opt_prey is True and opt_pred is False:
        taun = min(max(optm.minimize(lambda s_prey: -(cmax*epsn*s_prey*C/(s_prey*C+cmax)
                                                      - cp*taup * s_prey*P/(taup*s_prey*N + cp)
                                                      - mu0*s_prey - mu1), 0.5).x[0],0),1)

    elif opt_pred is True and opt_prey is False:
        taup = min(max(taup(taun, N),0),1)

    return taun, taup


#def opt_taup_find(y, taun, params):
#    C = y[0]
#    N = y[1]
#    P = y[2]
#    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()
#    if taun is 0:
#        return 0
#    else:
#        taun = np.array([taun])
        #print(np.min(np.concatenate([np.max(np.concatenate([cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun)), np.array([1]) ])), np.array[0]])))
#        res = cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun))
#        res[np.where(res < 0)] = 0
#        res[np.where(res > 1)] = 1
#        return res
def opt_taup_find(y, taun, params):
    C = y[0]
    N = y[1]
    P = y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()
    if taun is 0:
        return 0
    else:
        #taun = np.array([taun])
        #print(np.min(np.concatenate([np.max(np.concatenate([cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun)), np.array([1]) ])), np.array[0]])))
        res = cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun))
        if res < 0 or res > 1:
            tau1 = cp * eps * 1 * taun * N / (N * 1 * taun + cp) - phi0 * 1 - phi1
            tau0 = -phi1
            if tau1 > tau0:
                res = 1
            else:
                res = 0
        return res



def opt_taun_find(y, params):
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()


    taun_fitness_II = lambda s_prey: \
        epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * opt_taup_find(y, s_prey, params) * s_prey * P / (
                    opt_taup_find(y, s_prey, params) * s_prey * N + cp) - mu0 * s_prey - mu1
    taun_fitness_II_d = lambda s_prey: epsn*cmax**2*C/(s_prey*C+cmax)**2 \
                                       - 3/2*(N**2*cp*(eps/(phi0*N))**(1/2)*s_prey**(5/2))**(-1) - mu0

#    print(taun_fitness_II_d(np.linspace(0,50,100)))
    linsp = np.linspace(0.001, 1, 100)
    comparison_numbs = (taun_fitness_II_d(linsp))
    if len(np.where(comparison_numbs > 0)[0]) is 0:
        t0 = taun_fitness_II(0)
        t1 = taun_fitness_II(1)
        if t0 > t1:
            max_cands = 0
        else:
            max_cands = 1

    else:
        maxi_mill = linsp[np.where(comparison_numbs > 0)[0][-1]]
        max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[0.001, maxi_mill], method='brentq').root

    return max_cands


def optimal_behavior_trajectories(t, y, params, opt_prey = True, opt_pred=True, seasons = False, taun=1, taup=1):
    C = y[0]
    N = y[1]
    P = y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()
    if seasons is True:
        Cdot = lam*(cbar+0.5*cbar*np.cos(t*np.pi/180) - C) - N*taun*C/(taun*C+cmax)  # t is one month
    else:
        Cdot = lam*(cbar - C) - cmax*N*taun*C/(taun*C+cmax)
    flux_c_to_n = N*taun*C/(taun*C+cmax)
    flux_n_to_p = N*taup * taun*P*cp*1/(taup*taun*N + cp)  # Now these values are calculated twice..
    Ndot = N*(epsn*cmax*taun*C/(taun*C+cmax) - taup * taun*P*cp*1/(taup*taun*N + cp) - mu0*taun - mu1)
    Pdot = P*(cp*eps*taup*taun*N/(N*taup*taun + cp) - phi0*taup - phi1)
    return np.array([Cdot, Ndot, Pdot, flux_c_to_n, flux_n_to_p])


def semi_implicit_euler(t_final, y0, step_size, f, params, opt_prey = True, opt_pred = True):
    solution = np.zeros((y0.shape[0], int(t_final / step_size)))
    flux = np.zeros((2,int(t_final / step_size)))
    strat = np.zeros((2,int(t_final / step_size)))
    t = np.zeros(int(t_final / step_size))
    solution[:, 0] = y0
    for i in range(1, int(t_final / step_size)):
        taun_best, taup_best = strat_finder(solution[:,i-1], params, opt_prey = opt_prey, opt_pred = opt_pred)
        strat[:, i] = taun_best, taup_best

        fwd = f(t[i], solution[:, i - 1], taun_best, taup_best)
        t[i] = t[i-1] + step_size
        solution[:, i] = solution[:, i-1] + step_size*fwd[0:3]
        flux[:, i] = fwd[3:]
    return t, solution, flux, strat

base = 40
its = 0
step_size = 1
step_size_phi = 0.05
cbar = base
phi0_base = 0.4

cmax = 2
mu0 = 0.2
mu1 = 0.2
eps = 0.7
epsn = 0.7
cp = 2
phi0 = phi0_base
phi1 = 0.2
lam = 0.5

opt_prey = True
opt_pred = True
#Currently the affinity is set to 1!!

params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
          'resource': base, 'lam':lam}

def dynamic_pred_prey(phi0_dyn, step_size=step_size, its=its, params=params_ext):
#    solution_storer = np.zeros
    flux_and_strat_storer = []
    params['phi0'] = phi0_dyn

    t_end = 120

    init = np.array([0.8, 0.5, 0.5])
    time_b, sol_basic, flux_bas, strat_bas = semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
            optimal_behavior_trajectories(t, y, params, taun=tn, taup=tp), params, opt_prey = False, opt_pred = False)
    base_case = np.array([sol_basic[0, -1], sol_basic[1, -1], sol_basic[2, -1]])


    tim, sol, flux, strats = semi_implicit_euler(t_end, base_case, 0.001, lambda t, y, tn, tp:
    optimal_behavior_trajectories(t, y, params, taun=tn, taup=tp), params)

    strats = np.zeros((its, 2))
    fluxes = np.zeros((its, 2))
    pops = np.zeros((its, 3))
    t_end = 100
    for i in range(0, its):
        params['resource'] = base+step_size*i
        init = np.array([sol[0,-1], sol[1,-1], sol[2,-1]])
        tim, sol, flux, strat = semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
            optimal_behavior_trajectories(t, y, params, taun=tn, taup=tp), params, opt_prey = True, opt_pred = True)

        tim_OG, sol_OG, flux_OG, strat_OG = semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
            optimal_behavior_trajectories(t, y, params, taun=tn, taup=tp), params, opt_prey = False, opt_pred = False)
        pops[i] = sol[:,-1] #np.sum((sol-sol_OG)*0.001, axis = 1) #sol[:,-1] - sol_OG[:,-1]
        plt.plot(tim, sol[0, :])
        strats[i] = strat[:, -1]
        if strats[i, 0] is 0 or strats[i, 1] is 0:
            print(strat)
        fluxes[i] = np.sum((flux-flux_OG)*0.001, axis = 1)
        #print(fluxes[i], pops[i], phi0_dyn, base+step_size*i)
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

t_end = 80
init = np.array([10.02380589, 11.16997066,  1.26961184])
tim, sol, flux, strat = semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp), params_ext, opt_prey=True, opt_pred=True)

plt.figure()
plt.plot(tim, sol[0,:], label = 'Resource biomass')
plt.plot(tim, sol[1,:], label = 'Prey biomass')
plt.plot(tim, sol[2,:], label = 'Predator biomass')
plt.xlabel("Days")
plt.ylabel("kg/m^3")
plt.legend(loc = 'upper right')

plt.savefig("Indsvingning.png")

plt.figure()
plt.plot(tim[1:], strat[0,1:], label = "Prey foraging intensity")
plt.plot(tim[1:], strat[1,1:], label = "Predator foraging intensity")
plt.xlabel("Days")
plt.ylabel("Intensity")
plt.legend(loc = 'upper right')
plt.savefig("Indsvingning_strat.png")

print(sol[:,-1])
