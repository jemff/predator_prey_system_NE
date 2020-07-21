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
import numpy as np
from common_functions import *


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




def opt_taup_find_wazzup(y,s_prey,params): #This is the most recent derivation, it might be wrong tho. The other one is more battle-tested
    #s = 100
    #b = 330/12
    k_1 = params['cp']/params['nu1']*params['eps']*s_prey*y[1]/(2*params['phi0']) #k3 is phi0, k1 is b eps taun N s^{3/4} aka c_p #b*s**(3/4)
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


def semi_implicit_euler(t_final, y0, step_size, f, params, opt_prey = True, opt_pred = True, nash = True):
    solution = np.zeros((y0.shape[0], int(t_final / step_size)))
    flux = np.zeros((2,int(t_final / step_size)))
    strat = np.zeros((2,int(t_final / step_size)))
    t = np.zeros(int(t_final / step_size))
    solution[:, 0] = y0
    strat[0,0] = 0.5
    for i in range(1, int(t_final / step_size)):
        if nash is True:
            taun_best, taup_best = nash_eq_find(solution[:,i-1], params, opt_pred = opt_pred, opt_prey = opt_prey)
        else:
            taun_best, taup_best = strat_finder(solution[:,i-1], params, opt_prey = opt_prey, opt_pred = opt_pred, taun_old = strat[0,i-1])
        strat[:, i] = taun_best, taup_best
        t[i] = t[i-1] + step_size
        fwd = f(t[i], solution[:, i - 1], taun_best, taup_best)
#        print(fwd[0:3].squeeze(), solution[:, i-1], solution[:,i].shape, (solution[:, i-1].reshape(3) + step_size*fwd[0:3]).shape)
        solution[:, i] = solution[:, i-1] + step_size*fwd[0:3].reshape((3,))
        flux[:, i] = fwd[3:].reshape((2,))
    return t, solution, flux, strat




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
    Pdot = P*(cp*eps*taup*taun*N/(N*taup*taun + nu1) - phi0*taup**2 - phi1) #Square cost removed

    return np.array([Cdot, Ndot, Pdot, flux_c_to_n, flux_n_to_p])


#params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
#          'resource': base, 'lam':lam}


def dynamic_pred_prey(phi0_dyn, params, step_size=0.01, its=20):
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


its = 0
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
