import numpy as np
import scipy as scp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from io import StringIO
from scipy import optimize as optm
import scipy.integrate
import copy as copy
from common_functions import *

def opt_taup_find_old(y, taun, params):
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


def opt_taun_find_dumb_dumb(y, params, dummy):
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()

    taun_fitness_II = lambda s_prey: epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * taup(s_prey) * s_prey * P / (
                    taup(s_prey) * s_prey * N + cp) - mu0 * s_prey - mu1
    taun_fitness_II_d_old = lambda s_prey: epsn*cmax**2*C/(s_prey*C+cmax)**2 \
                                       -  P*cp*3/2*(N**2*cp*(eps/(phi0*N))**(1/2)*s_prey**(5/2))**(-1) - mu0

    p_term = lambda s_prey : (N*s_prey*taup(s_prey)+cp)


    taup = lambda s_prey: cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey)) -cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey))

    taup_prime = lambda s_prey: cp*(1/(N*s_prey**2) - 1/2*np.sqrt(eps/(phi0*N))*s_prey**(-3/2))

    taun_fitness_II_d = lambda s_prey: epsn*(cmax**2)*C/((s_prey*C+cmax)**2) - mu0 \
                                                 - (cp**2)*P*((s_prey*taup_prime(s_prey)+taup(s_prey))/(p_term(s_prey))**2)

    comparison_numbs = np.zeros(100)
    linsp = np.linspace(0.001, 1, 100)
    for i in range(100):
        comparison_numbs[i] = taun_fitness_II_d(linsp[i])

    if len(np.where(comparison_numbs > 0)[0]) is 0 or len(np.where(comparison_numbs < 0)[0]) is 0:

        max_cands = linsp[np.argmax(taun_fitness_II(linsp))]

    else:
        maxi_mill = linsp[np.where(comparison_numbs > 0)[0][0]]
        mini_mill = linsp[np.where(comparison_numbs < 0)[0][-1]]
        max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[mini_mill, maxi_mill], method='brentq').root

#    print(max_cands)s
    return max_cands



def opt_taun_find_old(y, params, dummy):
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()


    taun_fitness_II = lambda s_prey: \
        epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * opt_taup_find(y, s_prey, params) * s_prey * P / (
                    opt_taup_find(y, s_prey, params) * s_prey * N + cp) - mu0 * s_prey - mu1
    taun_fitness_II_d = lambda s_prey: epsn*cmax**2*C/(s_prey*C+cmax)**2 \
                                       - 3/2*(N**2*cp*(eps/(phi0*N))**(1/2)*s_prey**(5/2))**(-1) - mu0

    linsp = np.linspace(0.001, 1, 100)
    comparison_numbs = (taun_fitness_II_d(linsp))
    if len(np.where(comparison_numbs > 0)[0]) is 0 or len(np.where(comparison_numbs < 0)[0]) is 0:
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


def opt_taun_find_unstable(y, params, taun_old):
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()


    taun_fitness_II = lambda s_prey: \
        epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * opt_taup_find(y, s_prey, params) * s_prey * P / (opt_taup_find(y, s_prey, params) * s_prey * N + cp) - mu0 * s_prey - mu1
    val = optm.fminbound(lambda x: -taun_fitness_II(x), full_output=True, disp=False, x1 = 0, x2 = 1)[0]
#    print(taun_fitness_II(0.465827056568672), taun_fitness_II(val), N, P)
#    print(optm.fminbound(lambda x: -taun_fitness_II(x), full_output=True, disp=True, x1 = 0, x2 = 1))
#    print(val, "val", opt_taup_find(y, val, params))
    return val




def optimal_behavior_trajectories_version_2(y, params, strat = None, nash = True, Gill = False):
    C = y[0]
    N = y[1]
    P = y[2]

    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    taun, taup = combined_strat_finder(params, y, stackelberg = (not nash), x0 = strat, Gill = Gill)

    Cdot = lam*(cbar - C) - cmax*N*taun*C/(taun*C+nu0)
    Ndot = N*(epsn*cmax*taun*C/(taun*C+nu0) - taup * taun*P*cp/(taup*taun*N + nu1) - - mu0*taun**2 - mu1)
    Pdot = P*(cp*eps*taup*taun*N/(N*taup*taun + nu1) - phi0*taup**2 - phi1)

    return np.array([Cdot.squeeze(), Ndot.squeeze(), Pdot.squeeze()])


def continuation_slope_ODE(f, x0, params, strat = None, type = 'resource', h = 0.000001, nash = True, Gill = False, root = True):
    params_interior = copy.deepcopy(params)
    if root is True:
        params_interior[type] += h
        up = optm.root(lambda y: f(y, params_interior, nash = nash, strat = strat, Gill = Gill), x0=x0, method='hybr').x
        params_interior[type] -= 2*h
        down = optm.root(lambda y: f(y, params_interior, nash = nash, strat = strat, Gill = Gill), x0=x0, method='hybr').x
    else:
        params_interior[type] += h
        up = optm.least_squares(lambda y: f(y, params_interior, nash = nash, strat = strat, Gill = Gill), x0=x0, bounds = (0, np.inf)).x
        params_interior[type] -= 2*h
        down = optm.least_squares(lambda y: f(y, params_interior, nash = nash, strat = strat, Gill = Gill), x0=x0, bounds = (0, np.inf)).x


    return (up-down)/(2*h)

def continuation_strat(params, x, type = 'resource', h = 0.000005, nash = True, x0 = np.array([0.5, 0.5])):
    params_int = copy.deepcopy(params)
    params_int[type] += h
    up = combined_strat_finder(params, x, stackelberg=not nash, x0=x0)
    pass

def continuation_func_ODE(f, x0, params, start, stop, its, reverse = True, strat = np.array([0.5, 0.5]), type = 'resource', h = 0.000005, nash = True, Gill = False, root = True):
    interval = np.linspace(start, stop, its)
    step_size = interval[1]-interval[0]
    params_int = copy.deepcopy(params)
    big_old_values = np.zeros((its, *x0.shape))
    dire = 1

    all_strats = np.zeros((its, 2))
    if reverse is True:
        interval = interval[::-1]
        dire = - 1
    params_int[type] = interval[0]
    if root is True:
        big_old_values[0] = optm.root(lambda y: f(y, params_int, nash = nash, strat = strat, Gill = Gill), x0=x0, method='hybr').x
    else:
        big_old_values[0] = optm.least_squares(lambda y: f(y, params_int, nash = nash, strat = strat, Gill = Gill), x0=x0, bounds = (0, np.inf)).x

    all_strats[0] = combined_strat_finder(params, big_old_values[0], stackelberg = (not nash), x0 = strat, Gill = Gill)

    for i in range(1,its):
        #print(i, params_int)
        params_int[type] = interval[i]
        cont_guess = big_old_values[i-1] + dire*step_size*continuation_slope_ODE(f, big_old_values[i-1], params_int, strat = all_strats[i-1], type = type, h = h, nash = nash, root = root)
        cont_guess[cont_guess < 0] = 0
        if root is True:
            optm_obj = optm.root(lambda y: f(y, params_int, nash = nash, strat = all_strats[i-1], Gill = Gill), x0=cont_guess, method='hybr')
            x_temp = optm_obj.x
            if optm_obj.success is False:
                x_temp = cont_guess #big_old_values[i-1]
                print("Oh man", Gill, nash)
        else:
            optm_obj = optm.least_squares(lambda y: f(y, params_int, nash = nash, strat = strat, Gill = Gill), x0=x0, bounds = (0, np.inf))
            print(optm_obj)
            x_temp = optm_obj.x

        big_old_values[i] = x_temp
        print(x_temp-cont_guess, x_temp-big_old_values[i-1], optm_obj.success, type, reverse)
        all_strats[i] = combined_strat_finder(params, big_old_values[i], stackelberg = not nash, x0 = all_strats[i-1], Gill = Gill)
        if type == 'phi0':
            print(optm_obj, i)

    if reverse is True:
        big_old_values = big_old_values[::-1]
        all_strats = all_strats[::-1]


    return big_old_values, all_strats


def heatmap_plotter(data, title, image_name, ext):
    plt.figure()
    plt.title(title)
    #    plt.colorbar(res_nums, fraction=0.046, pad=0.04)
    plt.xlabel("Cbar, g/m^3")
    plt.ylabel("phi0, 10^(-2) g/(m^3 * week)")

    ax = plt.gca()
    im = ax.imshow(data, cmap='Reds', extent =ext)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    plt.savefig(image_name+".png", bbox_inches='tight')



its = 0

opt_prey = True
opt_pred = True
#Currently the affinity is set to 1!!

manual_max = False
if manual_max is True:
    for i in range(int(base*10)):
        for j in range(int(base*10)):
            for k in range(int(base*10)):
                test_num = np.array([0.1+0.1*i, 0.1+0.1*j, 0.1+0.1*k])
                is_opt = np.abs(optimal_behavior_trajectories(test_num, params_ext))
                if np.sum(is_opt) < 0.5*10**(-1):
                    print(test_num)

    print("I'm done")

def flux_calculator(R, C, P, taun, taup, params, linear = False):

    flux_01 = params['cmax']*C * taun * R / (taun * R + params['nu0'])
    flux_12 = C * taup * taun * P * params['cp'] * 1 / (taup * taun * C + params['nu1'])
    if linear is False:
        flux_2n = P*params['phi0'] * taup** 2
    else:
        flux_2n = P * params['phi0'] * taup

    return np.array([flux_01, flux_12, flux_2n])

def frp_calc(R, C, P, taun, taup, params):

    frp_C = taun * R / (taun * R + params['nu0']) #removed cmax
    frp_P = C * taup * taun * 1 / (taup * taun * C + params['nu1']) #Removed cp

    return np.array([frp_C, frp_P])

its = 0
if its > 0:
    taun_grid = np.zeros((its, its))
    taup_grid = np.zeros((its, its))
    res_nums = np.zeros((its, its))
    prey_nums = np.zeros((its, its))
    pred_nums = np.zeros((its, its))
    start_point = np.copy(sol_3.x)
    eigen_max = np.zeros((its, its))
    x_ext = np.zeros(3)
    x_prev = np.zeros(3)


    for i in range(its):
        params_ext['resource'] = base + step_size*i
        if i is 0:
            x_ext = np.copy(start_point)
        else:
            x_ext = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0=x_ext, method='hybr').x
    #    jac_temp = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext), x_ext, h)
    #    print(np.linalg.eig(jac_temp)[0])
        for j in range(its):
            params_ext['phi0'] = phi0_base+j*step_size_phi
            if j is 0:
                x_prev = np.copy(x_ext)
                sol_temp = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0=x_prev, method='hybr')
                x_prev = sol_temp.x
            else:
                sol_temp = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0=x_prev, method='hybr')
                x_prev = sol_temp.x
 #               if sol_temp.success is False or j is 0:
 #                   print(sol_temp.message)

            jac_temp = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext), x_prev, h)
            #print(np.linalg.eig(jac_temp)[0], x_prev, params_ext["phi0"], params_ext["resource"])
#            print(np.real(np.linalg.eig(jac_temp)[0].max()))
            eigen_max[i, j] = np.real(np.linalg.eig(jac_temp)[0].max())
            res_nums[i, j] = x_prev[0]
            prey_nums[i, j] = x_prev[1]
            pred_nums[i, j] = x_prev[2]
            taun_grid[i, j], taup_grid[i, j] = strat_finder(x_prev, params_ext)
#            print(taup_grid[i, j], j)
#            if eigen_max[i, j] > 0:
#                print("Oh no!")

#    print(prey_nums[:, 0])
    ran = [base, base+its*step_size, 100*phi0_base, 100*(phi0_base+its*step_size_phi)]
    print("Ran", ran)



    temporary_thingy = np.zeros((its, its))
    temporary_thingy[0,:] = 1
    heatmap_plotter(temporary_thingy, "test", "test", ran)
    heatmap_plotter(res_nums.T, 'Resource g/m^3', "resource_conc", ran)
    heatmap_plotter(prey_nums.T, 'Prey g/m^3', "prey_conc", ran)
    heatmap_plotter(pred_nums.T, 'Predators g/m^3', "pred_conc", ran)
    heatmap_plotter(taun_grid.T, 'Prey foraging intensity', "prey_for", ran)
    heatmap_plotter(taup_grid.T, 'Predator foraging intensity', "pred_for", ran)
    heatmap_plotter(eigen_max.T, 'Eigenvalues', "Eigenvalues", ran)

#    plt.figure()
#    plt.title('Resource kg/m^3')
#    #    plt.colorbar(res_nums, fraction=0.046, pad=0.04)
#    plt.xlabel("Cbar, kg/m^3")
#    plt.ylabel("phi0, kg/(m^3 * week)")
#
#    ax = plt.gca()
#    im = ax.imshow(res_nums, cmap='Reds', extent =ran)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)

#    plt.colorbar(im, cax=cax)

#    plt.savefig("resource_conc.png")
#    plt.show()

