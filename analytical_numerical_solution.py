import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import sys
from io import StringIO
from scipy import optimize as optm
import scipy.integrate
import copy as copy
from common_functions import *

def opt_taup_find_old(y, taun, params):

    """Deprecated"""
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
    """Deprecated"""
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
    """Deprecated"""
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
    """Deprecated"""
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()


    taun_fitness_II = lambda s_prey: \
        epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * opt_taup_find(y, s_prey, params) * s_prey * P / (opt_taup_find(y, s_prey, params) * s_prey * N + cp) - mu0 * s_prey - mu1
    val = optm.fminbound(lambda x: -taun_fitness_II(x), full_output=True, disp=False, x1 = 0, x2 = 1)[0]
#    print(taun_fitness_II(0.465827056568672), taun_fitness_II(val), N, P)
#    print(optm.fminbound(lambda x: -taun_fitness_II(x), full_output=True, disp=True, x1 = 0, x2 = 1))
#    print(val, "val", opt_taup_find(y, val, params))
    return val




def optimal_behavior_trajectories_version_2(y, params, strat = None, nash = True, Gill = False, linear = False):
    """
    System dynamics, in a version where the inner game is solved as part of the system.
    :param y: Current state
    :param params: System parameters
    :param strat: Initial value for strategy finding
    :param nash: Whether to find the nash or stackelberg equilibrium
    :param Gill: Whether to use Gilliams rule or instantaneous growth as fitness proxy
    :param linear: A linear or quadratic arena cost
    :return: Instantaneous growth
    """

    C = y[0]
    N = y[1]
    P = y[2]

    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    taun, taup = combined_strat_finder(params, y, stackelberg = (not nash), x0 = strat, Gill = Gill)

    Cdot = lam*(cbar - C) - cmax*N*taun*C/(taun*C+nu0)
    Ndot = N*(epsn*cmax*taun*C/(taun*C+nu0) - taup * taun*P*cp/(taup*taun*N + nu1) - - mu0*taun**2 - mu1)
    if linear is False:
        Pdot = P*(cp*eps*taup*taun*N/(N*taup*taun + nu1) - phi0*taup**2 - phi1)
    else:
        Pdot = P * (cp * eps * taup * taun * N / (N * taup * taun + nu1) - phi0 * taup - phi1)

    return np.array([Cdot.squeeze(), Ndot.squeeze(), Pdot.squeeze()])


def continuation_slope_ODE(f, x0, params, strat = None, type = 'resource', h = 0.000001, nash = True, Gill = False, root = True, linear = False):
    """
    This function calculates the slope of the fixed points for use in creating a continuation guess to second order.
    :param f: The vector field describing the dynamics
    :param x0: The previous fixed point
    :param params: System parameters
    :param strat: Strategy at the previous fixed point
    :param type: The sensitivity parameter
    :param h: Fineness
    :param nash: Are we calculating the Nash or stackelberg equilibrium
    :param Gill: Are we using Gilliams rule as fitness proxy
    :param root: Use a root-finding procedure to find the zero or a less accurate but more robust least-squares procedure
    :param linear: Whether the cost of staying in the arena is linear or quadratic
    :return: Slope at x0
    """

    params_interior = copy.deepcopy(params)
    if root is True:
        params_interior[type] += h
        up = optm.root(lambda y: f(y, params_interior, nash = nash, strat = strat, Gill = Gill, linear = linear), x0=x0, method='hybr').x
        params_interior[type] -= 2*h
        down = optm.root(lambda y: f(y, params_interior, nash = nash, strat = strat, Gill = Gill, linear = linear), x0=x0, method='hybr').x
    else:
        params_interior[type] += h
        up = optm.least_squares(lambda y: f(y, params_interior, nash = nash, strat = strat, Gill = Gill, linear = linear), x0=x0, bounds = (0, np.inf)).x
        params_interior[type] -= 2*h
        down = optm.least_squares(lambda y: f(y, params_interior, nash = nash, strat = strat, Gill = Gill, linear = linear), x0=x0, bounds = (0, np.inf)).x


    return (up-down)/(2*h)

def continuation_strat(params, x, type = 'resource', h = 0.000005, nash = True, x0 = np.array([0.5, 0.5])):
    """Incomplete"""
    params_int = copy.deepcopy(params)
    params_int[type] += h
    up = combined_strat_finder(params, x, stackelberg=not nash, x0=x0)
    pass


def function_wrapper(f, x, y, step = 10**(-4)):
    """Incomplete, the idea was to exploit the system stability to find fix points when the continuation procedure failed"""
    return y+step*f(x)

def fp_stable_de(tolerance):
    pass


def continuation_func_ODE(f, x0, params, start, stop, its, reverse = True, strat = np.array([0.5, 0.5]), type = 'resource', h = 0.000005, nash = True, Gill = False, root = True, linear = False, verbose = True):
    """
    This function performs the actual continuation procedure, forming the heart of the article. The function signature makes it look more generic than it actually is, but the function f must be the function optimal_behavior_trajectories_version_2.

    :param f: A supposedly generic function to perform continuation on, in reality optimal_behavior_trajectories_version_2 in disguise.
    :param x0: A guess for the initial value of the continuation interval.
    :param params: Baseline parameters
    :param start: Start of the parameter being varied
    :param stop: Final value of the parameter being varied
    :param its: Number of iterations, the inverse of the step-size.
    :param reverse: The direction of the continuation. If true the continuation goes upwards.
    :param strat: Strategy best guess for the initial value. Mainly useful when creating a grid.
    :param type: Which parameter value we are performing the sensitivity analysis for.
    :param h: Accuracy of second approximation when determining the slope for the continuation guess.
    :param nash: Whether we are looking for a Nash or Stackelberg equilibrium.
    :param Gill: Whether we are using Gilliams rule or instantaneous pr. capita growth.
    :param root: If true, we use a root finding procedure to find the fixed point. Else a more robust but less accurate least-squares method is used.
    :param linear: Whether the loss from staying in the arena is linear or quadratic. The loss is linear if True.
    :param verbose: If true, output status of continuation for each step. The recommendation is to disable when creating a grid, unless something is seriously weird.
    :return: An array of fixed points, and accompanying strategies.
    """

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
        big_old_values[0] = optm.root(lambda y: f(y, params_int, nash = nash, strat = strat, Gill = Gill, linear = linear), x0=x0, method='hybr').x
    else:
        big_old_values[0] = optm.least_squares(lambda y: f(y, params_int, nash = nash, strat = strat, Gill = Gill, linear = linear), x0=x0, bounds = (0, np.inf)).x

    all_strats[0] = combined_strat_finder(params_int, big_old_values[0], stackelberg = (not nash), x0 = strat, Gill = Gill, linear = linear)

    for i in range(1,its):
        #print(i, params_int)
        params_int[type] = interval[i]
        cont_guess = big_old_values[i-1] + dire*step_size*continuation_slope_ODE(f, big_old_values[i-1], params_int, strat = all_strats[i-1], type = type, h = h, nash = nash, root = root, linear = linear)
        cont_guess[cont_guess < 0] = 0
        if root is True:
            optm_obj = optm.root(lambda y: f(y, params_int, nash = nash, strat = all_strats[i-1], Gill = Gill, linear = linear), x0=cont_guess, method='hybr')
            #optm.newton(lambda y: np.array([1, 10, 10**3])*f(y, params_int, nash = nash, strat = all_strats[i-1], Gill = Gill, linear = linear), x0=big_old_values[i-1], x1=cont_guess)
            x_temp = optm_obj.x
            success_try = True
            if optm_obj.success is False or np.linalg.norm(x_temp-big_old_values[i-1])>step_size*np.min(big_old_values[i-1]):
                success_try = False
                inner_its = 0
                while success_try is False and inner_its<10: #In case the fixed point was not found with the continuation guess, keep trying values that are slightly bigger than the previous value, and if this ends up failing use the continuation guess as the fixed point.
                    optm_obj = optm.root(
                        lambda y: f(y, params_int, nash=nash, strat=all_strats[i - 1], Gill=Gill, linear=linear),
                        x0=big_old_values[i-1]+dire*its*step_size*big_old_values[i-1], method='hybr')
                    inner_its+=1
                    success_try = copy.deepcopy(optm_obj.success)
                if success_try is False:
                    x_temp = cont_guess #big_old_values[i-1]
                #print("Oh man", Gill, nash)
        else:
            optm_obj = optm.least_squares(lambda y: f(y, params_int, nash = nash, strat = strat, Gill = Gill, linear = linear), x0=x0, bounds = (0, np.inf))
            print(optm_obj)
            x_temp = optm_obj.x

        big_old_values[i] = x_temp
        if verbose is True:
            print(x_temp, optm_obj.success, cont_guess, type) #Print continuation output for manual inspection of success while running

        if x_temp[-1] < 10**(-10):
            big_old_values[i] = big_old_values[i-1]
            print(params_int['phi0'])
        #print(x_temp-cont_guess, x_temp-big_old_values[i-1], optm_obj.success, type, reverse)
        all_strats[i] = combined_strat_finder(params_int, big_old_values[i], stackelberg = not nash, x0 = all_strats[i-1], Gill = Gill)

    if reverse is True:
        big_old_values = big_old_values[::-1]
        all_strats = all_strats[::-1]

    print(interval[-1])
    return big_old_values, all_strats





its = 0

opt_prey = True
opt_pred = True
#Currently the affinity is set to 1!!

def flux_calculator(R, C, P, taun, taup, params, linear = False):
    """
    The funciton calculates the trophic production
    :param R: Resouces
    :param C: Consumers
    :param P: Predators
    :param taun: Consumer strategy
    :param taup: Predator strategy
    :param params: System parameters
    :param linear: Whether the loss from staying in the arena is linear or quadratic, default is False (quadratic)
    :return: Production
    """
    flux_01 = params['cmax']*C * taun * R / (taun * R + params['nu0'])
    flux_12 = C * taup * taun * P * params['cp'] * 1 / (taup * taun * C + params['nu1'])
    if linear is False:
        flux_2n = P*params['phi0'] * taup** 2
    else:
        flux_2n = P * params['phi0'] * taup

    return np.array([flux_01, flux_12, flux_2n])

def frp_calc(R, C, P, taun, taup, params):
    """
    Calculate the relative instantaneous growth rate without loss
    :param R: Resources
    :param C: Consumers
    :param P: Predators
    :param taun: Consumer strategy
    :param taup: Predator strategy
    :param params: System parameters
    :return: Instantanenous relative growth rates
    """
    frp_C = taun * R / (taun * R + params['nu0']) #removed cmax
    frp_P = C * taup * taun * 1 / (taup * taun * C + params['nu1']) #Removed cp

    return np.array([frp_C, frp_P])


