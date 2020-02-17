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

    x_m = np.copy(x)
    x_p = np.copy(x)
    x_m -= h
    x_p += h
    derr = (f(x_p) - f(x_m))/(2*h)
#    print(derr)
    return derr


def strat_finder(y, params, opt_prey = True, opt_pred = True, taun_old = 1):
    C, N, P = y[0], y[1], y[2]
    taun = 1
    taup = 1
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()

    if opt_prey is True and opt_pred is True:
        taun = min(max(opt_taun_find(y, params, taun_old), 0), 1)
        taup = opt_taup_find(y, taun, params)

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

def opt_taup_find(y, s_prey, params):
    k = s_prey * y[1] / params['cp']
    c = params['eps'] * s_prey * y[1] / params['phi0']
    x = 1 / 3 * (2 ** (2 / 3) / (
                3 * np.sqrt(3) * np.sqrt(27 * c ** 2 * k ** 8 + 8 * c * k ** 7) + 27 * c * k ** 4 + 4 * k ** 3) ** (1 / 3)
                 + (3 * np.sqrt(3) * np.sqrt(27 * c ** 2 * k ** 8 + 8 * c * k ** 7) + 27 * c * k ** 4 + 4 * k ** 3) ** (
                             1 / 3) / (2 ** (2 / 3) * k ** 2) - 2 / k)
    #print(x.shape, c.shape, k.shape, y.shape, s_prey.shape)
    return x

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
        res1 = cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun)) #np.log(np.exp(cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun)))+1)
        res2 = -res1

        if np.sum(np.array(res1).shape)>2:
            res1[res1 < 0.0] = 0
            res2[res2 < 0.0] = 0
            res1[res1 > 1] = 1
            res2[res2 > 1] = 1

            r1 = cp * eps * res1 * taun * N / (N * res1 * taun + cp) - phi0 * res1 - phi1

            r2 = cp * eps * res2 * taun * N / (N * res2 * taun + cp) - phi0 * res2 - phi1

            #print(r1.shape, r2.shape)
            res = res2
#            print(res1, res2)
            #print(res1, "res1", res2, "res2")
            #print(res1[np.where(r1 < r2)], "res2 bigger")
            res[np.where(r1 > r2)] = res1[np.where(r1 > r2)]
#            res[res < 0] = 0
#            res[res > 1] = 1

        else:
            #res1 = max(res1, res2)
#            res2 = res1
#            print(res1, res2)
#            print(res1)
            res1 = max(min(res1, 1), 0)
            res2 = max(min(res2, 1), 0)

            r1 = cp * eps * res1 * taun * N / (N * res1 * taun + cp) - phi0 * res1 - phi1

            r2 = cp * eps * res2 * taun * N / (N * res2 * taun + cp) - phi0 * res2 - phi1

            if r1 > r2:
                res = res1
            else:
                res = res2
#            print(res)
#            print(r1, res1, r2, res2, res)
#            res = max(min(res, 1), 0.0)
#            res2 = max(min(res2, 1), 0.0)

#            r1 = cp * eps * res1 * taun * N / (N * res1 * taun + cp) - phi0 * res1 - phi1 #0.5 has been added as a numerical band-aid due to instability of the method
#            r2 = cp * eps * res2 * taun * N / (N * res2 * taun + cp) - phi0 * res2 - phi1
#            r3 = cp * eps * 1 * taun * N / (N * 1 * taun + cp) - phi0 * 1 - phi1
#            r4 =  cp * eps * 0.0 * taun * N / (N * 0.0 * taun + cp) - phi0 * 0.0 - phi1 #-phi1
#            pos_max = np.array([r1, r3, r4])


#            max_p = np.max(pos_max)
#            if r1 == max_p:
#                res = res1
#            elif r2 == max_p:
#                res = res2
#            elif r3 == max_p:
#                res = 1
#            else:
#                res = 0.5 #As above, numerical band-aid



            #res = min(max(res, 0), 1)
#        print(res)

        return res


def opt_taun_find_dumb(y, params, taun_old):
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()
    fishy_fish = np.linspace(0.001, 1, 300)
#    fishy_fish_2 = np.linspace(0.3, 1, 1000)


    taun_fitness_II = lambda s_prey: \
        epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * opt_taup_find(y, s_prey, params) * s_prey * P / (opt_taup_find(y, s_prey, params) * s_prey * N + cp) - mu0 * s_prey - mu1
#    val = optm.fminbound(lambda x: -taun_fitness_II(x), full_output=True, disp=False, x1 = 0.0, x2 = 1.0001)[0]
#    print(taun_fitness_II(0.238), taun_fitness_II(val), val, C, N, P)
#    print(optm.fminbound(lambda x: -taun_fitness_II(x), full_output=True, disp=True, x1 = 0, x2 = 1))
#    max1 = np.max(taun_fitness_II(fishy_fish))

#    max2 = np.max(taun_fitness_II(fishy_fish_2))

    dumb_val = fishy_fish[np.argmax(taun_fitness_II(fishy_fish))]

#    if np.abs(max1 - max2) < 1 or max2 > max1:
 #       dumb_val = fishy_fish_2[np.argmax(taun_fitness_II(fishy_fish_2))]
##        print("fishy2")
#    else:
#        dumb_val = fishy_fish[np.argmax(taun_fitness_II(fishy_fish))]
#        print(max1, max2, np.abs(max1 - max2))
#        print("fishy1")
    #    print(val, "val", taun_fitness_II(0.9811888017062286), taun_fitness_II(val), np.max(taun_fitness_II(fishy_fish)), fishy_fish[np.argmax(taun_fitness_II(fishy_fish))], "CNP", C , N, P)

#    print(dumb_val, max1, max2)
    return dumb_val


def opt_taun_find(y, params, dummy):
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()

    taun_fitness_II = lambda s_prey: epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * taup(s_prey) * s_prey * P / (
                    taup(s_prey) * s_prey * N + cp) - mu0 * s_prey - mu1
    p_term = lambda s_prey : (N*s_prey*taup(s_prey)+cp)


    taup = lambda s_prey: opt_taup_find(y, s_prey, params) #cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey)) -cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey))

    taup_prime = lambda s_prey: (opt_taup_find(y, s_prey+0.0001, params)-opt_taup_find(y, s_prey-0.0001, params))/(2*0.0001) #cp*(1/(N*s_prey**2) - 1/2*np.sqrt(eps/(phi0*N))*s_prey**(-3/2))

    taun_fitness_II_d = lambda s_prey: epsn*(cmax**2)*C/((s_prey*C+cmax)**2) - mu0 \
                                                 - (cp**2)*P*((s_prey*taup_prime(s_prey)+taup(s_prey))/(p_term(s_prey))**2)
    linsp = np.linspace(0.0001 , 1, 100)
    comparison_numbs = taun_fitness_II_d(linsp)



    maxi_mill = linsp[np.where(comparison_numbs > 0)[0][-1]]
    mini_mill = linsp[np.where(comparison_numbs < 0)[0][-1]]
    max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[mini_mill, maxi_mill], method='brentq').root
    #max_cands_two[2] = max_cands

    return max_cands #max_cands_two[np.argmax(taun_fitness_II(max_cands_two))]


def opt_taun_find_for_linear(y, params, dummy):
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()

    taun_fitness_II = lambda s_prey: epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * taup(s_prey) * s_prey * P / (
                    taup(s_prey) * s_prey * N + cp) - mu0 * s_prey - mu1
    p_term = lambda s_prey : (N*s_prey*taup(s_prey)+cp)


    taup = lambda s_prey: opt_taup_find(y, s_prey, params) #cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey)) -cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey))

    taup_prime = lambda s_prey: (opt_taup_find(y, s_prey+0.0001, params)-opt_taup_find(y, s_prey-0.0001, params))/(2*0.0001) #cp*(1/(N*s_prey**2) - 1/2*np.sqrt(eps/(phi0*N))*s_prey**(-3/2))

    taun_fitness_II_d = lambda s_prey: epsn*(cmax**2)*C/((s_prey*C+cmax)**2) - mu0 \
                                                 - (cp**2)*P*((s_prey*taup_prime(s_prey)+taup(s_prey))/(p_term(s_prey))**2)

    com_num = np.linspace(0.0001, 1, 400)
    max_num = opt_taup_find(y, com_num, params)
    border_of_comp = com_num[np.where(max_num > 0)[0][0]]
    max_cands_two = np.zeros(4)

    max_cands_two += border_of_comp
    linsp = np.linspace(border_of_comp , 1, 100)
    comparison_numbs = taun_fitness_II_d(linsp)
    linsp = np.linspace(border_of_comp, 1, 100)


    max_cands_two[3] = 1
    if len(np.where(comparison_numbs > 0)[0]) is 0 or len(np.where(comparison_numbs < 0)[0]) is 0:
        max_cands_two[1] = linsp[np.argmax(taun_fitness_II(linsp))]

    else:
        maxi_mill = linsp[np.where(comparison_numbs > 0)[0][-1]]
        mini_mill = linsp[np.where(comparison_numbs < 0)[0][-1]]
        max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[mini_mill, maxi_mill], method='brentq').root
        max_cands_two[2] = max_cands
#    if max_cands_two[2] != max_cands_two[0]: #np.abs(taun_fitness_II(max_cands_two[2])-taun_fitness_II(max_cands_two[0])) < 10*(-1):
        #print(taun_fitness_II(max_cands_two[2])-taun_fitness_II(max_cands_two[0]))
#        return max_cands_two[2]
#    else:
    return max_cands_two[np.argmax(taun_fitness_II(max_cands_two))]



def opt_taun_find_old(y, params, dummy):
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
#        t0 = taun_fitness_II(0.001)
#        t1 = taun_fitness_II(1)
#        if t0 > t1:
#            max_cands = 0.001
#        else:
#            max_cands = 1

    else:
        maxi_mill = linsp[np.where(comparison_numbs > 0)[0][0]]
        mini_mill = linsp[np.where(comparison_numbs < 0)[0][-1]]
        max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[mini_mill, maxi_mill], method='brentq').root


    print(max_cands)
    return max_cands


def opt_taun_find_oldschool(y, params, dummy):
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
    strat[0,0] = 0.5
    for i in range(1, int(t_final / step_size)):
        taun_best, taup_best = strat_finder(solution[:,i-1], params, opt_prey = opt_prey, opt_pred = opt_pred, taun_old = strat[0,i-1])
        strat[:, i] = taun_best, taup_best

        fwd = f(t[i], solution[:, i - 1], taun_best, taup_best)
        t[i] = t[i-1] + step_size
        solution[:, i] = solution[:, i-1] + step_size*fwd[0:3]
        flux[:, i] = fwd[3:]
    return t, solution, flux, strat

def binary_search_max(f, n, err = 10**(-8)):
    line = np.linsp(0,1,n)
    error = 1
    max_f = f(0)
    max_old = 0
    while error > err:
        res = f(line)
        max_loc = np.argmax(res)
        line = line / 2
        if max_loc > n/2:
            line += 1/2

        error = np.max(np.abs(f(line))-max_old)



base = 20
its = 40
step_size = 0.5*2.5
step_size_phi = 0.0025*2.5 #0.00125
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

t_end = 40


init = np.array([8.85361793, 6.85670493, 6.48515033]) #np.array([5.753812957581866, 5.490194692112937, 1.626801718856221])#
tim, sol, flux, strat = semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp), params_ext, opt_prey=True, opt_pred=True)
tim, sol_2, flux_2, strat_2 = semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp), params_ext, opt_prey=False, opt_pred=False)

#print(optimal_behavior_trajectories(0, sol[:,-10], params_ext))

#params_ext["resource"] = 40
C, N, P =  C, N, P = sol[:,-1]

print(C, N, P, "CNP1", strat_finder(sol[:,-1], params_ext), )

y = np.array([C, N, P])
numbs = np.linspace(0,1,500)
taun_fitness_II = lambda s_prey: \
    epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * opt_taup_find(y, s_prey, params_ext) * s_prey * P / (
                opt_taup_find(y, s_prey, params_ext) * s_prey * N + cp) - mu0 * s_prey - mu1


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
    epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * opt_taup_find(y, s_prey, params_ext) * s_prey * P / (
                opt_taup_find(y, s_prey, params_ext) * s_prey * N + cp) - mu0 * s_prey - mu1

plt.scatter(numbs, (taun_fitness_II(numbs)))
plt.show()

plt.scatter(numbs, opt_taup_find(y, numbs, params_ext))

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
