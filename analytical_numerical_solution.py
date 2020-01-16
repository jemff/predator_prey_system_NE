import numpy as np
import scipy as scp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from io import StringIO
from scipy import optimize as optm
import scipy.integrate
from multiprocessing import Pool
def jacobian_calculator(f, x, h):
    jac = np.zeros((x.shape[0], x.shape[0]))
    x_m = np.copy(x)
    x_p = np.copy(x)
    for i in range(len((x))) :
        x_m[i] -= h
        x_p[i] += h
        jac[:, i] = (f(x_p) - f(x_m))/(2*h)
        x_m = np.copy(x)
        x_p = np.copy(x)

    return jac

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

    return np.array([taun]), np.array([taup])

def static_eq_calc(params):
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()

    phitild = phi0+phi1
    mutild = mu0 + mu1
    N_star = phitild*cp/(cp*eps-phitild) #-(epsn*cmax*lam + cp/(phitild*eps))/(cmax*epsn-cp/phitild-mutild)
    btild = cmax*(1+N_star/lam) - cbar

    C_star = 1/2*(-btild+np.sqrt(btild**2 + 4*cbar*cmax))

    P_star = (1/cp+1/N_star)*(epsn*lam*(cbar-C_star)-mutild*N_star)
#    print(np.array([C_star, N_star, P_star]))
    return np.array([C_star, N_star, P_star])

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

def optimal_behavior_trajectories(y, params, opt_prey = True, opt_pred = True):
    C = y[0]
    N = y[1]
    P = y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()
    taun, taup = strat_finder(y, params, opt_prey = opt_prey, opt_pred = opt_pred)
    Cdot = lam*(cbar - C) - cmax*N*taun*C/(taun*C+cmax)
    Ndot = N*(epsn*cmax*taun*C/(taun*C+cmax) - taup * taun*P*cp*1/(taup*taun*N + cp) - mu0*taun - mu1)
    Pdot = P*(cp*eps*taup*taun*N/(N*taup*taun + cp) - phi0*taup - phi1)
    return np.array([Cdot.squeeze(), Ndot.squeeze(), Pdot.squeeze()])

base = 90
its = 16
step_size = 1
step_size_phi = 0.05
cbar = base
phi0_base = 0.4

cmax = 2
mu0 = 0.2
mu1 = 0.3
epsn = 0.7

eps = 0.7
cp = 2
phi0 = phi0_base
phi1 = 0.2
lam = 0.5

opt_prey = True
opt_pred = True
#Currently the affinity is set to 1!!

params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
          'resource': base, 'lam':lam}

sol = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([2/3*base, 1/2*(base), 1/2*base]), method = 'hybr')
#sol_broy1 = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([2/3*base, 1/2*(base), 1/4*(base)]), method = 'broyden1')
#sol_hybr = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([2/3*base, 1/2*(base), 1/4*(base)]), method = 'hybr')
#sol_hybr = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = sol.x, method = 'hybr')


#sol = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([2/3*base, 1.5, 1.5]), method = 'krylov')
sol_static = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext, opt_pred = False, opt_prey = False), x0 = np.array([2/3*base, 1.5, 1.5]), method = 'hybr')

h = 0.0001
jac = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext), sol.x, h)
jac_stat = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext, opt_pred = False, opt_prey = False), sol_static.x, h)

#print(sol, "Krylov")
#print(sol_broy1, "Broyden1")
#print(sol_hybr, "Hybrid")

#rmat = np.zeros((3, 3))
#rmat[0] = sol_hybr.r[0:3]
#rmat[1,1:] = sol_hybr.r[3:5]
#rmat[2,2] = sol_hybr.r[-1]
##print(sol_hybr.r, rmat)
#jac = np.dot(sol_hybr.fjac, rmat)
#rmat[0] = sol_static.r[0:3]
#rmat[1,1:] = sol_static.r[3:5]
#rmat[2,2] = sol_static.r[-1]
#jac_stat = np.dot(sol_static.fjac, rmat)

#eq_stat = static_eq_calc(params_ext)
#print(eq_stat)
#print(optimal_behavior_trajectories(eq_stat, params_ext, opt_pred = False, opt_prey = False))
print(sol)
print(strat_finder(sol.x, params_ext))
print(sol_static)
#print(jac)
print(np.linalg.eig(jac)[0])
print(np.linalg.eig(jac_stat)[0])
#print(np.linalg.eig(jac))

eq_stat = static_eq_calc(params_ext)
#print(eq_stat.shape)
print(optimal_behavior_trajectories(eq_stat, params_ext, opt_pred=False, opt_prey=False))
