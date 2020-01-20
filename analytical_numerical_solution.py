import numpy as np
import scipy as scp
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


def heatmap_plotter(data, title, image_name, ext):
    plt.figure()
    plt.title(title)
    #    plt.colorbar(res_nums, fraction=0.046, pad=0.04)
    plt.xlabel("Cbar, kg/m^3")
    plt.ylabel("phi0, kg/(m^3 * week)")

    ax = plt.gca()
    im = ax.imshow(data, cmap='Reds', extent =ext)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    plt.savefig(image_name+".png")

base = 40
its = 40
step_size = 0.5
step_size_phi = 0.00125
cbar = base
phi0_base = 0.4

cmax = 2
mu0 = 0.2
mu1 = 0.2
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

eq_stat = static_eq_calc(params_ext)
#sol_4 = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([base, base, 4/10*base]), method = 'hybr') #Apparently only three real roots.

sol_3 = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([14.32501445, 16.32699008,  7.17698779]), method = 'hybr')
#sol_3 = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([12.24914961,  0.8,  0.98489586]), method = 'hybr')
sol_2 = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([2/3*base, 1/2*(base), 1/6*base]), method = 'hybr')
sol = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = eq_stat, method = 'hybr')
sol_4 = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([12.24914961,  0.8,  0.98489586]), method = 'broyden1')
#sol_hybr = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([2/3*base, 1/2*(base), 1/4*(base)]), method = 'hybr')
#sol_hybr = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = sol.x, method = 'hybr')


#sol = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0 = np.array([2/3*base, 1.5, 1.5]), method = 'krylov')
sol_static = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext, opt_pred = False, opt_prey = False), x0 = np.array([2/3*base, 1.5, 1.5]), method = 'hybr')

h = 0.00005
jac = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext), sol.x, h)
jac_stat = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext, opt_pred = False, opt_prey = False), eq_stat, h)
jac_2 = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext), sol_2.x, h)
jac_3 = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext), sol_3.x, h)
jac_4 = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext), sol_4.x, h)


print(sol.success)
print(sol.message, sol.x, "Sol 1")
print(sol_2.message, sol_2.x, "Sol 2")
print(sol_3.message, sol_3.x, "Sol 3")
#print(sol_3)
print(sol_4.message, sol_4.x, "SOL FOUR")

#sol_final = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext, opt_pred = False, opt_prey = False)/((y-sol.x)*(y-sol_2.x)*(y-sol_3.x)*(y-sol_4.x)), x0 = np.array([base,base,base]), method = 'hybr')

#print(sol_final, "DER ENDLÖSUNG")
#print(strat_finder(sol.x, params_ext))
print(sol_static.x)
#print(jac)
print(np.linalg.eig(jac)[0], "Jac1")
#print(np.linalg.eig(jac_2)[0], "Jac2")
print(np.linalg.eig(jac_3)[0], "Jac3")
print(np.linalg.eig(jac_4)[0], "Jac4")


print(np.linalg.eig(jac_stat)[0])
#print(np.linalg.eig(jac))

eq_stat = static_eq_calc(params_ext)
#print(eq_stat.shape)
#print(optimal_behavior_trajectories(eq_stat, params_ext, opt_pred=False, opt_prey=False))

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
 #       else:
 #           x_ext = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0=2*x_ext, method='hybr').x
    #    jac_temp = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext), x_ext, h)
    #    print(np.linalg.eig(jac_temp)[0])
        for j in range(its):
            params_ext['phi0'] = phi0_base+j*step_size_phi
            if i is 0:
                x_prev = np.copy(x_ext)
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
#            if eigen_max[i, j] > 0:
#                print("Oh no!")

#    print(prey_nums[:, 0])
    ran = [base, base+its*step_size, 100*phi0_base, 100*(phi0_base+its*step_size_phi)]
    print("Ran", ran)





    heatmap_plotter(res_nums, 'Resource kg/m^3', "resource_conc", ran)
    heatmap_plotter(prey_nums, 'Prey kg/m^3', "prey_conc", ran)
    heatmap_plotter(pred_nums, 'Predators kg/m^3', "pred_conc", ran)
    heatmap_plotter(taun_grid, 'Prey foraging intensity', "prey_for", ran)
    heatmap_plotter(taup_grid, 'Predator foraging intensity', "pred_for", ran)
    heatmap_plotter(eigen_max, 'Predator foraging intensity', "Eigenvalues", ran)

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

