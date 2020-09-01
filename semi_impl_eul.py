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
import common_functions as cf


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


def semi_implicit_euler(t_final, y0, step_size, f, params, opt_prey = True, opt_pred = True, nash = True, linear = False):
    solution = np.zeros((y0.shape[0], int(t_final / step_size)))
    strat = np.zeros((2,int(t_final / step_size)))
    t = np.zeros(int(t_final / step_size))
    solution[:, 0] = y0
    strat[:, 0] = cf.combined_strat_finder(params, y0, stackelberg=(not nash), x0=np.array([0.5, 0.5]), Gill = False, linear = linear)
    for i in range(1, int(t_final / step_size)):
        if opt_prey is True and opt_pred is True:
            taun_best, taup_best = cf.combined_strat_finder(params, solution[:, i-1], stackelberg=(not nash), x0=strat[:, i-1], Gill = False, linear = linear)
            print(t[i-1], "Time and strategy", taun_best)
        else:
            taun_best, taup_best = 1, 1
        strat[:, i] = taun_best, taup_best
        t[i] = t[i-1] + step_size
        fwd = f(t[i], solution[:, i - 1], taun_best, taup_best)
        solution[:, i] = solution[:, i-1] + step_size*fwd[0:3].reshape((3,))

    return t, solution, strat



def optimal_behavior_trajectories(t, y, params, taun = 1, taup = 1, linear = False):
    C = y[0]
    N = y[1]
    P = y[2]

    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    Cdot = lam*(cbar - C) - cmax*N*taun*C/(taun*C+nu0)
    Ndot = N*(epsn*cmax*taun*C/(taun*C+nu0) - taup * taun*P*cp/(taup*taun*N + nu1) - - mu0*taun**2 - mu1)
    if linear is False:
        Pdot = P*(cp*eps*taup*taun*N/(N*taup*taun + nu1) - phi0*taup**2 - phi1)
    else:
        Pdot = P * (cp * eps * taup * taun * N / (N * taup * taun + nu1) - phi0 * taup - phi1)

    return np.array([Cdot.squeeze(), Ndot.squeeze(), Pdot.squeeze()])



def optimal_behavior_trajectories_euler_OG(t, y, params, seasons = False, taun=1, taup=1):
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
