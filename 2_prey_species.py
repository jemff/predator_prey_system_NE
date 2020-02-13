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

def strat_finder(y, params):

    taun_scared, taun_bold = opt_taun_find(y, params)


    #print(taun_scared, taun_bold)
    taup = opt_taup_find(y, max(taun_scared, taun_bold), params)


    return taun_scared, taun_bold, taup



def opt_taup_find(y, taun, params):
    C = y[0]
    N1 = y[1]
    N2 = y[2]

    N = N2 + N1

    taun = np.array([taun])
    if np.sum(taun.shape)>1:
        taun = np.squeeze(taun)
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()
    res1 = cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun)) #np.log(np.exp(cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun)))+1)
    res2 = -res1

    res1[res1 < 0.0] = 0
    res2[res2 < 0.0] = 0
    res1[res1 > 1] = 1
    res2[res2 > 1] = 1

    r1 = cp * eps * res1 * taun * N / (N * res1 * taun + cp) - phi0 * res1 - phi1

    r2 = cp * eps * res2 * taun * N / (N * res2 * taun + cp) - phi0 * res2 - phi1

    res = np.copy(res2)
    res[np.where(r1 > r2)] = res1[np.where(r1 > r2)]

    return res

def opt_taun_find(y, params):
    C, N_scared, N_bold, P = y[0], y[1], y[2], y[3]

    N = N_bold +N_scared

    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()
#    taun_fitness_II = lambda s_prey: \
#        epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * opt_taup_find(y, s_prey, params) * s_prey * P / (
#                opt_taup_find(y, s_prey, params) * s_prey * N + cp) - mu0 * s_prey - mu1

    p_term = lambda s_prey : (N*s_prey*taup(s_prey)+cp)

    taup = lambda s_prey: opt_taup_find(y, s_prey, params) #cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey)) -cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey))

    taup_prime = lambda s_prey: cp*(1/(N*s_prey**2) - 1/2*np.sqrt(eps/(phi0*N))*s_prey**(-3/2))

    taun_fitness_II_d = lambda s_prey: epsn*(cmax**2)*C/((s_prey*C+cmax)**2) - mu0 \
                                                 - (cp**2)*P*((s_prey*taup_prime(s_prey)+taup(s_prey))/(p_term(s_prey))**2)

    com_num = np.linspace(0.0001,1,500)
    max_num = opt_taup_find(y, com_num, params)
    border_of_comp = com_num[np.where(max_num > 0)[0][0]-1]

    print(max(taun_fitness_II(com_num, y, params)))
    timid_prey_options = np.linspace(0.0001, 2*border_of_comp, 100)

    timid_prey = timid_prey_options[np.argmax(taun_fitness_II(timid_prey_options, y, params))]

    max_cands_two = np.zeros(4)

    max_cands_two += border_of_comp*2
    linsp = np.linspace(border_of_comp*2 , 1, 200)
    comparison_numbs = taun_fitness_II_d(linsp)
    linsp = np.linspace(border_of_comp*2, 1, 200)


    max_cands_two[3] = 1
    if len(np.where(comparison_numbs > 0)[0]) is 0 or len(np.where(comparison_numbs < 0)[0]) is 0:
        max_cands_two[1] = linsp[np.argmax(taun_fitness_II(linsp,y,params))] #1

    else:
        maxi_mill = linsp[np.where(comparison_numbs > 0)[0][-1]]
        mini_mill = linsp[np.where(comparison_numbs < 0)[0][-1]]
        max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[mini_mill, maxi_mill], method='brentq').root
#        print(taun_fitness_II(border_of_comp, y, params), taun_fitness_II(max_cands, y, params))
        max_cands_two[2] = max_cands

#    print(max_cands_two[np.argmax(taun_fitness_II(max_cands_two))], border_of_comp)

    return timid_prey, max_cands_two[np.argmax(taun_fitness_II(max_cands_two, y, params))]


def taun_fitness_II(s_prey, y, params):
    C, N_scared, N_bold, P = y[0], y[1], y[2], y[3]

    N = N_bold + N_scared


    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()

    return epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * opt_taup_find(y, s_prey, params) * s_prey * P / (
                opt_taup_find(y, s_prey, params) * s_prey * N + cp) - mu0 * s_prey - mu1


def optimal_behavior_trajectories(t, y, params, seasons = False, taun_scared=1, taun_bold = 1, taup=1):
    C = y[0]
    N_scared = y[1]
    N_bold = y[2]
    N_tot = y[1]+y[2] #
    rat_s = N_scared/N_tot  #taun_scared*N_scared/N_tot
    rat_b = N_bold/N_tot #taun_bold*N_bold/N_tot

    P = y[3]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()
    if seasons is True:
        Cdot = lam*(cbar+0.5*cbar*np.cos(t*np.pi/12) - C) - cmax*(N_scared*taun_scared*C/(taun_scared*C+cmax) + N_bold*taun_bold*C/(taun_bold*C+cmax)) # t is one month
    else:
        Cdot = lam*(cbar - C) - cmax*(N_scared*taun_scared*C/(taun_scared*C+cmax) + N_bold*taun_bold*C/(taun_bold*C+cmax))
    #flux_c_to_n = N*taun*C/(taun*C+cmax)
    #flux_n_to_p = N*taup * taun*P*cp*1/(taup*taun*N + cp)  # Now these values are calculated twice..
    N_scareddot = N_scared*(epsn*cmax*taun_scared*C/(taun_scared*C+cmax) - mu0*taun_scared - mu1 - rat_s*cp*taup*taun_scared*N_scared/(rat_s*N_scared*taup*taun_scared + cp)) #Well, still weird. Maybe remove the option to eat the scared guys after all?? All very weird. #, removed them from the pool now. I guess the idea was they didn't get eaten at all.
    N_bolddot = N_bold*(epsn*cmax*taun_bold*C/(taun_bold*C+cmax) - rat_b*taup * taun_bold*P*cp*1/(rat_b*taup*taun_bold*N_bold + cp) - mu0*taun_bold - mu1)
    Pdot = P*(rat_b*cp*eps*taup*taun_bold*N_bold/(rat_b*N_bold*taup*taun_bold + cp) + rat_s*cp*eps*taup*taun_scared*N_scared/(rat_s*N_scared*taup*taun_scared + cp) - phi0*taup - phi1) # - The scared guys should also be eaten. Nothing else to do. #
    return np.array([Cdot, N_scareddot, N_bolddot, Pdot])


def semi_implicit_euler(t_final, y0, step_size, f, params, opt_prey = True, opt_pred = True):
    solution = np.zeros((y0.shape[0], int(t_final / step_size)))
    flux = np.zeros((2,int(t_final / step_size)))
    strat = np.zeros((3,int(t_final / step_size)))
    t = np.zeros(int(t_final / step_size))
    solution[:, 0] = y0
    strat[0,0] = 0.5
    for i in range(1, int(t_final / step_size)):
        taun_scared_best, taun_bold_best, taup_best = strat_finder(solution[:,i-1], params)
        strat[:, i] = taun_scared_best, taun_bold_best, taup_best
        fwd = f(t, solution[:, i-1], taun_scared_best, taun_bold_best, taup_best)
        t[i] = t[i-1] + step_size
        solution[:, i] = solution[:, i-1] + step_size*fwd
    return t, solution, flux, strat



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

base = 20
its = 0 #40
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

params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
          'resource': base, 'lam':lam}

t_end = 40

init = np.array([3.85361793, 8, 4, 2.48515033]) #np.array([5.753812957581866, 5.490194692112937, 1.626801718856221])#
tim, sol, flux, strat = semi_implicit_euler(t_end, init, 0.0005, lambda t, y, tn_s, tn_b, tp:
optimal_behavior_trajectories(t, y, params_ext, taun_scared=tn_s, taun_bold = tn_b, taup=tp), params_ext, opt_prey=True, opt_pred=True)

C = sol[0,-1]
N = sol[1,-1]+sol[2,-1]
P = sol[-1,-1]

N_rel = 1 #sol[2,-1]/N

y = np.array([C, N, P])
numbs = np.linspace(0.0001,1,500)
#taun_fitness_II = lambda s_prey: \
#    epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * opt_taup_find(y, s_prey, params_ext) * s_prey * P / (
#                opt_taup_find(y, s_prey, params_ext) * s_prey * N + cp) - mu0 * s_prey - mu1
#print(max(taun_fitness_II(numbs, sol[:,-1], params_ext)), C, N, P, params_ext)
print(opt_taup_find(sol[:,-1], numbs[np.argmax(taun_fitness_II(numbs, sol[:,-1], params_ext))], params_ext), max(taun_fitness_II(numbs, sol[:,-1], params_ext)), sol[:,-1])
plt.scatter(numbs, (taun_fitness_II(numbs, sol[:,-1], params_ext)))
plt.show()



plt.plot(tim[1:], strat[0,1:], 'x', label = "Scared Prey foraging intensity",alpha=0.5)
plt.plot(tim[1:], strat[1,1:], 'x', label = "Bold Prey foraging intensity",alpha=0.5)
plt.plot(tim[1:], strat[2,1:], 'x', label = "Predator foraging intensity", alpha=0.5)
plt.xlabel("Months")
plt.ylabel("Intensity")
plt.legend(loc = 'center left')
plt.savefig("Indsvingning_strat.png")


plt.figure()
#plt.plot(tim, sol[0,:], label = 'Resource biomass')
plt.plot(tim, sol[1,:], label = 'Scared Prey biomass', color = 'Green')
plt.plot(tim, sol[2,:], label = 'Bold Prey biomass', color = 'Blue')
plt.plot(tim, sol[1,:]+sol[2,:], label = 'Total Prey biomass', color = 'Black')
plt.plot(tim, sol[3,:], label = 'Dynamic predator biomass', color = 'Red')


plt.xlabel("Months")
plt.ylabel("g/m^3")
plt.legend(loc = 'lower left')

plt.savefig("Indsvingning_2p.png")

plt.figure()


