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



taup = lambda s_prey, N : cp*(np.sqrt(eps/(phi0*s_prey*N)) - 1/(N*s_prey))


#prey_strats = np.linspace(0.01,1,198)

#prey_numbers = np.linspace(0.1,30,198)

#X, Y = np.meshgrid(prey_strats, prey_numbers)

#Z = taup(X, Y)

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 50)

taun_fitness_II = lambda s_prey, C, P, N, : cmax*s_prey*C/(s_prey*C+cmax) - cp * taup(s_prey, N) * s_prey*P/(taup(s_prey, N)*s_prey*N + cp) - mu0*s_prey - mu1




def optimal_behavior_trajectories(t, y, cmax, mu0, mu1, eps, cp, phi0, phi1, cbar, lam, opt_prey = False, opt_pred=True, seasons = False):
    C = y[0]
    N = y[1]
    P = y[2]
    taun = 1
    taup_opt = 1
    if opt_prey is True and opt_pred is True:
        taun = min(max(optm.minimize(lambda test: -taun_fitness_II(test, C, P, N), 0.5).x[0],0),1)

    elif opt_prey is True:
        #print(optm.minimize(lambda s_prey: -(2*s_prey*C/(s_prey*C+cmax) - taup_opt * s_prey*P/(taup_opt*s_prey*N + cp) - mu0*s_prey - mu1), 0.5).x[0])
        taun = min(max(optm.minimize(lambda s_prey: -(2*s_prey*C/(s_prey*C+cmax) - taup_opt * s_prey*P/(taup_opt*s_prey*N + cp) - mu0*s_prey - mu1), 0.5).x[0],0),1)
    elif opt_pred is True:
        taup_opt = min(max(taup(taun, N),0),1)
    if seasons is True:
        Cdot = lam*(cbar+0.5*cbar*np.cos(t*np.pi/180) - C) - N*taun*C/(taun*C+cmax) #t is one month
    else:
        Cdot = lam*(cbar - C) - cmax*N*taun*C/(taun*C+cmax)
    flux_c_to_n = N*taun*C/(taun*C+cmax)
    flux_n_to_p = N*taup_opt * taun*P*cp*1/(taup_opt*taun*N + cp) #Now these values are calculated twice..
    Ndot = N*(cmax*taun*C/(taun*C+cmax) - taup_opt * taun*P*cp*1/(taup_opt*taun*N + cp) - mu0*taun - mu1)
    Pdot = P*(cp*eps*taup_opt*taun*N/(N*taup_opt*taun + cp) - phi0*taup_opt - phi1)
    return np.array([Cdot, Ndot, Pdot, taun, taup_opt, flux_c_to_n, flux_n_to_p])


def optimal_behavior_trajectories_basic(t, y, cmax, mu0, mu1, eps, cp, phi0, phi1, cbar, lam, prey_opt, pred_opt, seasons = False):
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


    Ndot = N*(cmax*taun*C/(taun*C+cmax) - taup_opt * taun*P*cp*1/(taup_opt*taun*N + cp) - mu0*taun - mu1)
    Pdot = P*(cp*eps*taup_opt*taun*N/(N*taup_opt*taun + cp) - phi0*taup_opt - phi1)
    return np.array([Cdot, Ndot, Pdot, taun, taup_opt, flux_c_to_n, flux_n_to_p])


def runge_kutta_step_3_order_autonomous(y, step_size, f):
    k1 = step_size * f(y)[0:3]
    k2 = step_size * f(y + 1 / 3 * k1)[0:3]
    k3 = step_size * f(y + 2 / 3 * k2)[0:3]

    return y + 1 / 4 * (k1 + 3 * k3), f(y)[3:]


def rk23_w_flux(t_final, y0, step_size, f):
    solution = np.zeros((y0.shape[0], int(t_final / step_size)))
    flux_and_strat = np.zeros((4,int(t_final / step_size)))

    t = np.zeros(int(t_final / step_size))
    solution[:, 0] = y0
    for i in range(1, int(t_final / step_size)):
        solution[:, i], flux_and_strat[:,i] = runge_kutta_step_3_order_autonomous(solution[:, i - 1], step_size, f)
        # print(runge_kutta_step_4_order_autonomous(solution[:,i-1], step_size, f))
        t[i] = t[i - 1] + step_size

    return t, solution, flux_and_strat



cbar = 4*1.1
cmax = 2
mu0 = 0.4 
mu1 = 0.2
eps = 0.5
cp = 2
phi0 = 0.20
phi1 = 0.4 
cbar = 3
lam = 0.5
opt_prey = True
opt_pred = True
initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, cbar, lam, opt_prey, opt_pred]

t_start = 0
t_end = 20

init = np.array([0.8, 0.5, 0.5])
time_b, sol_basic, flux_and_strat_bas = rk23_w_flux(t_end, init, 0.01, lambda y :
                                        optimal_behavior_trajectories_basic(0, y, *initial_conditions_5))




# In[97]:


init_0 = np.array([sol_basic[0,-1], sol_basic[1,-1], sol_basic[2,-1]])




# In[159]:


static_store = []
static_store_flux = []

for i in range(0,60):
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, 4.4+i*0.1, lam, opt_prey, opt_pred]
    if i is 0:
        init = init_0
    else:
        init = np.array([sol_basic[0,-1], sol_basic[1,-1], sol_basic[2,-1]])
    time_b, sol_basic, flux_and_strat_bas = rk23_w_flux(t_end, init, 0.01, lambda y:
    optimal_behavior_trajectories_basic(0, y, *initial_conditions_5))

    static_store_flux.append(flux_and_strat_bas)
    static_store.append(sol_basic)


# In[158]:


solution_storer = []
flux_and_strat_storer = []

for i in range(0, 60):
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, 4.4+0.1*i, lam, opt_prey, opt_pred]
    t_start = 0
    t_end = 20
    if i is 0:
        init = init_0
    else:
        init = np.array([sol[0,-1], sol[1,-1], sol[2,-1]])

    tim, sol, flux_and_strat = rk23_w_flux(t_end, init, 0.01, lambda y:
        optimal_behavior_trajectories(0, y, *initial_conditions_5)) #Consider solving semi-implicitly.

    flux_and_strat_storer.append(flux_and_strat)
    solution_storer.append(sol)


flux_diff_n = np.zeros(len(solution_storer))
flux_diff_p = np.zeros(len(solution_storer))
resource = np.zeros(len(solution_storer))
for i in range(len(solution_storer)):
    flux_diff_n[i] = np.sum(0.01 * static_store_flux[i][-2]) - np.sum(flux_and_strat_storer[i][-2] * 0.01)
    flux_diff_p[i] = np.sum(0.01 * static_store_flux[i][-1]) - np.sum(flux_and_strat_storer[i][-1] * 0.01)
    resource[i] = 4.4+i*0.1

plt.plot(resource, flux_diff_n, 'x', label = 'Flux diff n')
plt.plot(resource, flux_diff_p, 'x', label = 'Flux diff p')
plt.legend(loc = 'lower left')
plt.show()

plt.plot(resource, flux_and_strat_storer[:][1], 'x', label = 'taup')
plt.plot(resource, flux_and_strat_storer[:][0], 'x', label = 'taun')
plt.legend(loc = 'lower left')
plt.show()
