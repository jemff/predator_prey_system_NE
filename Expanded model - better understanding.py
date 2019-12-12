#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as scp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# In[2]:


from scipy import optimize as optm


# In[3]:


import scipy.integrate


# In[4]:


eps = 0.5
cp = 2.5
phi0 = 0.3
phi1 = 0.2


# In[5]:


taup = lambda s_prey, N : cp*(np.sqrt(eps/(phi0*s_prey*N)) - 1/(N*s_prey))


# In[6]:


prey_strats = np.linspace(0.01,1,198)


# In[7]:


prey_numbers = np.linspace(0.1,30,198)


# In[8]:


X, Y = np.meshgrid(prey_strats, prey_numbers)


# In[9]:


Z = taup(X, Y)


# In[10]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50)


# In[11]:


taup(0.1, 10)


# In[12]:


Z[Z<0] = 0


# In[13]:


np.max(Z)


# In[14]:


taup(prey_strats, 5000)


# In[15]:


Z=Z/25 #Maximum found by numeric experimentation. It _could_ be found analytically. Note that it is global.


# In[16]:


taun_fitness = lambda s_prey, C, P, N, : cmax*s_prey*C/(s_prey*C+cmax) - taup(s_prey, N) * s_prey*P - mu0*s_prey - mu1


# In[17]:


cmax = 0.3
mu0 = 0.08
mu1 = 0.02


# In[18]:


N = 1000
P = 10
C = 2000


# In[19]:


taun_fitness(prey_strats, C, P, N)


# In[65]:


taun_fitness_II = lambda s_prey, C, P, N, : cmax*s_prey*C/(s_prey*C+cmax) - cp * taup(s_prey, N) * s_prey*P/(taup(s_prey, N)*s_prey*N + cp) - mu0*s_prey - mu1


# In[21]:


taun_fitness_II(prey_strats, 2000, 50, 350)


# In[22]:


optm.minimize(lambda t: -taun_fitness(t, 2000, 1, 2000), 0.5)


# In[23]:


optm.minimize(lambda t: -taun_fitness_II(t, 5, 0, 0.8), 0.5).x[0]


# In[24]:


P


# In[72]:


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
    Ndot = N*(cmax*taun*C/(taun*C+cmax) - taup_opt * taun*P*cp*1/(taup_opt*taun*N + cp) - mu0*taun - mu1)
    Pdot = P*(cp*eps*taup_opt*taun*N/(N*taup_opt*taun + cp) - phi0*taup_opt - phi1)
    print(taun, taup_opt)
    return [Cdot, Ndot, Pdot]


# In[73]:


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
    Ndot = N*(cmax*taun*C/(taun*C+cmax) - taup_opt * taun*P*cp*1/(taup_opt*taun*N + cp) - mu0*taun - mu1)
    Pdot = P*(cp*eps*taup_opt*taun*N/(N*taup_opt*taun + cp) - phi0*taup_opt - phi1)
    return [Cdot, Ndot, Pdot]


# In[74]:


cmax = 0.3
mu0 = 0.48 
mu1 = 0.08
eps = 0.8
cp = 0.1
phi0 = 0.20
phi1 = 0.4 
cbar = 20
lam = 0.1 
initial_conditions_1 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, cbar, lam]


# In[75]:


cmax = 0.3
mu0 = 0.48 
mu1 = 0.08
eps = 0.8
cp = 0.1
phi0 = 0.20
phi1 = 0.4 
cbar = 20
lam = 0.1 
opt_prey = True
opt_pred = False
initial_conditions_2 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, cbar, lam, opt_prey, opt_pred]


# In[76]:


cmax = 0.3
mu0 = 0.48 
mu1 = 0.08
eps = 0.8
cp = 0.1
phi0 = 0.20
phi1 = 0.4 
cbar = 20
lam = 0.1 
opt_prey = True
opt_pred = True
initial_conditions_3 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, cbar, lam, opt_prey, opt_pred]


# In[77]:


cbar = 30
cmax = 0.3
mu0 = 0.48 
mu1 = 0.08
eps = 0.8
cp = 0.1
phi0 = 0.20
phi1 = 0.4 
cbar = 20
lam = 0.1 
opt_prey = True
opt_pred = True
initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, cbar, lam, opt_prey, opt_pred] #Using IC3 values


# In[78]:


import sys
from io import StringIO
capture = StringIO()
save_stdout = sys.stdout
sys.stdout = capture

t_start = 0
t_end = 50
init = [0.35, 1.8, 0.1] #[0.23, 0.3, 0.13]#
sol = scipy.integrate.solve_ivp(fun = lambda t, y: optimal_behavior_trajectories(t, y, *initial_conditions_5), method = 'RK45',
                                t_span = [t_start, t_end], y0 = init, max_step = 0.01)
sys.stdout = save_stdout


# In[103]:


behaviors = np.reshape(np.array([float(x) for x in capture.getvalue().split()]),(int(len(capture.getvalue().split())/2),2))


# In[105]:


plt.plot(sol.t, behaviors[:,0])


# In[117]:


cbar = 30
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


# In[118]:


sol_basic = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories_basic(t, y, *initial_conditions_5), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.1)


plt.plot(sol_basic.t, sol_basic.y[0])
plt.plot(sol_basic.t, sol_basic.y[1])
plt.plot(sol_basic.t, sol_basic.y[2])


# In[97]:


init = [sol_basic.y[0,-1], sol_basic.y[1,-1], sol_basic.y[2,-1]]


# In[130]:


plt.figure()
for i in range(8,9):
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, (i+1)*1.1, lam, opt_prey, opt_pred]
    t_start = 0
    t_end = 130
    #init = [0.8, 0.5, 0.5] #[0.35, 1.8, 0.1]
    sol_basic = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories_basic(t, y, *initial_conditions_5), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.1)
    plt.plot(sol_basic.t, sol_basic.y[2])

plt.show()


# In[132]:


init_0 = [sol_basic.y[0,-1], sol_basic.y[1,-1], sol_basic.y[2,-1]]


# In[159]:


static_store = []
for i in range(8,11):
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, (i+1)*1.1, lam, opt_prey, opt_pred]
    t_start = 0
    t_end = 120
    if i is 8:
        init = init_0
    else:
        init = [sol_basic.y[0,-1], sol_basic.y[1,-1], sol_basic.y[2,-1]]
    sol_basic = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories_basic(t, y, *initial_conditions_5), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.1)
    
    static_store.append(sol_basic)


# In[158]:


solution_storer = []
for i in range(8,11): #Original 12
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, (i+1)*1.1, lam, opt_prey, opt_pred]
    t_start = 0
    t_end = 120
    if i is 8:
        init = init_0
    else:
        init = [sol.y[0,-1], sol.y[1,-1], sol.y[2,-1]]
    sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories(t, y, *initial_conditions_5), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.01)
    
    solution_storer.append(sol)


# In[168]:


fig1 = plt.figure()
plt.plot(solution_storer[0].t, solution_storer[0].y[0])
plt.plot(solution_storer[0].t, solution_storer[0].y[1])
plt.plot(solution_storer[0].t, solution_storer[0].y[2])

plt.plot(static_store[0].t, static_store[1].y[0])
plt.plot(static_store[0].t, static_store[0].y[1])
plt.plot(static_store[0].t, static_store[0].y[2])




fig2 = plt.figure()

plt.plot(solution_storer[1].t, solution_storer[1].y[0])
plt.plot(solution_storer[1].t, solution_storer[1].y[1])
plt.plot(solution_storer[1].t, solution_storer[1].y[2])

plt.plot(static_store[1].t, static_store[1].y[0])
plt.plot(static_store[1].t, static_store[1].y[1])
plt.plot(static_store[1].t, static_store[1].y[2])


fig3 = plt.figure()

#plt.plot(solution_storer[2].t, solution_storer[2].y[0], label = 'Resource withs static consumption')
#plt.plot(solution_storer[2].t, solution_storer[2].y[1], label = 'Static prey')
plt.plot(solution_storer[2].t, solution_storer[2].y[2], label = 'Dynamic predators')

#plt.plot(static_store[2].t, static_store[2].y[0], label = 'Resource with dynamic consumption')
#plt.plot(static_store[2].t, static_store[2].y[1], label = 'Dynamic prey')
plt.plot(static_store[2].t, static_store[2].y[2], label = 'Static predators')
plt.title('Time evolution of a dynamic predator-prey system')
plt.xlabel('Days')
plt.ylabel('kg/m^3')
plt.legend(loc = 'lower left')
plt.savefig('Dynamic_animals.png')


# In[154]:



plt.plot(static_store[3].t, static_store[3].y[2])


# In[142]:


solution_storer[0].t


# In[71]:


init = [1, 0.5, 0.5]
sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories(t, y, *initial_conditions_5), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.05)


# In[15]:


plt.figure()
for i in range(8,12):
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, 5*(i+1), lam, opt_prey, opt_pred]
    t_start = 0
    t_end = 360
    init = [0.8, 0.5, 0.5]
    sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories(t, y, *initial_conditions_5, seasons = True), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.05, t_eval = np.linspace(t_start, t_end, 1000))
    plt.plot(sol.t, sol.y[2])
    plt.plot(sol.t, sol.y[1]) 

plt.show()


# In[34]:


plt.figure()
for i in range(0,4):

    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, 75+5*i, lam, opt_prey, opt_pred]
    t_start = 0
    t_end = 100
    init = [1.2, 0.58, 0.83]
    sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories(t, y, *initial_conditions_5), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.05)
    plt.plot(sol.t, sol.y[2])
#plt.plot(sol.t, sol.y[1])
plt.show()


# In[35]:


plt.figure()
for i in range(0,4):

    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, 75+5*i, lam, opt_prey, opt_pred]
    t_start = 0
    t_end = 100
    init = [1.2, 0.58, 0.83]
    sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories_basic(t, y, *initial_conditions_5), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.05)
    plt.plot(sol.t, sol.y[2])
#plt.plot(sol.t, sol.y[1])
plt.show()


# In[50]:


plt.figure()
for i in range(0,4):

    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, 75+5*i, lam, opt_prey, opt_pred]
    t_start = 0
    t_end = 100
    init = [1.2, 0.58, 0.83]
    sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories(t, y, *initial_conditions_5), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.05)
    plt.plot(sol.t, sol.y[2])
#plt.plot(sol.t, sol.y[1])
plt.show()


# In[41]:


plt.figure()
for i in range(1,4):

    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, i*0.05, lam, opt_prey, opt_pred]
    t_start = 0
    t_end = 100
    init = [0.12, 0.002, 0.00]
    sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories_basic(t, y, *initial_conditions_5), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.05)
    plt.plot(sol.t, sol.y[1])
#plt.plot(sol.t, sol.y[1])
plt.show()


# In[46]:


plt.figure()
for i in range(1,4):
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, i*0.05, lam, True, False]
    t_start = 0
    t_end = 600
    init = [0.12, 0.002, 0.00]
    sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories(t, y, *initial_conditions_5, ), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.1)
    plt.plot(sol.t, sol.y[1])
    #plt.plot(sol.t, sol.y[0])

plt.show()


# In[51]:


plt.figure()
for i in range(1,4):
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, i*0.05, lam, True, False]
    t_start = 0
    t_end = 600
    init = [0.12, 0.002, 0.00]
    sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories(t, y, *initial_conditions_5, seasons = True), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.1)
    plt.plot(sol.t, sol.y[1])
    #plt.plot(sol.t, sol.y[0])

plt.show()


# In[79]:


plt.figure()
for i in range(3,5):
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, i*0.028, lam, True, False]
    t_start = 0
    t_end = 600
    init = [0.12, 0.00000001, 0.00]
    sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories(t, y, *initial_conditions_5, ), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.1) #Should try changing the steady-state mortality. 
    plt.plot(sol.t, sol.y[1])
    #plt.plot(sol.t, sol.y[0])

tplt.show()


# In[78]:


plt.figure()
for i in range(3,5):
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, i*0.028, lam, True, False]
    t_start = 0
    t_end = 300
    init = [0.12, 0.00000001, 0.00]
    sol = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories_basic(t, y, *initial_conditions_5, ), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.1)
    plt.plot(sol.t, sol.y[1])
    #plt.plot(sol.t, sol.y[0])

plt.show()


# In[ ]:





# In[45]:


plt.plot(sol.t,sol.y[0])


# In[842]:


plt.figure()
for i in range(8,12):
    initial_conditions_5 =  [cmax, mu0, mu1, eps, cp, phi0, phi1, 5*(i+1), lam, opt_prey, opt_pred]
    t_start = 0
    t_end = 360
    init = [0.8, 0.5, 0.5] #[0.35, 1.8, 0.1]
    sol_basic = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories_basic(t, y, *initial_conditions_5, seasons = True), 
                                    t_span = [t_start, t_end], y0 = init, max_step = 0.01)
    plt.plot(sol_basic.t, sol_basic.y[2])

plt.show()


# In[848]:


plt.plot(sol_basic.t, sol_basic.y[1])


# In[756]:


t_start = 0
t_end = 130
init = [0.35, 1.8, 0.1]
sol_basic = scipy.integrate.solve_ivp(fun = lambda t, y : optimal_behavior_trajectories_basic(t, y, *initial_conditions_5), 
                                t_span = [t_start, t_end], y0 = init, max_step = 0.1)


# In[759]:


plt.plot(sol_basic.t, sol_basic.y[2])


# In[ ]:





# In[645]:


plt.plot(sol_basic.t, sol_basic.y[1])


# In[646]:


plt.plot(sol_basic.t, sol_basic.y[2])


# In[462]:





# In[458]:


t_start = 0
t_end = 130
init = [0.35, 1.8, 0.1]
sol_basic = scipy.integrate.solve_ivp(optimal_behavior_trajectories_basic, 
                                t_span = [t_start, t_end], y0 = init, max_step = 0.1)

