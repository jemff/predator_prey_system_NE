"The purpose of this file is to provide a common interface for the functions in analytical_numerical_solution and semi_impl_eul"
from common_functions import *
import analytical_numerical_solution as an_sol
import semi_impl_eul as num_sol
import matplotlib.pyplot as plt

mass_vector = np.array([1, 1, 100])

its = 0
step_size = 0.5
one_dim_its = 30

cost_of_living, nu, growth_max, lam = parameter_calculator_mass(mass_vector)
base = 30
cbar = base
phi1 = cost_of_living[1] #/3 #*2#/2#*5
phi0 = cost_of_living[1]
eps = 0.7
epsn = 0.7

cmax, cp = growth_max
mu0 = 0 # cost_of_living[0] #/3 #*2#/2#*5 #*60 #/2
mu1 = cost_of_living[0]#/3 #*2 #*2#/2#*5 #*2 #*10 #/2
nu0 = nu[0] #nu
nu1 = nu[1] #nu

params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
          'resource': base, 'lam':lam, 'nu0':nu0, 'nu1': nu1}


t_end = 80

init = 0.01*np.array([3.252678746139226, 2.2460419897613613, 0.7498879285084376]) #np.array([5.753812957581866, 5.490194692112937, 1.626801718856221])#
tim, sol, flux, strat = num_sol.semi_implicit_euler(t_end, init, 0.0005, lambda t, y, tn, tp:
num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=True, opt_pred=True)
tim, sol_2, flux_2, strat_2 = semi_implicit_euler(t_end, init, 0.0005, lambda t, y, tn, tp:
num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=False, opt_pred=False)
tim, sol_3, flux_3, strat_3 = semi_implicit_euler(t_end, init, 0.0005, lambda t, y, tn, tp:
num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=True, opt_pred=False)
tim, sol_4, flux_4, strat_4 = semi_implicit_euler(t_end, init, 0.0005, lambda t, y, tn, tp:
num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=False, opt_pred=True)

print(sol_2)
C, N, P =  C, N, P = sol[:,-1]

print(C, N, P, "CNP1", strat_finder(sol[:,-1], params_ext))

y = np.array([C, N, P])
numbs = np.linspace(0,1,500)
taun_fitness_II = lambda s_prey: \
    epsn * cmax * s_prey * C / (s_prey * C + nu0) - cp * opt_taup_find(y, s_prey, params_ext) * s_prey * P / (
                opt_taup_find(y, s_prey, params_ext) * s_prey * N + nu1) - mu0 * s_prey**2 - mu1


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
    epsn * cmax * s_prey * C / (s_prey * C + nu0) - cp * opt_taup_find(y, s_prey, params_ext) * s_prey * P / (
                opt_taup_find(y, s_prey, params_ext) * s_prey * N + nu1) - mu0 * s_prey**2 - mu1

plt.scatter(numbs, (taun_fitness_II(numbs)))
plt.show()

plt.scatter(numbs, opt_taup_find(y, numbs, params_ext))
#print(opt_taup_find(y, numbs, params_ext), "taup", params_ext)
plt.show()

plt.figure()
#plt.plot(tim, sol[0,:], label = 'Resource biomass')
plt.plot(tim, sol[1,:], label = 'Dynamic prey biomass', color = 'Green')
plt.plot(tim, sol[2,:], label = 'Dynamic predator biomass', color = 'Red')
plt.plot(tim, sol_2[1,:], label = 'Static prey biomass', linestyle = '-.', color = 'Green')
plt.plot(tim, sol_2[2,:], label = 'Static predator biomass', linestyle = '-.', color = 'Red')

plt.xlabel("Months")
plt.ylabel("g/m^3")
plt.legend(loc = 'upper left')

plt.savefig("Indsvingning.png")

plt.figure()



plt.plot(tim[1:], strat[0,1:], 'x', label = "Prey foraging intensity",alpha=0.5)
plt.plot(tim[1:], strat[1,1:], 'x', label = "Predator foraging intensity", alpha=0.5)
plt.xlabel("Months")
plt.ylabel("Intensity")
plt.legend(loc = 'center left')
plt.savefig("Indsvingning_strat.png")

print(strat[0,1:])

plt.figure()
plt.plot(tim, sol_3[1,:], label = 'Dynamic prey biomass', color = 'Green')
plt.plot(tim, sol_3[2,:], label = 'Static predator biomass', color = 'Red')

plt.title("Dynamic prey, static predator")
plt.xlabel("Months")
plt.ylabel("g/m^3")
plt.legend(loc = 'upper left')

plt.savefig("Indsvingning_dynprey.png")

plt.figure()
plt.plot(tim, sol_4[1,:], label = 'Static prey biomass', linestyle = '-.', color = 'Green')
plt.plot(tim, sol_4[2,:], label = 'Dynamic predator biomass', linestyle = '-.', color = 'Red')

plt.title("Static prey, dynamic predator")
plt.xlabel("Months")
plt.ylabel("g/m^3")
plt.legend(loc = 'upper left')

plt.savefig("Indsvingning_mixed.png")

base = 0

dynamic_equilibria = np.zeros((one_dim_its, 3))

optimal_prey = np.zeros((one_dim_its, 3))
optimal_pred = np.zeros((one_dim_its, 3))



print(an_sol.optimal_behavior_trajectories(np.array([0.27757345235331043, 0.29627977517967535, 0.1416850251665414 ]), params_ext))

sol_3 = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x0 = np.array([0.27757345235331043, 0.29627977517967535, 0.001416850251665414 ]), method = 'hybr')
sol_static = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, opt_pred = False, opt_prey = False), x0 = np.array([2/3*base, 1.5, 1.5]), method = 'hybr')

h = 0.00005
jac_3 = jacobian_calculator(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), sol_3.x, h)
print(sol_3.message, sol_3.x, "Sol 3")
print(sol_static.x)
print(np.linalg.eig(jac_3)[0], "Jac3")


list_of_cbars = np.linspace(0, one_dim_its, one_dim_its)
parms_list = [params_ext]*one_dim_its
equilibria = np.zeros((one_dim_its, 3))
eigen_values = np.zeros((one_dim_its, 4))
flows = np.zeros((one_dim_its, 6))
optimal_strategies = np.zeros((one_dim_its,2))


for i in range(one_dim_its):
    params_ext['resource'] = base + step_size * i
    parms_list[i]['resource'] = list_of_cbars[i]
    equilibria[i] = static_eq_calc(parms_list[i])


    if i<10:
        sol_temp = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x0=equilibria[i], method='hybr')
        x_ext = sol_temp.x #optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0=equilibria[i], method='hybr').x
        print(sol_temp.message, an_sol.optimal_behavior_trajectories(x_ext, params_ext))
        x_prey = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, opt_pred = False), x0=equilibria[i], method='hybr').x
        x_pred = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, opt_prey = False), x0=equilibria[i], method='hybr').x
    else:
        sol_temp = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x0=x_ext, method='hybr')
        x_ext = sol_temp.x
        print(sol_temp.message, an_sol.optimal_behavior_trajectories(x_ext, params_ext), opt_taun_find(x_ext, params_ext, 1), )
        x_prey = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, opt_pred = False), x0=x_prey, method='hybr').x
        x_pred = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, opt_prey = False), x0=x_pred, method='hybr').x



    tc, tp = strat_finder(x_ext, params_ext)
    optimal_strategies[i,0] = tc
    optimal_strategies[i,1] = tp
    dynamic_equilibria[i] = x_ext
    optimal_prey[i] = x_prey
    optimal_pred[i] = x_pred
    flows[i,0:3] = an_sol.flux_calculator(x_ext[0], x_ext[1], x_ext[2], tc, tp, params_ext)
    flows[i, 3:] = an_sol.flux_calculator(equilibria[i,0], equilibria[i,1], equilibria[i,2], 1, 1, params_ext)


    jac0 = jacobian_calculator(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, opt_prey = False, opt_pred = False), equilibria[i], h)
    jac0[np.isnan(jac0)] = 0
    jac1 = jacobian_calculator(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x_ext, h)
    jac1[np.isnan(jac1)] = 0

    jac2 = jacobian_calculator(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x_prey, h)
    jac2[np.isnan(jac2)] = 0
    jac3 = jacobian_calculator(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x_pred, h)
    jac3[np.isnan(jac3)] = 0

    eigen_values[i, 0] = max(np.linalg.eig(jac0)[0])
    eigen_values[i, 1] = max(np.linalg.eig(jac1)[0])
    eigen_values[i, 2] = max(np.linalg.eig(jac2)[0])
    eigen_values[i, 3] = max(np.linalg.eig(jac3)[0])





opponent_strats = np.linspace(0.01,1,100)

#print(opponent_strats)

C, N, P = dynamic_equilibria[-1,0], dynamic_equilibria[-1,1], dynamic_equilibria[-1,2]

dynamic_equilibria[-1,2] = P
print(C,N,P)
#print(cmax, cp)
#print(opt_taup_find(dynamic_equilibria[-1], opponent_strats, params_ext))
test_func = lambda s_prey: - cp * 0.2* s_prey * P / (
        0.2 * s_prey * N + nu1) - mu1 + epsn * cmax * s_prey * C / (s_prey * C + nu0) #
print("wut", opt_taun_analytical(opponent_strats))
#print(opt_taun_analytical(dynamic_equilibria[-1], opponent_strats, 100, 0.7, 0.5454545454))

plt.figure()
plt.plot(opponent_strats,test_func(opponent_strats))
plt.show()

plt.figure()
plt.plot(opponent_strats, opt_taun_analytical(dynamic_equilibria[-1], opponent_strats, mass_vector[2], epsn, params_ext['nu0']))
plt.plot(opponent_strats, opt_taup_find(dynamic_equilibria[-1], opponent_strats, params_ext))
plt.savefig('Strategy_Intersection.png')

equilibria = np.log(equilibria)
dynamic_equilibria = np.log(dynamic_equilibria)
optimal_prey = np.log(optimal_prey)
optimal_pred = np.log(optimal_pred)


plt.figure()
#plt.plot(list_of_cbars, equilibria[:,0])
plt.plot(list_of_cbars, equilibria[:,1], label = 'Static consumer')
plt.plot(list_of_cbars, equilibria[:,2], label = 'Static predator')
#plt.savefig('Bifurcation_plot.png')

#plt.figure()
#plt.plot(list_of_cbars, equilibria[:,0])
#plt.plot(list_of_cbars, dynamic_equilibria[:, 0])
plt.plot(list_of_cbars, dynamic_equilibria[:, 1], label = 'Dynamic consumer')
plt.plot(list_of_cbars, dynamic_equilibria[:, 2], label = 'Dynamic predator')
plt.xlabel('Nutrient biomass in mp/m^3')
plt.ylabel('Log biomass in mp')
plt.legend(loc  = 'lower right')
plt.savefig('Bifurcation_plot_dynamic.png')


plt.figure()
#plt.plot(list_of_cbars, equilibria[:,0])
plt.plot(list_of_cbars, optimal_prey[:, 0])
plt.plot(list_of_cbars, optimal_prey[:, 1])
plt.plot(list_of_cbars, optimal_prey[:, 2])
plt.savefig('Bifurcation_plot_dynamic_prey.png')

plt.figure()
#plt.plot(list_of_cbars, equilibria[:,0])
plt.plot(list_of_cbars, optimal_pred[:, 0])
plt.plot(list_of_cbars, optimal_pred[:, 1])
plt.plot(list_of_cbars, optimal_pred[:, 2])
plt.savefig('Bifurcation_plot_dynamic_pred.png')

plt.figure()
plt.plot(list_of_cbars, np.exp(eigen_values[:,0]), label = 'Static model')
plt.plot(list_of_cbars, np.exp(eigen_values[:,1]), label = 'Doubly dynamic')
#plt.plot(list_of_cbars, np.exp(eigen_values[:,2]), label = 'Dynamic prey')
#plt.plot(list_of_cbars, np.exp(eigen_values[:,3]), label ='Dynamic predator')
plt.xlabel('Exp of dominating eigenvalue')
plt.ylabel("Resource in mp")
plt.legend(loc ='upper left')
plt.savefig("Eigenvalues_compare.png")

plt.figure()
plt.plot(list_of_cbars, flows[:,0]-flows[:,3], label = 'Difference in flow from 0 to 1, Dynamic-Static')
plt.plot(list_of_cbars, flows[:,1]-flows[:,4], label = 'Difference in flow from 1 to 2, Dynamic-Static')
plt.plot(list_of_cbars, flows[:,2]-flows[:,5], label = 'Difference in flow from 2 to n, Dynamic-Static')

plt.xlabel("Resource in mp")
plt.ylabel("mp/day")
plt.legend(loc = 'upper left')


plt.savefig("Flows_Compare.png")


plt.figure()
plt.plot(list_of_cbars, optimal_strategies[:,0], label = 'Consumer')
plt.plot(list_of_cbars, optimal_strategies[:,1], label = 'Predator')
plt.ylabel("Activity level")
plt.xlabel("mp/day")
plt.legend(loc = 'upper left')

plt.savefig("Strategies.png")
#print(equilibria[:,1]/equilibria[:,2])
#print(optimal_strategies)
#print(opt_taup_find(np.exp(dynamic_equilibria[-1]), np.linspace(0,1,100), params_ext))
