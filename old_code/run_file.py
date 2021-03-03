"The purpose of this file is to provide a common interface for the functions in analytical_numerical_solution and semi_impl_eul"
from common_functions import *
import analytical_numerical_solution as an_sol
import semi_impl_eul as num_sol
import matplotlib.pyplot as plt

configuration = {'verbose' : False, 'quadratic' : True, 'metabolic_cost' : False}


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


mass_vector = np.array([1, 1, 100])

its = 0
step_size = 0.5 #0.5
total_range = 18
one_dim_its = int(total_range/step_size)

cost_of_living, nu, growth_max, lam = parameter_calculator_mass(mass_vector, v = 0.1)
base = 16
phi1 = cost_of_living[1]#*2/3 #/3 #*2#/2#*5
phi0 = cost_of_living[1]#*4/3 #The problem is that nash equiibrium does not exist in the low levels...
eps = 0.7
epsn = 0.7

cmax, cp = growth_max
mu0 = 0 # cost_of_living[0] #/3 #*2#/2#*5 #*60 #/2
mu1 = cost_of_living[0]#/3 #*2 #*2#/2#*5 #*2 #*10 #/2
nu0 = nu[0] #nu
nu1 = nu[1] #nu

params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
          'resource': base, 'lam':lam, 'nu0':nu0, 'nu1': nu1}


t_end = 5
init = 2*static_eq_calc(params_ext) #np.array([0.5383403,  0.23503815, 0.00726976]) #np.array([5.753812957581866, 5.490194692112937, 1.626801718856221])#

#params_ext['resource'] = 16 This si for the low-resource paradigm
tim, sol, flux, strat = num_sol.semi_implicit_euler(t_end, init, 0.0005, lambda t, y, tn, tp:
num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=True, opt_pred=True)
tim, sol_2, flux_2, strat_2 = num_sol.semi_implicit_euler(t_end, init, 0.0005, lambda t, y, tn, tp:
num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=False, opt_pred=False)

fidelity = 100
resource_variation = np.linspace(sol[0,-1]*0.01, sol[0,-1], fidelity)
prey_variation = np.linspace(sol[1,-1]*0.01, sol[1,-1], fidelity)
frp = np.zeros((fidelity,2))
frc = np.zeros((fidelity,2))
for i in range(fidelity):
    frp[i] = nash_eq_find(np.array([sol[0,-1], prey_variation[i], sol[2,-1]]), params_ext)
    frc[i] = nash_eq_find(np.array([resource_variation[i], sol[1,-1], sol[2,-1]]), params_ext)


plt.figure()
plt.title("Functional response of predator, P " + str(np.round(sol[2,-1], 2)) + " R " + str(np.round(sol[0, -1],2)))
plt.plot(prey_variation, params_ext['cp']*prey_variation/(prey_variation+params_ext['nu0']), 'x', label = "Predator functional response, non-optimal",alpha=0.5)
plt.plot(prey_variation, params_ext['cp']*frp[:,0]*frp[:,1]*prey_variation/(frp[:,0]*frp[:,1]*prey_variation+params_ext['nu0']), 'x', label = "Predator functional response, optimal",alpha=0.5)
plt.xlabel("Prey in $m_p/m^3$")
plt.ylabel("Functional response")
plt.legend(loc = 'center left')
plt.savefig("Functional_response_predator.png")

plt.figure()
plt.title("Functional response of consumer, C " + str(np.round(sol[1,-1], 2)) + " P " + str(np.round(sol[2, -1],2)))
plt.plot(resource_variation, params_ext['cmax']*resource_variation/(resource_variation+params_ext['nu0']), 'x', label = "Consumer functional response, non-optimal",alpha=0.5)
plt.plot(resource_variation, params_ext['cmax']*frc[:,0]*resource_variation/(frc[:,0]*resource_variation+params_ext['nu0']), 'x', label = "Consumer functional response, optimal",alpha=0.5)
plt.xlabel("Resource in $m_p/m^3$")
plt.ylabel("Functional response")
plt.legend(loc='center left')
plt.savefig("Functional_response_consumer.png")

C, N, P = sol[:,-2]


plt.figure()
#plt.plot(tim, sol[0,:], label = 'Resource biomass')
plt.plot(tim, np.log(sol[1,:]), label = 'Dynamic prey biomass', color = 'Green')
plt.plot(tim, np.log(sol[2,:]), label = 'Dynamic predator biomass', color = 'Red')
plt.plot(tim, np.log(sol_2[1,:]), label = 'Static prey biomass', linestyle = '-.', color = 'Green')
plt.plot(tim, np.log(sol_2[2,:]), label = 'Static predator biomass', linestyle = '-.', color = 'Red')

plt.xlabel("Months")
plt.ylabel("Log $m_p/m^3$")
plt.legend(loc = 'lower left')

plt.savefig("Indsvingning.png")

#print(sol_2)
plt.figure()

plt.plot(tim[1:], strat[0,1:], 'x', label = "Prey foraging intensity",alpha=0.5)
plt.plot(tim[1:], strat[1,1:], 'x', label = "Predator foraging intensity", alpha=0.5)
plt.xlabel("Months")
plt.ylabel("Intensity")
plt.legend(loc = 'center left')
plt.savefig("Indsvingning_strat.png")


opponent_strats = np.linspace(0.01,1,100)

taun_data = opt_taun_analytical(sol[:,-1], opponent_strats, mass_vector[2], epsn, params_ext['nu0'])

taun_data[taun_data > 1] = 1
taun_data[taun_data < 0] = 0
taups = opt_taup_find(sol[:,-1], opponent_strats, params_ext)
nesh = opt_taun_analytical(sol[:,-1], opt_taup_find(sol[:,-1], opponent_strats, params_ext), mass_vector[2], epsn, params_ext['nu0'])
nush = opt_taup_find(sol[:,-1], opt_taun_analytical(sol[:,-1], opponent_strats, mass_vector[2], epsn, params_ext['nu0']), params_ext)
nesh = nesh - opponent_strats
nush = nush - opponent_strats
plt.figure()
plt.plot(taun_data, opponent_strats, label = "Best response of consumer")
plt.plot(opponent_strats, opt_taup_find(sol[:,-1], opponent_strats, params_ext), label = 'Best response of predator')
#plt.plot(opponent_strats,nesh, label = 'Best response of prey - prey strategy' )
#plt.plot(opponent_strats,nush, label = 'Best response of predator - predator strategy' )
#plt.ylabel("Intensity")
plt.ylabel("$\\tau_1$")
plt.xlabel("$\\tau_0$")
plt.title("Strategies for R " + str(np.round(sol[0,-1], 8)) + " C: " + str(np.round(sol[1,-1], 8)) + " P: " +  str(np.round(sol[2,-1], 8)) ) #, np.round(dynamic_equilibria[-1], 2))
plt.legend(loc = 'center left')
plt.savefig('Strategy_Intersection.png')


dynamic_equilibria = np.zeros((one_dim_its, 3))
dynamic_equilibria_stack = np.copy(dynamic_equilibria)


optimal_prey = np.zeros((one_dim_its, 3))
optimal_pred = np.zeros((one_dim_its, 3))

optimal_prey_stack = np.copy(optimal_prey)
optimal_pred_stack = np.copy(optimal_pred)

base = 30
params_ext['resource'] = base
#print(an_sol.optimal_behavior_trajectories(np.array([0.27757345235331043, 0.29627977517967535, 0.1416850251665414 ]), params_ext))

sol_3 = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x0 = np.array([0.9500123,  0.4147732,  0.01282899 ]), method = 'hybr')
print("Does it work here?")
#sol_static = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, opt_pred = False, opt_prey = False), x0 = np.array([2/3*base, 1.5, 1.5]), method = 'hybr')

h = 0.00005
jac_3 = jacobian_calculator(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), sol_3.x, h)
print(sol_3.message, sol_3.x, "Sol 3")
#print(sol_static.x)
print(np.linalg.eig(jac_3)[0], "Jac3")




list_of_cbars = np.linspace(base-total_range, base, one_dim_its)
parms_list = [params_ext]*one_dim_its
equilibria = np.zeros((one_dim_its, 3))
eigen_values = np.zeros((one_dim_its, 4))
flows = np.zeros((one_dim_its, 6))
optimal_strategies = np.zeros((one_dim_its,2))

optimal_strategies_stack = np.zeros((one_dim_its, 2))

x_ext = sol_3.x
x_ext_stack = np.copy(x_ext)

for i in range(one_dim_its):
#    print(base, step_size, "Base, Stepsize")
    params_ext['resource'] = base - i*step_size #- step_size * i
    #parms_list[i]['resource'] = list_of_cbars[i]
    equilibria[i] = static_eq_calc(params_ext)

    sol_temp = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x0=x_ext, method='hybr')
    x_ext = sol_temp.x
    sol_temp_stackelberg = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, nash = False), x0=x_ext_stack, method='hybr')
    x_ext_stack = sol_temp_stackelberg.x
    print(sol_temp.message, base - i*step_size)



    tc, tp = nash_eq_find(x_ext, params_ext) #strat_finder(x_ext, params_ext)
    print(tp)
    optimal_strategies[i,0] = tc[0]
    optimal_strategies[i,1] = tp[0]
    dynamic_equilibria[i] = x_ext

    tc_s, tp_s = strat_finder(x_ext_stack, params_ext)
    optimal_strategies_stack[i,0] = tc_s
    optimal_strategies_stack[i,1] = tp_s
    dynamic_equilibria_stack[i] = x_ext_stack

    flows[i,0:3] = an_sol.flux_calculator(x_ext[0], x_ext[1], x_ext[2], tc, tp, params_ext).reshape((3,))
    flows[i, 3:] = an_sol.flux_calculator(equilibria[i,0], equilibria[i,1], equilibria[i,2], 1, 1, params_ext).reshape((3,))


    jac0 = jacobian_calculator(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, opt_prey = False, opt_pred = False), equilibria[i], h)
    jac0[np.isnan(jac0)] = 0
    jac1 = jacobian_calculator(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x_ext, h)
    jac1[np.isnan(jac1)] = 0

    eigen_values[i, 0] = max(np.linalg.eig(jac0)[0])
    eigen_values[i, 1] = max(np.linalg.eig(jac1)[0])

equilibria = equilibria[::-1]
dynamic_equilibria = dynamic_equilibria[::-1]
eigen_values = eigen_values[::-1]
flows = flows[::-1]
dynamic_equilibria_stack = dynamic_equilibria_stack[::-1]
optimal_strategies_stack = optimal_strategies_stack[::-1]
optimal_strategies = optimal_strategies[::-1]

opponent_strats = np.linspace(0.01,1,100)

print(epsn, params_ext['nu0'], mass_vector[2])

equilibria = np.log(equilibria)
dynamic_equilibria = np.log(dynamic_equilibria)
dynamic_equilibria_stack = np.log(dynamic_equilibria_stack)
optimal_prey = np.log(optimal_prey)
optimal_pred = np.log(optimal_pred)



plt.figure()
plt.plot(list_of_cbars, equilibria[:,1], label = 'Static consumer')
plt.plot(list_of_cbars, dynamic_equilibria[:, 1], label = 'Nash consumer')

plt.plot(list_of_cbars, dynamic_equilibria_stack[:, 1], label = 'Stackelberg consumer')

plt.xlabel('Maximmal resource biomass in $m_p/m^3$')
plt.ylabel('Log consumer biomass in $m_p/m^3$')
plt.legend(loc  = 'upper left')
plt.savefig('Bifurcation_plot_dynamic.png')


plt.figure()
plt.plot(list_of_cbars, equilibria[:,2], label = 'Static predator')
plt.plot(list_of_cbars, dynamic_equilibria[:, 2], label = 'Nash predator')
plt.plot(list_of_cbars, dynamic_equilibria_stack[:, 2], label = 'Stackelberg predator')

plt.xlabel('Maximal resource biomass in $m_p/m^3$')
plt.ylabel('Log predator biomass in $m_p/m^3$')
plt.legend(loc  = 'upper left')
plt.savefig('Bifurcation_plot_dynamic_predator.png')


plt.figure()
plt.plot(list_of_cbars, np.exp(eigen_values[:,0]), label = 'Static model')
plt.plot(list_of_cbars, np.exp(eigen_values[:,1]), label = 'Nash model')
plt.ylabel('Exp of dominating eigenvalue')
plt.xlabel("Max resource in $m_p/m^3$")
plt.legend(loc ='upper left')
plt.savefig("Eigenvalues_compare.png")

plt.figure()
plt.plot(list_of_cbars, flows[:,0]/flows[:,3], label = 'Ratio of flow from 0 to 1, Nash/Static')
plt.plot(list_of_cbars, flows[:,1]/flows[:,4], label = 'Ratio of flow from 1 to 2, Nash/Static')
plt.plot(list_of_cbars, flows[:,2]/flows[:,5], label = 'Ratio of flow from 2 to n, Nash/Static')

plt.xlabel("Max resource in $m_p/m^3$")
plt.ylabel("$m_p/(m^3 \cdot day)$")
plt.legend(loc = 'center left')


plt.savefig("Flows_Compare.png")


plt.figure()
plt.plot(list_of_cbars, optimal_strategies[:,0], label = 'Nash Consumer')
plt.plot(list_of_cbars, optimal_strategies[:,1], label = 'Nash Predator')

plt.plot(list_of_cbars, optimal_strategies_stack[:,0], label = 'Stackelberg Consumer')
plt.plot(list_of_cbars, optimal_strategies_stack[:,1], label = 'Stackelberg Predator')

plt.ylabel("Activity level")
plt.xlabel("$m_p/(m^3)$")
plt.legend(loc = 'upper right')

plt.savefig("Strategies.png")

plt.figure()
plt.title("Functional response of consumer C at steady state")
plt.plot(list_of_cbars, params_ext['cmax']*np.exp(dynamic_equilibria[:,0])/(np.exp(dynamic_equilibria[:,0])+params_ext['nu0']), 'x', label = "Consumer functional response, non-optimal",alpha=0.5)
plt.plot(list_of_cbars, params_ext['cmax']*optimal_strategies[:,0]*np.exp(dynamic_equilibria[:,0])/(optimal_strategies[:,0]*np.exp(dynamic_equilibria[:,0])+params_ext['nu0']), 'x', label = "Consumer functional response, optimal",alpha=0.5)
plt.xlabel("Max resource in $m_p/m^3$ ")
plt.ylabel("Functional response consumer")
plt.legend(loc='center left')
plt.savefig("Functional_response_consumer_resource_load.png")


plt.figure()
plt.title("Functional response of predator, P " + str(np.round(sol[2,-1], 2)) + " R " + str(np.round(sol[0, -1],2)))
plt.plot(np.exp(dynamic_equilibria[:,0]), params_ext['cp']*np.exp(dynamic_equilibria[:,1])/(np.exp(dynamic_equilibria[:,1])+params_ext['nu0']), 'x', label = "Predator functional response, non-optimal",alpha=0.5)
plt.plot(np.exp(dynamic_equilibria[:,0]), params_ext['cp']*optimal_strategies[:,0]*optimal_strategies[:,1]*np.exp(dynamic_equilibria[:,1])/(optimal_strategies[:,0]*optimal_strategies[:,1]*np.exp(dynamic_equilibria[:,1])+params_ext['nu0']), 'x', label = "Predator functional response, optimal",alpha=0.5)
plt.xlabel("Prey")
plt.ylabel("Functional response")
plt.legend(loc = 'center left')
plt.savefig("Functional_response_predator_resource_load.png")
