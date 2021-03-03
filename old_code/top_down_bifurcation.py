from common_functions import *
import analytical_numerical_solution as an_sol
import semi_impl_eul as num_sol
import matplotlib.pyplot as plt

configuration = {'verbose' : False, 'quadratic' : True, 'metabolic_cost' : False}


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


mass_vector = np.array([1, 1, 100])

one_dim_its = 500

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

sol_3 = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x0 = np.array([0.9500123,  0.4147732,  0.01282899 ]), method = 'hybr')
print("Does it work here?")
#sol_static = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, opt_pred = False, opt_prey = False), x0 = np.array([2/3*base, 1.5, 1.5]), method = 'hybr')

h = 0.00005
jac_3 = jacobian_calculator(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), sol_3.x, h)
print(sol_3.message, sol_3.x, "Sol 3")
#print(sol_static.x)
print(np.linalg.eig(jac_3)[0], "Jac3")




parms_list = [params_ext]*one_dim_its
equilibria = np.zeros((one_dim_its, 3))
eigen_values = np.zeros((one_dim_its, 4))
flows = np.zeros((one_dim_its, 6))
optimal_strategies = np.zeros((one_dim_its,2))

optimal_strategies_stack = np.zeros((one_dim_its, 2))
dynamic_equilibria = np.zeros((one_dim_its, 3))
dynamic_equilibria_stack = np.copy(dynamic_equilibria)

x_ext = sol_3.x
x_ext_stack = np.copy(x_ext)

top_down = np.linspace(0.01, 1, one_dim_its)

params_ext['resource'] = 16
for i in range(one_dim_its):
    params_ext['phi0'] = top_down[-(i+1)]*phi0
    equilibria[i] = static_eq_calc(params_ext)
    sol_temp = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext), x0=x_ext, method='hybr')
    x_ext = sol_temp.x
    sol_temp_stackelberg = optm.root(lambda y: an_sol.optimal_behavior_trajectories(y, params_ext, nash = False), x0=x_ext_stack, method='hybr')
    x_ext_stack = sol_temp_stackelberg.x

    tc, tp = nash_eq_find(x_ext, params_ext) #strat_finder(x_ext, params_ext)
    optimal_strategies[i,0] = tc[0]
    optimal_strategies[i,1] = tp[0]
    dynamic_equilibria[i] = x_ext

    tc_s, tp_s = strat_finder(x_ext_stack, params_ext)
    optimal_strategies_stack[i,0] = tc_s
    optimal_strategies_stack[i,1] = tp_s
    dynamic_equilibria_stack[i] = x_ext_stack

    flows[i,0:3] = an_sol.flux_calculator(x_ext[0], x_ext[1], x_ext[2], tc, tp, params_ext).reshape((3,))
    flows[i, 3:] = an_sol.flux_calculator(equilibria[i,0], equilibria[i,1], equilibria[i,2], 1, 1, params_ext).reshape((3,))


equilibria = equilibria[::-1]
dynamic_equilibria = dynamic_equilibria[::-1]
flows = flows[::-1]
dynamic_equilibria_stack = dynamic_equilibria_stack[::-1]
optimal_strategies_stack = optimal_strategies_stack[::-1]
optimal_strategies = optimal_strategies[::-1]


print(epsn, params_ext['nu0'], mass_vector[2])

equilibria = np.log(equilibria)
dynamic_equilibria = np.log(dynamic_equilibria)
dynamic_equilibria_stack = np.log(dynamic_equilibria_stack)

plt.figure()
plt.plot(top_down, equilibria[:,1], label = 'Static consumer')
plt.plot(top_down, dynamic_equilibria[:, 1], label = 'Nash consumer')

plt.plot(top_down, dynamic_equilibria_stack[:, 1], label = 'Stackelberg consumer')

plt.xlabel('Maximmal resource biomass in $m_p/m^3$')
plt.ylabel('Log consumer biomass in $m_p/m^3$')
plt.legend(loc  = 'upper left')
plt.savefig('Bifurcation_plot_dynamic_TD.png')


plt.figure()
plt.plot(top_down, equilibria[:,2], label = 'Static predator')
plt.plot(top_down, dynamic_equilibria[:, 2], label = 'Nash predator')
plt.plot(top_down, dynamic_equilibria_stack[:, 2], label = 'Stackelberg predator')

plt.xlabel('Maximal resource biomass in $m_p/m^3$')
plt.ylabel('Log predator biomass in $m_p/m^3$')
plt.legend(loc  = 'upper left')
plt.savefig('Bifurcation_plot_dynamic_predator_TD.png')


plt.figure()
plt.plot(top_down, np.exp(eigen_values[:,0]), label = 'Static model')
plt.plot(top_down, np.exp(eigen_values[:,1]), label = 'Nash model')
plt.ylabel('Exp of dominating eigenvalue')
plt.xlabel("Max resource in $m_p/m^3$")
plt.legend(loc ='upper left')
plt.savefig("Eigenvalues_compare_TD.png")

plt.figure()
plt.plot(top_down, flows[:,0]/flows[:,3], label = 'Ratio of flow from 0 to 1, Nash/Static')
plt.plot(top_down, flows[:,1]/flows[:,4], label = 'Ratio of flow from 1 to 2, Nash/Static')
plt.plot(top_down, flows[:,2]/flows[:,5], label = 'Ratio of flow from 2 to n, Nash/Static')

plt.xlabel("Max resource in $m_p/m^3$")
plt.ylabel("$m_p/(m^3 \cdot day)$")
plt.legend(loc = 'center left')


plt.savefig("Flows_Compare_TD.png")


plt.figure()
plt.plot(top_down, optimal_strategies[:,0], label = 'Nash Consumer')
plt.plot(top_down, optimal_strategies[:,1], label = 'Nash Predator')

plt.plot(top_down, optimal_strategies_stack[:,0], label = 'Stackelberg Consumer')
plt.plot(top_down, optimal_strategies_stack[:,1], label = 'Stackelberg Predator')

plt.ylabel("Activity level")
plt.xlabel("$m_p/(m^3)$")
plt.legend(loc = 'upper right')

plt.savefig("Strategies_TD.png")