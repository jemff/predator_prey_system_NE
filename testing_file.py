from common_functions import *
import analytical_numerical_solution as an_sol
import semi_impl_eul as num_sol
import numpy as np
import matplotlib.pyplot as plt


mass_vector = np.array([1, 1, 100])


cost_of_living, nu, growth_max, lam = parameter_calculator_mass(mass_vector)
print(nu, "nu", growth_max)
base = 3
phi0= cost_of_living[1]/2 #*5
phi1 = cost_of_living[1]/2 #/2#*5
eps = 0.7
epsn = 0.7

cmax, cp = growth_max
mu0 = 0 # cost_of_living[0] #/3 #*2#/2#*5 #*60 #/2
mu1 = cost_of_living[0]#/3 #*2 #*2#/2#*5 #*2 #*10 #/2
nu0 = nu[0] #nu
nu1 = nu[1] #nu

params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
          'resource': base, 'lam':lam, 'nu0':nu0, 'nu1': nu1}

#print(nash_eq_find(static_eq_calc(params_ext), params_ext))

#    return opt_taun_analytical(value, opt_taup_find(value, x, params_ext), 10, 0.7, 0.54545454)[0] - x

#NE = optm.root(f, x0 = 0.5)
#print(NE.x, opt_taup_find(value, NE.x, params_ext)[0])
#def instantaneous_intertrophic_flux(state, params):

t_end = 5
init = static_eq_calc(params_ext) #np.array([0.5383403,  0.23503815, 0.00726976]) #np.array([5.753812957581866, 5.490194692112937, 1.626801718856221])#
print(init)
tim, sol, flux, strat = num_sol.semi_implicit_euler(t_end, init, 0.0005, lambda t, y, tn, tp:
num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=True, opt_pred=True)
tim, sol_2, flux_2, strat_2 = num_sol.semi_implicit_euler(t_end, init, 0.0005, lambda t, y, tn, tp:
num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, seasons = False), params_ext, opt_prey=False, opt_pred=False)

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

plt.figure()

plt.plot(tim[1:], strat[0,1:], 'x', label = "Prey foraging intensity",alpha=0.5)
plt.plot(tim[1:], strat[1,1:], 'x', label = "Predator foraging intensity", alpha=0.5)
plt.xlabel("Months")
plt.ylabel("Intensity")
plt.legend(loc = 'center left')
plt.savefig("Indsvingning_strat.png")



fidelity = 500
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
plt.xlabel("Prey")
plt.ylabel("Functional response")
plt.legend(loc = 'center left')
plt.savefig("Functional_response_predator.png")

plt.figure()
plt.title("Functional response of consumer, C " + str(np.round(sol[1,-1], 2)) + " P " + str(np.round(sol[1, -1],2)))
plt.plot(resource_variation, params_ext['cmax']*resource_variation/(resource_variation+params_ext['nu0']), 'x', label = "Consumer functional response, non-optimal",alpha=0.5)
plt.plot(resource_variation, params_ext['cmax']*frc[:,0]*resource_variation/(frc[:,0]*resource_variation+params_ext['nu0']), 'x', label = "Consumer functional response, optimal",alpha=0.5)
plt.xlabel("Resource")
plt.ylabel("Functional response consumer")
plt.legend(loc = 'center left')
plt.savefig("Functional_response_consumer.png")


print(opponent_strats[(opt_taup_find(sol[:,-1], opt_taun_analytical(sol[:,-1], opponent_strats, mass_vector[2], epsn, params_ext['nu0']), params_ext) - opponent_strats) < 0])

print(opt_taup_find(sol[:,-1], opt_taun_analytical(sol[:,-1], strat[1,-1], mass_vector[2], epsn, params_ext['nu0']), params_ext), strat[1,-1])
print(opt_taun_analytical(sol[:,-1], opt_taup_find(sol[:,-1], strat[0,-1], params_ext), 100, params_ext['eps'], params_ext['nu0']), strat[0,-1])
print(nash_eq_find(sol[:,-1], params_ext, opt_prey = True, opt_pred = True), working_nash_eq_find(sol[:,-1], params_ext))