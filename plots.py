from common_functions import *
import analytical_numerical_solution as an_sol
import semi_impl_eul as num_sol
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

settings = {'gen_data': False, 'plot': True}

mass_vector = np.array([1, 1, 100])

cost_of_living, nu, growth_max, lam = parameter_calculator_mass(mass_vector, v = 0.1)
base = 12
phi1 = cost_of_living[1]#*2/3 #/3 #*2#/2#*5
phi0 = cost_of_living[1]#*4/3 #The problem is that nash equiibrium does not exist in the low levels...
eps = 0.7
epsn = 0.7

cmax, cp = growth_max
mu0 = 0 # cost_of_living[0] #/3 #*2#/2#*5 #*60 #/2
mu1 = cost_of_living[0]#/3 #*2 #*2#/2#*5 #*2 #*10 #/2
nu0 = nu[0] #nu
nu1 = nu[1] #nu

if settings['gen_data'] is True:

    params_ext = {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
              'resource': base, 'lam':lam, 'nu0':nu0, 'nu1': nu1}
    init = np.array([0.9500123,  0.4147732,  0.01282899 ])

    its = 300
    reverse = True
    start = 3
    stop = 30
    x_axis_res = np.linspace(start, stop, its)

    nash_GM_res, strat_nash_GM_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse)
    nash_Gill_res, strat_nash_Gill_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, Gill = True)
    stack_GM_res, strat_stack_GM_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, nash = False)
    stack_Gill_res, strat_stack_Gill_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, Gill = True, nash = False)



    start = 0.1*phi0
    stop = phi0
    x_axis_phi0 = np.linspace(start, stop, its)

    nash_GM_phi0, strat_nash_GM_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, type = 'phi0')
    nash_Gill_phi0, strat_nash_Gill_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, type = 'phi0', Gill = True)
    stack_GM_phi0, strat_stack_GM_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, type = 'phi0', nash = False)
    stack_Gill_phi0, strat_stack_Gill_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, type = 'phi0', Gill = True, nash = False)

    with open('bifurcation_data.npy', 'wb') as f:
        np.save(f, x_axis_res)

        np.save(f, nash_GM_res)
        np.save(f, strat_nash_GM_res)

        np.save(f, nash_Gill_res)
        np.save(f, strat_nash_Gill_res)

        np.save(f, stack_GM_res)
        np.save(f, strat_stack_GM_res)

        np.save(f, stack_Gill_res)
        np.save(f, strat_stack_Gill_res)

        np.save(f, x_axis_phi0)
        np.save(f, nash_GM_phi0)
        np.save(f, strat_nash_GM_phi0)

        np.save(f, nash_Gill_phi0)
        np.save(f, strat_nash_Gill_phi0)

        np.save(f, stack_GM_phi0)
        np.save(f, strat_stack_GM_phi0)

        np.save(f, stack_Gill_phi0)
        np.save(f, strat_stack_Gill_phi0)

elif settings['gen_data'] is False:
    with open('test.npy', 'rb') as f:  # Save time by not generating all the data every time.

        x_axis_res = np.load(f)

        nash_GM_res = np.load(f)
        strat_nash_GM_res = np.load(f)

        nash_Gill_res = np.load(f)
        strat_nash_Gill_res = np.load(f)

        stack_GM_res = np.load(f)
        strat_stack_GM_res = np.load(f)

        stack_Gill_res = np.load(f)
        strat_stack_Gill_res = np.load(f)

        x_axis_phi0= np.load(f)

        nash_GM_phi0 = np.load(f)
        strat_nash_GM_phi0 = np.load(f)

        nash_Gill_phi0 = np.load(f)
        strat_nash_Gill_phi0 = np.load(f)

        stack_GM_phi0 = np.load(f)
        strat_stack_GM_phi0 = np.load(f)

        stack_Gill_phi0 = np.load(f)
        strat_stack_Gill_phi0 = np.load(f)


if settings['plot'] is True:
    # Creates just a figure and only one subplot
    fig, ax = plt.subplots()
    ax.plot(x_axis_res, y)
    ax.set_title('Simple plot')

    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Creates four polar axes, and accesses them through the returned array
    fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
    axes[0, 0].plot(x, y)
    axes[1, 1].scatter(x, y)

    # Share a X axis with each column of subplots
    plt.subplots(2, 2, sharex='col')

    # Share a Y axis with each row of subplots
    plt.subplots(2, 2, sharey='row')