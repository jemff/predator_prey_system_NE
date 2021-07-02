from common_functions import *
import analytical_numerical_solution as an_sol
import semi_impl_eul as num_sol
import matplotlib.pyplot as plt
import copy as copy
from multiprocessing import Pool


"""
This file generates all data and plots for the article "Lower productivity and higher populations: The influence of optimal behavior in a tri-trophic system"

"""

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size = 8)
# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150), 
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148), 
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199), 
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

settings = {'simulation': False, 'sensitivity': False, 'func_dyn': False, 'flux_stat_func': False, 'heatmap_up': False, 'heatmap_down': False, 'plot': True, 'linear' : False}
"""
The boolean settings above indicate whether to generate the data. Term by term:
    'simulation': Runs the time simulation and generates the data
    'sensitivity': Performs the sensitivity analysis for top-down pressure and bottom-up pressure
    'func_dyn': Generates the emergent functional response data
    'flux_stat_func': Generates the data for the ecosystem production
    'heatmap_up': Generates data for heatmaps based on the sensitivity data,
    'heatmap_down': Deprecated; Originally to split heatmap generation into 2 stages to save time
    'plot': Generate all plots
    'linear': Indicates the predator loss from staying in the arena, default is quadratic. 
"""

mass_vector = np.array([1, 1, 100])

cost_of_living, nu, growth_max, lam = parameter_calculator_mass(mass_vector, v = 0.08) #Metabolic parameter calculation
base = 1
phi1 = cost_of_living[1] #Predator metabolic loss
phi0 = 0.5*cost_of_living[1] #Predator loss from staying in the foraging arena, note that this parameter is varied.
eps = 0.3 #Conversion factor
epsn = 0.3

cmax, cp = growth_max #Maximal growth rates
mu0 = 0 #Consumer loss from staying in the arena
mu1 = cost_of_living[0] #Consumer metabolic loss
nu0 = nu[0] #nu
nu1 = nu[1] #nu

its = 30 #The fineness of the bottom-up pressure grid
its_mort = 30 #The fineness of the top-down pressure grid
fidelity = 20 #Number of lines for emergent functional response
pred_var_fid = 6


params_ext = {'cmax': cmax, 'mu0': mu0, 'mu1': mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0': phi0, 'phi1': phi1,
              'resource': base, 'lam': lam, 'nu0': nu0, 'nu1': nu1}


print(params_ext)
if settings['sensitivity'] is True:
    # np.array([1.29649220e+01, 5.55740326e-01, 1.11775631e-03])#0.5*np.array([0.9500123,  0.4147732,  0.01282899 ])
    params_ext['phi0'] = 0.5*phi1
    init = np.array([0.54630611, 0.17370847, 0.0017048])#static_eq_calc(params_ext)
    reverse = True
    start = 0.2
    stop = 2
    x_axis_res = np.linspace(start, stop, its)
    print(params_ext)
    nash_GM_res, strat_nash_GM_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, linear = settings['linear']) #Generating the bottom-up variation based on a first-order continuation procedure

    params_ext['resource'] = 1
    start = 0.3 * phi1
    stop = 3 * phi1

    init = 0.25* np.array([2.18788587, 0.17012115, 0.00327292])
    x_axis_phi0 = np.linspace(start, stop, its_mort)

    nash_GM_phi0, strat_nash_GM_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its_mort, reverse = False, type = 'phi0', linear = settings['linear']) #Generating the top-down variation based on a first-order continuation procedure

    with open('bifurcation_data.npy', 'wb') as f:
        np.save(f, x_axis_res)
        np.save(f, nash_GM_res)
        np.save(f, strat_nash_GM_res)

        np.save(f, x_axis_phi0)
        np.save(f, nash_GM_phi0)
        np.save(f, strat_nash_GM_phi0)

#    params_ext['resource'] = 3
#    params_ext['phi0'] = 0.3*phi1


elif settings['sensitivity'] is False:  # Split into simulation, sensisitivty, derived data, heat map.
    with open('bifurcation_data.npy', 'rb') as f:  # Save time by not generating all the data every time.

        x_axis_res = np.load(f)

        nash_GM_res = np.load(f)
        strat_nash_GM_res = np.load(f)

        x_axis_phi0 = np.load(f)

        nash_GM_phi0 = np.load(f)
        strat_nash_GM_phi0 = np.load(f)
    #     results = np.load(f)

if settings['simulation'] is True:
    params_ext['resource'] = 0.5
    params_ext['phi0'] = 0.6*phi1
    t_end = 96
    print(params_ext, "simulation parameters")
    init = static_eq_calc(params_ext)*2 #The function static_eq_calc calculates the equilibrium of the system with constant behavior
    print(init)
    init[-1] = init[-1]    # np.array([0.5383403,  0.23503815, 0.00726976]) #np.array([5.753812957581866, 5.490194692112937, 1.626801718856221])#
    # params_ext['resource'] = 16 This si for the low-resource paradigm
    tim, sol, strat = num_sol.semi_implicit_euler(t_end, init, 0.01, lambda t, y, tn, tp:
    num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, linear = settings['linear']), params_ext, opt_prey=True,
                                                  opt_pred=True, linear = settings['linear'])
    tim, sol_2, strat_2 = num_sol.semi_implicit_euler(t_end, init, 0.01, lambda t, y, tn, tp:
    num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp), params_ext,
                                                      opt_prey=False, opt_pred=False)

    with open('simulate_data.npy', 'wb') as g:
        np.save(g, tim)
        np.save(g, sol)
        np.save(g, strat)

        np.save(g, sol_2)
        np.save(g, strat_2)

    params_ext['resource'] = 3
    params_ext['phi0'] = 0.5*phi0

elif settings['simulation'] is False:
    with open('simulate_data.npy', 'rb') as g:  # Save time by not generating all the data every time.
        tim = np.load(g)
        sol = np.load(g)
        strat = np.load(g)

        sol_2 = np.load(g)
        strat_2 = np.load(g)

if settings['heatmap_up'] is True:
    its_heat = int(its) # int(its/2)
    input_data = []
    for i in range(its_mort):
        """
        We create a dictionary pairing initial values and iteration numbers, for multithreading of the grid generation.
        """

        par_t = copy.deepcopy(params_ext)
        par_t['phi0'] = x_axis_phi0[i]
        input_data.append({'values': nash_GM_phi0[i], 'parameters': par_t, 'strat': strat_nash_GM_phi0[i], 'iteration': i})

    start = 1 #x_axis_res[0]
    stop = x_axis_res[-1]
    reverse = False

    agents = 7 #The number of simultaneous processes, can be adjusted to whatever is optimal for the current computing setup.


    def temp_func(ivp, its = its_heat, start = start, stop = stop, settings = settings, reverse = reverse):
        """
        This function wraps the continuation logic in order to perform multithreaded continuation,
         which allows the simultaneous calculation of the grid lines for the heatmap, speeding up the calculation massively.

        :param ivp: Initial value for this specific continuation
        :param its: Grid fineness
        :param start: Start of bottom-up variation
        :param stop: Stop of bottom-up variation
        :param settings: Specifying the shape of the predator loss function
        :param reverse: Whether to perform the continuation from the left or the right.
        :return: Returns strategies and population levels
        """

        outputs = np.zeros((its, 5))
        values, strategies = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2,
                                     ivp['values'], ivp['parameters'], start, stop, its,
                                     reverse=reverse, type='resource',
                                     linear=settings['linear'], strat=ivp['strat'], verbose = False)
        outputs[:, 0:3] = values
        outputs[:, 3:] = strategies
        print(ivp['iteration'], ivp['parameters']['phi0'], ivp['strat'], strat_nash_GM_phi0[0])

        return outputs


    with Pool(processes = agents) as pool:
        #Multiprocess data generation for the heatmap
        grid_data = pool.map(temp_func, input_data)
    #print(combined_strat_finder(input_data[0]['parameters'], [1.04194356, 0.21399234, 0.00927935], x0 = np.array([0.5, 0.5]) ))
    #grid_data = temp_func(input_data[0])
    #print(grid_data[0], strat_nash_GM_phi0[0], nash_GM_phi0[0])
    grid_data = np.array(grid_data)


    start = x_axis_res[0] #Change to 0.6*phi0, but for now we use phi0 for stability reasons
    stop = 1
    reverse = True
    def temp_func(ivp, its = its_heat, start = start, stop = stop, settings = settings, reverse = reverse):

        outputs = np.zeros((its, 5))
        values, strategies = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2,
                                     ivp['values'], ivp['parameters'], start, stop, its,
                                     reverse=reverse, type='resource',
                                     linear=settings['linear'], strat=ivp['strat'], verbose=False)
        outputs[:, 0:3] = values
        outputs[:, 3:] = strategies
        print(ivp['iteration'], start, stop)
        return outputs

    #for i in range(int(its/60)):
    #    print(i*60, input_data[i*60])
    #    temp_func(input_data[i*60])

    with Pool(processes = agents) as pool:
        grid_data_down = pool.map(temp_func, input_data)

    grid_data_down = np.array(grid_data_down)

    grid_data = np.concatenate([grid_data_down.T, grid_data.T]).T
    with open('heatmap_data.npy', 'wb') as g:
        np.save(g, grid_data)

elif settings['heatmap_up'] is False:
    with open('heatmap_data.npy', 'rb') as g:  # Save time by not generating all the data every time.
        grid_data = np.load(g)


if settings['heatmap_down'] is True:
    its_heat = int(its)  #int(its/2)
    input_data = []
    for i in range(its_mort):
        par_t = copy.deepcopy(params_ext)
        par_t['phi0'] = x_axis_phi0[i]
        input_data.append({'values': nash_GM_phi0[i], 'parameters': par_t, 'strat': strat_nash_GM_phi0[i], 'iteration': i})

    start = x_axis_res[0] #Change to 0.6*phi0, but for now we use phi0 for stability reasons
    stop = 1
    reverse = True

    agents = 7
    def temp_func(ivp, its = its_heat, start = start, stop = stop, settings = settings, reverse = reverse):

        outputs = np.zeros((its, 5))
        values, strategies = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2,
                                     ivp['values'], ivp['parameters'], start, stop, its,
                                     reverse=reverse, type='resource',
                                     linear=settings['linear'], strat=ivp['strat'])
        outputs[:, 0:3] = values
        outputs[:, 3:] = strategies
        print(ivp['iteration'], start, stop)
        return outputs

    #for i in range(int(its/60)):
    #    print(i*60, input_data[i*60])
    #    temp_func(input_data[i*60])

    with Pool(processes = agents) as pool:
        grid_data_down = pool.map(temp_func, input_data)

    grid_data_down = np.array(grid_data)
    with open('heatmap_down_data.npy', 'wb') as g:
        np.save(g, grid_data_down)

elif settings['heatmap_down'] is False:
    with open('heatmap_down_data.npy', 'rb') as g:  # Save time by not generating all the data every time.
        grid_data_down = np.load(g)


if settings['func_dyn'] is True:
    """
        This part generates the emergent functional response, we pick a point of resources in from the sensitivity analysis as res_m and prey_m, generating the varying predator levels as pred_m and the varying top-predation as xi_var. 
        The code uses zeroth order continuation to generate the Nash equilibria, as we quickly move away from the area where iterated best response works well. 
    """

    res_m = nash_GM_res[int(its/3), 0]
    prey_m = nash_GM_res[int(its/3), 1]
    fix_pred = nash_GM_res[int(its/3), 2]

    pred_m = np.linspace(0.75*nash_GM_res[int(its/3), 2], 1.5*nash_GM_res[int(its/3), 2], pred_var_fid + 1)

#    print("Values", nash_GM_res[int(its/2)], strat_nash_GM_res[int(its/2)])

    params_t = copy.deepcopy(params_ext)
    params_t_down = copy.deepcopy(params_ext)

    xi_var = np.linspace(0.01*params_ext['phi0'], 2*params_ext['phi0'], pred_var_fid+1)
    print(params_ext, "func_dyn_pars")
    frp = np.zeros((fidelity + 1, pred_var_fid+1, 2))
    frc = np.zeros((fidelity + 1, pred_var_fid+1, 2))
    resource_variation = np.linspace(0.001*res_m, 2*res_m, fidelity + 1)
    prey_variation = np.linspace(0.001*prey_m, 2*prey_m, fidelity + 1)



    frc[int(fidelity/2), int(pred_var_fid/2)] = combined_strat_finder(params_t, np.array([res_m, prey_variation[int(fidelity/2)], pred_m[int(pred_var_fid / 2)]]), x0=np.array([0.5, 0.5]), linear=settings['linear'])
    frp[int(fidelity/2), int(pred_var_fid/2)] = combined_strat_finder(params_t, np.array([res_m, prey_variation[int(fidelity/2)], fix_pred]), x0=np.array([0.5, 0.5]), linear=settings['linear'])

    for k in range(int(pred_var_fid / 2)):
        params_t['phi0'] = xi_var[int(pred_var_fid / 2) + k + 1]
        params_t_down['phi0'] = xi_var[int(pred_var_fid / 2) - k - 1]

        frp[int(fidelity/2), int(pred_var_fid / 2) + k + 1] = combined_strat_finder(params_t, np.array([res_m, prey_variation[int(fidelity/2)], fix_pred]), x0=frp[int(fidelity/2), int(pred_var_fid / 2) + k], linear=settings['linear'])
        frc[int(fidelity/2), int(pred_var_fid / 2) + k + 1] = combined_strat_finder(params_ext, np.array([resource_variation[int(fidelity/2)], prey_m, pred_m[int(pred_var_fid / 2) + k + 1]])) #, x0=frc[int(fidelity/2), int(pred_var_fid / 2) + k ],       linear=settings['linear'])

        frp[int(fidelity/2), int(pred_var_fid / 2) - 1 - k] = combined_strat_finder(params_t_down, np.array([res_m, prey_variation[int(fidelity/2)], fix_pred]), x0=frp[int(fidelity/2), int(pred_var_fid / 2) - k], linear=settings['linear'])
        frc[int(fidelity/2), int(pred_var_fid / 2) - 1 - k] = combined_strat_finder(params_ext, np.array([resource_variation[int(fidelity/2)], prey_m, pred_m[int(pred_var_fid / 2) - k]]))  #, x0=frc[int(fidelity/2), int(pred_var_fid / 2) - k], linear=settings['linear'])

    print(frc[int(fidelity/2),:,0], "FRC DEBUG")
    for i in range(1, int(fidelity/2)+1):
        for k in range(int(pred_var_fid / 2) + 1):
            params_t['phi0'] = xi_var[int(pred_var_fid / 2) + k]
            params_t_down['phi0'] = xi_var[int(pred_var_fid / 2) - k]

            frp[int(fidelity/2) - i + 1, int(pred_var_fid/2) + k] = combined_strat_finder(params_t, np.array([res_m, prey_variation[int(fidelity/2) - i + 1], fix_pred]),
                                                                                          x0 = frp[int(fidelity/2) - i + 2, int(pred_var_fid/2) + k], linear = settings['linear'])
            frp[int(fidelity/2) + i, int(pred_var_fid/2) + k] = combined_strat_finder(params_t, np.array([res_m, prey_variation[int(fidelity/2) + i], fix_pred]),
                                                                                      x0 = frp[int(fidelity/2) + i - 1, int(pred_var_fid/2) + k], linear = settings['linear'])
            frp[int(fidelity/2) - i + 1, int(pred_var_fid/2) - k] = combined_strat_finder(params_t_down, np.array([res_m, prey_variation[int(fidelity/2) - i + 1], fix_pred]),
                                                                                          x0 = frp[int(fidelity/2) - i + 2, int(pred_var_fid/2) - k], linear = settings['linear'])
            frp[int(fidelity/2) + i, int(pred_var_fid/2) - k] = combined_strat_finder(params_t_down, np.array([res_m, prey_variation[int(fidelity/2) + i], fix_pred]),
                                                                                      x0 = frp[int(fidelity/2) + i - 1, int(pred_var_fid/2) - k], linear = settings['linear'])

    for k in range(int(pred_var_fid / 2) + 1):
        for i in range(0, int(fidelity / 2) + 1):
            if i is 0:
                print(frc[int(fidelity/2) - i, int(pred_var_fid/2) - k ])
            frc[int(fidelity/2) - i + 1, int(pred_var_fid/2) + k] \
                = complementary_nash(np.array([resource_variation[int(fidelity/2) - i + 1], prey_m, pred_m[int(pred_var_fid/2) + k]]), params_ext)#,
                                        #x0 = frc[int(fidelity/2) - i, int(pred_var_fid/2) + k ], linear = settings['linear'])
            frc[int(fidelity/2) - i + 1, int(pred_var_fid/2) - k] = complementary_nash(np.array([resource_variation[int(fidelity/2) - i + 1], prey_m, pred_m[int(pred_var_fid/2) - k]]), params_ext)#,
                                                                                          #x0 = frc[int(fidelity/2) - i, int(pred_var_fid/2) - k ], linear = settings['linear'])

            frc[int(fidelity/2) + i, int(pred_var_fid/2) + k] = complementary_nash(np.array([resource_variation[int(fidelity/2) + i], prey_m, pred_m[int(pred_var_fid/2) + k]]), params_ext)#, x0 = frc[int(fidelity/2) + i , int(pred_var_fid/2) + k ], linear = settings['linear'])
            frc[int(fidelity/2) + i, int(pred_var_fid/2) - k] = complementary_nash(np.array([resource_variation[int(fidelity/2) + i], prey_m, pred_m[int(pred_var_fid/2) - k]]), params_ext)#, x0 = frc[int(fidelity/2) + i, int(pred_var_fid/2) - k ], linear = settings['linear'])


    with open('func_dyn_data.npy', 'wb') as g:
        np.save(g, frp)
        np.save(g, frc)
        np.save(g, resource_variation)
        np.save(g, prey_variation)

elif settings['func_dyn'] is False:
    with open('func_dyn_data.npy', 'rb') as g:  # Save time by not generating all the data every time.
        frp = np.load(g)
        frc = np.load(g)
        resource_variation = np.load(g)
        prey_variation = np.load(g)

if settings['flux_stat_func'] is True:
    """
        The fixed points for the static behavior is generated here, and the production levels are calculated. 
    """

    params_ext['resource'] = 3
    params_ext['phi0'] = 0.3*phi1
    static_values_res = np.zeros((its, 3))
    static_values_phi0 = np.zeros((its_mort, 3))
    flux_nash_GM_phi0 = np.zeros((its_mort, 3))
    flux_static_values_phi0 = np.zeros((its_mort, 3))

    ones = np.repeat(1, its)

    params_temp_res = copy.deepcopy(params_ext)
    params_temp_phi0 = copy.deepcopy(params_ext)
    func_nash_GM_phi0 = np.zeros((its_mort, 2))
    func_static_values_phi0 = np.zeros((its_mort, 2))

    for i in range(its):
        params_temp_res['resource'] = x_axis_res[i]
        static_values_res[i] = static_eq_calc(params_temp_res)
        #print(num_sol.optimal_behavior_trajectories(0.1, static_values_res[i], params_temp_res, taun=1, taup=1), "RES_MOVING")
    for i in range(its_mort):
        params_temp_phi0['phi0'] = x_axis_phi0[i]
        static_values_phi0[i] = static_eq_calc(params_temp_phi0)
        #print(num_sol.optimal_behavior_trajectories(0.1, static_values_phi0[i], params_temp_phi0, taun=1, taup=1), "PHI MOVING")

        #print(static_eq_calc(params_temp_phi0), params_temp_phi0['phi0'], params_temp_phi0['phi1'])
        flux_nash_GM_phi0[i] = an_sol.flux_calculator(nash_GM_phi0[i, 0], nash_GM_phi0[i, 1], nash_GM_phi0[i, 2],
                                                      strat_nash_GM_phi0[i, 0], strat_nash_GM_phi0[i, 1], params_temp_phi0, linear = settings['linear'])

        flux_static_values_phi0[i] = an_sol.flux_calculator(static_values_phi0[i, 0], static_values_phi0[i, 1],
                                                            static_values_phi0[i, 2], 1, 1, params_temp_phi0, linear = settings['linear'])

        func_nash_GM_phi0[i] = an_sol.frp_calc(nash_GM_phi0[i, 0], nash_GM_phi0[i, 1], nash_GM_phi0[i, 2],
                                               strat_nash_GM_phi0[i, 0], strat_nash_GM_phi0[i, 1], params_temp_phi0)
        func_static_values_phi0[i] = an_sol.frp_calc(static_values_phi0[i, 0], static_values_phi0[i, 1],
                                                     static_values_phi0[i, 2], 1, 1, params_temp_phi0)


    flux_nash_GM_res = an_sol.flux_calculator(nash_GM_res[:, 0], nash_GM_res[:, 1], nash_GM_res[:, 2], strat_nash_GM_res[:, 0], strat_nash_GM_res[:, 1], params_ext, linear = settings['linear'])
    flux_static_values_res = an_sol.flux_calculator(static_values_res[:, 0], static_values_res[:, 1], static_values_res[:, 2], ones, ones, params_ext, linear = settings['linear'])

    func_nash_GM_res = an_sol.frp_calc(nash_GM_res[:, 0], nash_GM_res[:, 1], nash_GM_res[:, 2], strat_nash_GM_res[:, 0], strat_nash_GM_res[:, 1], params_ext)
    func_static_values_res = an_sol.frp_calc(static_values_res[:, 0], static_values_res[:, 1], static_values_res[:, 2], ones, ones, params_ext)

    with open('flux_stat_func_data.npy', 'wb') as g:
        np.save(g, flux_nash_GM_res)
        np.save(g, flux_static_values_res)
        np.save(g, func_nash_GM_res)
        np.save(g, func_static_values_res)
        np.save(g, static_values_res)

        np.save(g, flux_nash_GM_phi0)
        np.save(g, flux_static_values_phi0)
        np.save(g, func_nash_GM_phi0)
        np.save(g, func_static_values_phi0)
        np.save(g, static_values_phi0)


elif settings['flux_stat_func'] is False:
    with open('flux_stat_func_data.npy', 'rb') as g:  # Save time by not generating all the data every time.
        flux_nash_GM_res = np.load(g)
        flux_static_values_res = np.load(g)

        func_nash_GM_res = np.load(g)
        func_static_values_res = np.load(g)

        static_values_res = np.load(g)

        flux_nash_GM_phi0 = np.load(g)
        flux_static_values_phi0 = np.load(g)

        func_nash_GM_phi0 = np.load(g)
        func_static_values_phi0 = np.load(g)

        static_values_phi0 = np.load(g)

if settings['plot'] is True:
    """
    All the plots are generated here, with a lot of repeated boilerplate code. This section will fail to run if all the data has not been generated ahead of time. 
    """

#    print(frp[int(fidelity/2), :, 1], frp[0, :, 1], frp[0, :, 0])

    fig, ax = plt.subplots(6, 2, gridspec_kw={'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5]}, sharex='col', sharey = 'row')
    fig.set_size_inches((12/2.54, 14/2.54))

    ax[0, 0].set_ylabel('Resource,\n $m_c$  m$^{-3}$')
    ax[-1, 0].set_xlabel('Carrying capacity $(\overline{R})$ \n $m_c$ m$^{-3}$')


    ax[0, 0].plot(x_axis_res, nash_GM_res[:, 0], color = tableau20[6], linestyle = '-')
    ax[0, 0].plot(x_axis_res, static_values_res[:, 0], color = tableau20[0], linestyle = '-')
#    ax[0,0].text(0, 1.1, "Population levels and behavior (1)", transform=ax[0,0].transAxes)
#    ax[0,1].text(0, 1.1, "(2)", transform=ax[0,1].transAxes)

    ax[0, 0].text(1.05, 0.9, "(a)", transform=ax[0, 0].transAxes)
    ax[1, 0].text(1.05, 0.8, "(b)", transform=ax[1, 0].transAxes)
    ax[2, 0].text(1.05, 0.9, "(c)", transform=ax[2, 0].transAxes)
    ax[3, 0].text(1.05, 0.8, "(d)", transform=ax[3, 0].transAxes)
    ax[4, 0].text(1.05, 0.9, "(e)", transform=ax[4, 0].transAxes)
    ax[5, 0].text(1.05, 0.8, "(f)", transform=ax[5, 0].transAxes)

    ax[0, 1].text(1.05, 0.9, "(g)", transform=ax[0, 1].transAxes)
    ax[1, 1].text(1.05, 0.8, "(h)", transform=ax[1, 1].transAxes)
    ax[2, 1].text(1.05, 0.9, "(i)", transform=ax[2, 1].transAxes)
    ax[3, 1].text(1.05, 0.8, "(j)", transform=ax[3, 1].transAxes)
    ax[4, 1].text(1.05, 0.9, "(k)", transform=ax[4, 1].transAxes)
    ax[5, 1].text(1.05, 0.8, "(l)", transform=ax[5, 1].transAxes)

    ax[1, 0].set_ylabel('$\\tau_c$')

    ax[1, 0].fill_between(x_axis_res, strat_nash_GM_res[:, 0], y2 =0, alpha = 0.5, color = tableau20[6], linestyle = '-')
    ax[1, 0].set_ylim((0, 1))

    ax[2, 0].set_ylabel('Consumer, \n $10^{-1} m_c$ m$^{-3}$')

    ax[2, 0].plot(x_axis_res, 10*nash_GM_res[:, 1], color =  tableau20[6], linestyle = '-')
    ax[2, 0].plot(x_axis_res, 10*static_values_res[:, 1], color = tableau20[0], linestyle = '-')

    ax[3, 0].set_ylabel('$\\tau_p  \\tau_c$')

    ax[3, 0].fill_between(x_axis_res, strat_nash_GM_res[:, 1]*strat_nash_GM_res[:, 0], y2 = 0, alpha = 0.5, color = tableau20[6], linestyle = '-')
    ax[3, 0].set_ylim((0, 1))

    ax[4, 0].set_ylabel('Predator, \n $10^{-3} m_c$ m$^{-3}$')

    ax[4, 0].plot(x_axis_res, 1000*nash_GM_res[:, 2], color = tableau20[6], linestyle = '-')
    ax[4, 0].plot(x_axis_res, 1000*static_values_res[:, 2], color = tableau20[0], linestyle = '-')

    ax[5, 0].set_ylabel('$\\tau_p$')
    ax[5, 0].set_ylim((0, 1))
    ax[5, 0].fill_between(x_axis_res, strat_nash_GM_res[:, 1], y2 = 0, alpha = 0.5, color = tableau20[6], linestyle = '-')

    #    ax[1, 0].set_ylabel('Resource, $m_c m^{-3}$')
    ax[-1, 1].set_xlabel('Top predation pressure ($\\xi$) \n $month^{-1}$')

    ax[0, 1].plot(x_axis_phi0, nash_GM_phi0[:, 0], color=tableau20[6], linestyle='-')
    ax[0, 1].plot(x_axis_phi0, static_values_phi0[:, 0], color=tableau20[0], linestyle='-')

    ax[1, 1].fill_between(x_axis_phi0, strat_nash_GM_phi0[:, 0], y2=0, color=tableau20[6],
                          linestyle='-', alpha=0.5)


    ax[2, 1].plot(x_axis_phi0, nash_GM_phi0[:, 1], color=tableau20[6], linestyle='-')
    ax[2, 1].plot(x_axis_phi0, static_values_phi0[:, 1], color=tableau20[0], linestyle='-')


    ax[3, 1].fill_between(x_axis_phi0, strat_nash_GM_phi0[:, 1] * strat_nash_GM_phi0[:, 0],
                          y2=0, color=tableau20[6], alpha=0.5,
                          linestyle='-')

    ax[4, 1].plot(x_axis_phi0, 10**3*nash_GM_phi0[:, 2], color=tableau20[6], linestyle='-')
    ax[4, 1].plot(x_axis_phi0, 10**3*static_values_phi0[:, 2], color=tableau20[0], linestyle='-')

    ax[5, 1].fill_between(x_axis_phi0, strat_nash_GM_phi0[:, 1], y2=0, alpha=0.5,
                          color=tableau20[6], linestyle='-')

    if settings['linear'] is False:
        plt.savefig('sensitivity.pdf')
    else:
        plt.savefig('sensitivity_linear.pdf')


    fig3, ax3 = plt.subplots(3, 2, sharex='col', sharey = 'row')
    fig3.set_size_inches((12/2.54, 14/2.54))
#    ax3[0,0].text(0, 1.1, "Production, static vs. optimal (1)", transform=ax3[0,0].transAxes)
#    ax3[0,1].text(0, 1.1, "(2)", transform=ax3[0,1].transAxes)

    ax3[0, 0].text(1.05, 0.9, "(a)", transform=ax3[0, 0].transAxes)
    ax3[1, 0].text(1.05, 0.9, "(b)", transform=ax3[1, 0].transAxes)
    ax3[2, 0].text(1.05, 0.9, "(c)", transform=ax3[2, 0].transAxes)

    ax3[0, 1].text(1.05, 0.9, "(d)", transform=ax3[0, 1].transAxes)
    ax3[1, 1].text(1.05, 0.9, "(e)", transform=ax3[1, 1].transAxes)
    ax3[2, 1].text(1.05, 0.9, "(f)", transform=ax3[2, 1].transAxes)

    ax3[0,1].plot(x_axis_phi0, flux_nash_GM_phi0[:, 0], color = tableau20[6], linestyle = '-')
    ax3[0,1].plot(x_axis_phi0, flux_static_values_phi0[:, 0], color = tableau20[0], linestyle = '-')
    ax3[1,1].plot(x_axis_phi0, flux_nash_GM_phi0[:, 1], color = tableau20[6], linestyle = '-')
    ax3[0, 0].set_ylabel('$R\\to C$, \n $m_c$ month$^{-1}$')
    ax3[1, 0].set_ylabel('$C\\to P$, \n $m_c$ month$^{-1}$')
    ax3[2, 0].set_ylabel('$P\\to Top$, \n $m_c$ month$^{-1}$')


    ax3[1,1].plot(x_axis_phi0, flux_static_values_phi0[:, 1], color = tableau20[0], linestyle = '-')
    ax3[2,1].plot(x_axis_phi0, flux_nash_GM_phi0[:, 1], color = tableau20[6], linestyle = '-')
    ax3[2,1].plot(x_axis_phi0, flux_static_values_phi0[:, 1], color = tableau20[0], linestyle = '-')


    ax3[0,0].plot(x_axis_res, flux_nash_GM_res[0], color = tableau20[6], linestyle = '-')
    ax3[1,0].plot(x_axis_res, flux_nash_GM_res[1], color = tableau20[6], linestyle = '-')
    ax3[2,0].plot(x_axis_res, flux_nash_GM_res[2], color = tableau20[6], linestyle = '-')

    ax3[0,0].plot(x_axis_res, flux_static_values_res[0], color = tableau20[0], linestyle = '-')
    ax3[1,0].plot(x_axis_res, flux_static_values_res[1], color = tableau20[0], linestyle = '-')
    ax3[2,0].plot(x_axis_res, flux_static_values_res[2], color = tableau20[0], linestyle = '-')
    #ax3[1].plot(x_axis_res, flux_static_values_res[0]/flux_static_values_res[0], color = tableau20[0], linestyle = '-')

    ax3[-1,1].set_xlabel('Top predation pressure ($\\xi$) \n month$^{-1}$')

    ax3[-1,0].set_xlabel('Carrying capacity ($\overline{R}$) $m_c$m$^{-3}$')



    fig3.tight_layout()
    if settings['linear'] is False:
        plt.savefig('top_down_flux.pdf')
    else:
        plt.savefig('top_down_flux_linear.pdf')



    fig5, ax5 = plt.subplots(2, 2,  sharex='col', sharey = 'row')
    fig5.set_size_inches((12/2.54, 12/2.54))

#    ax5[0,0].text(0, 1.1, "Equilibrium consumption rate (1)", transform=ax5[0,0].transAxes)
#    ax5[0,1].text(0, 1.1, "(2)", transform=ax5[0,1].transAxes)

    ax5[0, 0].text(1.05, 0.9, "(a)", transform=ax5[0, 0].transAxes)
    ax5[1, 0].text(1.05, 0.9, "(b)", transform=ax5[1, 0].transAxes)

    ax5[0, 1].text(1.05, 0.9, "(c)", transform=ax5[0, 1].transAxes)
    ax5[1, 1].text(1.05, 0.9, "(d)", transform=ax5[1, 1].transAxes)

    ax5[0, 0].plot(x_axis_res, func_nash_GM_res[0], color = tableau20[6], linestyle = '-')
    ax5[0, 0].plot(x_axis_res, func_static_values_res[0], color = tableau20[0], linestyle = '-')
    ax5[-1, 0].set_xlabel('Carrying capacity ($\overline{R}$), \n $m_c$ m$^{-3}$')
    ax5[0, 0].set_ylabel("Consumer \n Consumption/Max")

    ax5[1, 0].plot(x_axis_res, func_nash_GM_res[1], color = tableau20[6], linestyle = '-')
    ax5[1, 0].plot(x_axis_res, func_static_values_res[1], color = tableau20[0], linestyle = '-')
    ax5[1, 0].set_ylabel("Predator, \n Consumption/Max")
    ax5[0, 1].plot(x_axis_phi0, func_nash_GM_phi0[:,0], color = tableau20[6], linestyle = '-')
    ax5[0, 1].plot(x_axis_phi0, func_static_values_phi0[:,0], color = tableau20[0], linestyle = '-')
    ax5[-1, 1].set_xlabel('Max predation pressure ($\\xi$) \n month$^{-1}$')
    #ax5[0, 1].set_ylabel("C consumption/Max")

    ax5[1, 1].plot(x_axis_phi0, func_nash_GM_phi0[:,1], color = tableau20[6], linestyle = '-')
    ax5[1, 1].plot(x_axis_phi0, func_static_values_phi0[:,1], color = tableau20[0], linestyle = '-')
#    ax5[1, 1].set_ylabel("P consumption/Max")

    fig5.tight_layout()
    if settings['linear'] is False:
        plt.savefig('functional_response_compare.pdf')
    else:
        plt.savefig('functional_response_compare_linear.pdf')



    fig6, ax6 = plt.subplots(1, 2, sharey=True)
    fig6.set_size_inches((12/2.54, 6/2.54))
    #plt.title(
    #    "Functional response of predator, P " + str(np.round(pred_m, 2)) + " R " + str(np.round(res_m, 2)))
#    ax6[0].text(0, 1.1, "Optimal consumption rate: predator,", transform=ax6[0].transAxes)
#    ax6[1].text(0, 1.1, "consumer", transform=ax6[1].transAxes)


    ax6[1].plot(prey_variation, prey_variation / (prey_variation + params_ext['nu0']),
             label="P consumption, static", color = tableau20[0], linestyle = '-')
    dc = []
    #print(tableau20[0])
    for k in range(pred_var_fid+1):
        dyn_col = ((pred_var_fid-k))*np.array(tableau20[0])+(k)*np.array(tableau20[6])
        dyn_col = dyn_col/np.max(dyn_col)
        dc.append((dyn_col[0], dyn_col[1], dyn_col[2]))
        print(dyn_col, tableau20[0], tableau20[6])
        ax6[1].plot(prey_variation, frp[:, k, 0] * frp[:, k, 1] * prey_variation / (
                    frp[:, k, 0] * frp[:, k, 1] * prey_variation + params_ext['nu0']), color = dc[k], linestyle = '-.',
             label="P consumption, optimal")
        #Changed to relative functional, Changed from 0.2 to accomodate theoretical-ecology
    ax6[1].set_xlabel("Consumers $(C)$, \n $m_c$m$^{-3}$")
    ax6[0].set_ylabel("Consumption/Max")
    ax6[0].plot(resource_variation, resource_variation / (resource_variation + params_ext['nu0']),
             color = tableau20[0], linestyle = '-', label="C consumption, static")

    for k in range(pred_var_fid+1):
        ax6[0].plot(resource_variation,
                 frc[:, k, 0] * resource_variation / (frc[:, k, 0] * resource_variation + params_ext['nu0']),
                 color = dc[k], linestyle = '-.')
        #alpha = 0.5 #Changed from 0.2 to accomodate theoretical-ecology
        print(frc[:, k, 0] * resource_variation / (frc[:, k, 0] * resource_variation + params_ext['nu0']))
    ax6[0].set_xlabel("Resource $(R)$, \n $m_c$m$^{-3}$")
    fig6.tight_layout()
    if settings['linear'] is False:
        plt.savefig("Functional_response_consumer.pdf")
    else:
        plt.savefig("Functional_response_consumer_linear.pdf")

    #print((frp[:, 0, 0] * frp[:, 0, 1] * prey_variation / (frp[:, 0, 0] * frp[:, 0, 1] * prey_variation + params_ext['nu0']))[25])



    fig8, ax8 = plt.subplots(6, 1, sharex=True, gridspec_kw={'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5]})
    fig8.set_size_inches((12/2.54, 14/2.54))

    #ax8[0].text(0, 1.1, "Time Dynamics", transform=ax8[0].transAxes)

    ax8[0].text(1.05, 0.9, "(a)", transform=ax8[0].transAxes)
    ax8[1].text(1.05, 0.8, "(b)", transform=ax8[1].transAxes)
    ax8[2].text(1.05, 0.9, "(c)", transform=ax8[2].transAxes)
    ax8[3].text(1.05, 0.8, "(d)", transform=ax8[3].transAxes)
    ax8[4].text(1.05, 0.9, "(e)", transform=ax8[4].transAxes)
    ax8[5].text(1.05, 0.8, "(f)", transform=ax8[5].transAxes)
    #ax8[5].set_title('Population dynamics of optimal populations with bottom-up control')
    ax8[0].set_ylabel('Resource, \n $m_c m^{-3}$')
    ax8[-1].set_xlabel('Months')


    ax8[0].plot(tim, sol[0, :], color = tableau20[6], linestyle = '-')
    ax8[0].plot(tim, sol_2[0, :], color = tableau20[0], linestyle = '-')
    ax8[1].set_ylim((0, 1))

    ax8[1].set_ylabel('$\\tau_c$')

    ax8[1].fill_between(tim, strat[0, :], y2 = 0, alpha = 0.5,color = tableau20[6], linestyle = '-')

    ax8[2].set_ylabel('Consumer, \n $10^{-1} m_c$ m$^{-3}$')

    ax8[2].plot(tim, 10*sol[1, :], color =  tableau20[6], linestyle = '-')
    ax8[2].plot(tim, 10*sol_2[1, :], color = tableau20[0], linestyle = '-')

    ax8[3].set_ylabel('$\\tau_p  \\tau_c $')

    ax8[3].fill_between(tim, strat[1, :]*strat[0, :], y2 =0, alpha = 0.5, color = tableau20[6], linestyle = '-')
    ax8[3].set_ylim((0, 1))

    ax8[4].set_ylabel('Predator, \n $10^{-3} m_c$ m$^{-3}$')

    ax8[4].plot(tim, 1000*sol[2, :], color = tableau20[6], linestyle = '-')
    ax8[4].plot(tim, 1000*sol_2[2, :], color = tableau20[0], linestyle = '-')

    ax8[5].set_ylabel('$\\tau_p$')
    ax8[5].set_ylim((0, 1))
    ax8[5].fill_between(tim, strat[1, :], y2 = 0, alpha = 0.5, color = tableau20[6], linestyle = '-')

    fig8.tight_layout()
    if settings['linear'] is False:
        plt.savefig('simulation_dynamics.pdf')
    else:
        plt.savefig('simulation_dynamics_linear.pdf')

    heatmap_plotter([grid_data[:,:, 0]], "res_var", [x_axis_res[0], x_axis_res[-1], x_axis_phi0[0], x_axis_phi0[-1]])
    heatmap_plotter([grid_data[:,:,1]], "cons_var", [x_axis_res[0], x_axis_res[-1], x_axis_phi0[0], x_axis_phi0[-1]])
    heatmap_plotter([grid_data[:,:,2]], "pred_var", [x_axis_res[0], x_axis_res[-1], x_axis_phi0[0], x_axis_phi0[-1]])
    heatmap_plotter([grid_data[:,:,3], grid_data[:,:,4]], "strat_var", [x_axis_res[0], x_axis_res[-1], x_axis_phi0[0], x_axis_phi0[-1]])
    #heatmap_plotter(, "taup_var", [20, x_axis_res[-1], 0.6*params_ext['phi0'], 3*params_ext['phi0']])

    #heatmap_plotter([np.vstack([grid_data_down, grid_data])[:,:,3], np.vstack([grid_data_down, grid_data])[:,:,4]], "strat_var_down", [5, 30, 0.6*params_ext['phi0'], 3*params_ext['phi0']])

#    print(grid_data.shape)
#    print(grid_data[:, 0, 4].shape)
#    print(grid_data[:, 0, 3].shape)
#    print(strat_nash_GM_phi0[:, 0].shape)
#    print(strat_nash_GM_phi0[0, 0], grid_data[0, 0, 3])
#    print(strat_nash_GM_phi0[:, 1] - grid_data[:, 0, 4])

#    print(strat_nash_GM_phi0)
#    print(2*nash_GM_res[int(its/2), 2])
#    print(x_axis_phi0)
print("Resource: ",  2*nash_GM_res[int(its/3), 0], "Prey: ",nash_GM_res[int(its/3), 1]," ", 1.5*nash_GM_res[int(its/3), 1], "Predator: ", 0.75*nash_GM_res[int(its/3), 1], 1.5*nash_GM_res[int(its/3), 1])
#print(resource_variation[int(resource_variation.size/2)-1])
#print(pred_m)