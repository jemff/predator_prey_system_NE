from common_functions import *
import analytical_numerical_solution as an_sol
import semi_impl_eul as num_sol
import matplotlib.pyplot as plt
import copy as copy
from multiprocessing import Pool


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size = 10)
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

mass_vector = np.array([1, 1, 100])

cost_of_living, nu, growth_max, lam = parameter_calculator_mass(mass_vector, v = 0.1)
base = 20
phi1 = cost_of_living[1] #*2/3 #/3 #*2#/2#*5
phi0 = cost_of_living[1] #*4/3 #The problem is that nash equiibrium does not exist in the low levels...
eps = 0.7
epsn = 0.7

cmax, cp = growth_max
mu0 = 0 # cost_of_living[0] #/3 #*2#/2#*5 #*60 #/2
mu1 = cost_of_living[0]#/3 #*2 #*2#/2#*5 #*2 #*10 #/2
nu0 = nu[0] #nu
nu1 = nu[1] #nu

its = 600
its_mort = 1200
fidelity = 50
pred_var_fid = 50

print(phi0)

params_ext = {'cmax': cmax, 'mu0': mu0, 'mu1': mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0': phi0, 'phi1': phi1,
              'resource': base, 'lam': lam, 'nu0': nu0, 'nu1': nu1}



if settings['sensitivity'] is True:
    init = np.array([0.9500123,  0.4147732,  0.01282899 ])
    params_ext['resource'] = 20
    reverse = True
    start = 5
    stop = 30
    x_axis_res = np.linspace(start, stop, its)

    nash_GM_res, strat_nash_GM_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, linear = settings['linear'])

    start = 0.6*phi0
    stop = 3*phi0

    x_axis_phi0 = np.linspace(start, stop, its_mort)

    nash_GM_phi0, strat_nash_GM_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its_mort, reverse = reverse, type = 'phi0', linear = settings['linear'])

    with open('bifurcation_data.npy', 'wb') as f:
        np.save(f, x_axis_res)
        np.save(f, nash_GM_res)
        np.save(f, strat_nash_GM_res)

        np.save(f, x_axis_phi0)
        np.save(f, nash_GM_phi0)
        np.save(f, strat_nash_GM_phi0)

       # np.save(f, results)

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
    params_ext['resource'] = 20
    t_end = 5
    init = static_eq_calc(params_ext)*1.1
    init[-1] = 0.001*init[-1]    # np.array([0.5383403,  0.23503815, 0.00726976]) #np.array([5.753812957581866, 5.490194692112937, 1.626801718856221])#
    # params_ext['resource'] = 16 This si for the low-resource paradigm
    tim, sol, strat = num_sol.semi_implicit_euler(t_end, init, 0.0001, lambda t, y, tn, tp:
    num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp, linear = settings['linear']), params_ext, opt_prey=True,
                                                  opt_pred=True, linear = settings['linear'])
    tim, sol_2, strat_2 = num_sol.semi_implicit_euler(t_end, init, 0.0001, lambda t, y, tn, tp:
    num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp), params_ext,
                                                      opt_prey=False, opt_pred=False)

    with open('simulate_data.npy', 'wb') as g:
        np.save(g, tim)
        np.save(g, sol)
        np.save(g, strat)

        np.save(g, sol_2)
        np.save(g, strat_2)


elif settings['simulation'] is False:
    with open('simulate_data.npy', 'rb') as g:  # Save time by not generating all the data every time.
        tim = np.load(g)
        sol = np.load(g)
        strat = np.load(g)

        sol_2 = np.load(g)
        strat_2 = np.load(g)

if settings['heatmap_up'] is True:
    its_heat = its
    input_data = []
    for i in range(its_mort):
        par_t = copy.deepcopy(params_ext)
        par_t['phi0'] = x_axis_phi0[i]
        input_data.append({'values': nash_GM_phi0[i], 'parameters': par_t, 'strat': strat_nash_GM_phi0[i], 'iteration': i})

    start = 20 #Change to 0.6*phi0, but for now we use phi0 for stability reasons
    stop = 30
    reverse = False

    agents = 4
    def temp_func(ivp, its = its_heat, start = start, stop = stop, settings = settings, reverse = reverse):
        outputs = np.zeros((its, 5))
        values, strategies = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2,
                                     ivp['values'], ivp['parameters'], start, stop, its,
                                     reverse=reverse, type='resource',
                                     linear=settings['linear'], strat=ivp['strat'])
        outputs[:, 0:3] = values
        outputs[:, 3:] = strategies
        print(ivp['iteration'], ivp['parameters']['phi0'], ivp['strat'], strat_nash_GM_phi0[0])

        return outputs


    with Pool(processes = agents) as pool:
        grid_data = pool.map(temp_func, input_data)
    #print(combined_strat_finder(input_data[0]['parameters'], [1.04194356, 0.21399234, 0.00927935], x0 = np.array([0.5, 0.5]) ))
    #grid_data = temp_func(input_data[0])
    #print(grid_data[0], strat_nash_GM_phi0[0], nash_GM_phi0[0])
    grid_data = np.array(grid_data)
    with open('heatmap_data.npy', 'wb') as g:
        np.save(g, grid_data)

elif settings['heatmap_up'] is False:
    with open('heatmap_data.npy', 'rb') as g:  # Save time by not generating all the data every time.
        grid_data = np.load(g)


if settings['heatmap_down'] is True:
    its_heat = int(its*1.5)
    input_data = []
    for i in range(its_mort):
        par_t = copy.deepcopy(params_ext)
        par_t['phi0'] = x_axis_phi0[i]
        input_data.append({'values': nash_GM_phi0[i], 'parameters': par_t, 'strat': strat_nash_GM_phi0[i], 'iteration': i})

    start = 5 #Change to 0.6*phi0, but for now we use phi0 for stability reasons
    stop = 20
    reverse = True

    agents = 4
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

#elif settings['heatmap_down'] is False:
#    with open('heatmap_down_data.npy', 'rb') as g:  # Save time by not generating all the data every time.
#        grid_data_down = np.load(g)


if settings['func_dyn'] is True:

    res_m = nash_GM_res[500, 0]
    prey_m = nash_GM_res[500, 1]

    pred_m = np.linspace(0.001*nash_GM_res[500, 2], 2*nash_GM_res[500, 2], pred_var_fid + 1)

    print("Values", nash_GM_res[500])
    #20 is a good magic number

    params_t = copy.deepcopy(params_ext)
    params_t_down = copy.deepcopy(params_ext)

    xi_var = np.linspace(0.001*params_ext['phi0'], 2*params_ext['phi0'], fidelity+1)

    frp = np.zeros((fidelity + 1, pred_var_fid+1, 2))
    frc = np.zeros((fidelity + 1, pred_var_fid+1, 2))
    resource_variation = np.linspace(0.001*res_m, 2*res_m, fidelity + 1)
    prey_variation = np.linspace(0.001*prey_m, 2*prey_m, fidelity + 1)



    frc[int(fidelity/2), int(pred_var_fid/2)] = strat_nash_GM_res[500] #This all assumes symmetry
    frp[int(fidelity/2), int(pred_var_fid/2)] = strat_nash_GM_res[500]
    for k in range(int(pred_var_fid / 2)):
        params_t['phi0'] = xi_var[int(fidelity / 2) + k + 1]
        params_t_down['phi0'] = xi_var[int(fidelity / 2) - k - 1]

        frp[int(fidelity/2), int(pred_var_fid / 2) + k + 1] = combined_strat_finder(params_t, np.array([res_m, prey_variation[int(fidelity/2)], pred_m[int(pred_var_fid / 2)]]), x0=frp[int(fidelity/2), int(pred_var_fid / 2) + k], linear=settings['linear'])
        frc[int(fidelity/2), int(pred_var_fid / 2) + k + 1] = combined_strat_finder(params_ext, np.array([resource_variation[int(fidelity/2)], prey_m, pred_m[int(pred_var_fid / 2) - k]]), x0=frc[int(fidelity/2), int(pred_var_fid / 2) + k ],       linear=settings['linear'])

        frp[int(fidelity/2), int(pred_var_fid / 2) - 1 - k] = combined_strat_finder(params_t_down, np.array([res_m, prey_variation[int(fidelity/2)], pred_m[int(pred_var_fid / 2) + k]]), x0=frp[int(fidelity/2), int(pred_var_fid / 2) - k], linear=settings['linear'])
        frc[int(fidelity/2), int(pred_var_fid / 2) - 1 - k] = combined_strat_finder(params_ext, np.array([resource_variation[int(fidelity/2)], prey_m, pred_m[int(pred_var_fid / 2) - k]]), x0=frc[int(fidelity/2), int(pred_var_fid / 2) - k], linear=settings['linear'])


    for i in range(1, int(fidelity/2)+1):
        for k in range(int(pred_var_fid / 2) + 1):
            params_t['phi0'] = xi_var[int(fidelity / 2) + k]
            params_t_down['phi0'] = xi_var[int(fidelity / 2) - k]

            frc[int(fidelity/2) - i + 1, int(pred_var_fid/2) + k] = combined_strat_finder(params_ext, np.array([resource_variation[int(fidelity/2) - i + 1], prey_m, pred_m[int(pred_var_fid/2) - k]]), x0 = frc[int(fidelity/2) - i + 2, int(pred_var_fid/2) + k ], linear = settings['linear'])
            frc[int(fidelity/2) - i + 1, int(pred_var_fid/2) - k] = combined_strat_finder(params_ext, np.array([resource_variation[int(fidelity/2) - i + 1], prey_m, pred_m[int(pred_var_fid/2) - k]]), x0 = frc[int(fidelity/2) - i + 2, int(pred_var_fid/2) - k ], linear = settings['linear'])
            frc[int(fidelity/2) + i, int(pred_var_fid/2) + k] = combined_strat_finder(params_ext, np.array([resource_variation[int(fidelity/2) + i], prey_m, pred_m[int(pred_var_fid/2) - k]]), x0 = frc[int(fidelity/2) + i - 1, int(pred_var_fid/2) + k ], linear = settings['linear'])
            frc[int(fidelity/2) + i, int(pred_var_fid/2) - k] = combined_strat_finder(params_ext, np.array([resource_variation[int(fidelity/2) + i], prey_m, pred_m[int(pred_var_fid/2) - k]]), x0 = frc[int(fidelity/2) + i - 1, int(pred_var_fid/2) - k ], linear = settings['linear'])


            frp[int(fidelity/2) - i + 1, int(pred_var_fid/2) + k] = combined_strat_finder(params_t, np.array([res_m, prey_variation[int(fidelity/2) - i + 1], pred_m[int(pred_var_fid/2)]]), x0 = frp[int(fidelity/2) - i + 2, int(pred_var_fid/2) + k], linear = settings['linear'])
            frp[int(fidelity/2) + i, int(pred_var_fid/2) + k] = combined_strat_finder(params_t, np.array([res_m, prey_variation[int(fidelity/2) + i], pred_m[int(pred_var_fid/2)]]), x0 = frp[int(fidelity/2) + i - 1, int(pred_var_fid/2) + k], linear = settings['linear'])

            frp[int(fidelity/2) - i + 1, int(pred_var_fid/2) - k] = combined_strat_finder(params_t_down, np.array([res_m, prey_variation[int(fidelity/2) - i + 1], pred_m[int(pred_var_fid/2)]]), x0 = frp[int(fidelity/2) - i + 2, int(pred_var_fid/2) - k], linear = settings['linear'])
            frp[int(fidelity/2) + i, int(pred_var_fid/2) - k] = combined_strat_finder(params_t_down, np.array([res_m, prey_variation[int(fidelity/2) + i], pred_m[int(pred_var_fid/2)]]), x0 = frp[int(fidelity/2) + i - 1, int(pred_var_fid/2) - k], linear = settings['linear'])


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
    for i in range(its_mort):
        params_temp_phi0['phi0'] = x_axis_phi0[i]
        static_values_phi0[i] = static_eq_calc(params_temp_phi0)

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
    print(frp[int(fidelity/2), :, 1], frp[0, :, 1], frp[0, :, 0])

    fig, ax = plt.subplots(6, 2, gridspec_kw={'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5]}, sharex='col', sharey = 'row')
    fig.set_size_inches((16/2.54, 16/2.54))

    #ax[5].set_title('Population dynamics of optimal populations with bottom-up control')
    ax[0, 0].set_ylabel('Resource, $m_p\cdot m^{-3}$')
    ax[-1, 0].set_xlabel('Carrying capacity $(\overline{R})$ $m_c\cdot m^{-3}$')


    ax[0, 0].plot(x_axis_res, nash_GM_res[:, 0], color = tableau20[6], linestyle = '-')
    ax[0, 0].plot(x_axis_res, static_values_res[:, 0], color = tableau20[0], linestyle = '-')

    ax[1, 0].set_ylabel('$\\tau_c$')

    ax[1, 0].fill_between(x_axis_res, strat_nash_GM_res[:, 0], y2 =0, alpha = 0.5, color = tableau20[6], linestyle = '-')

    ax[2, 0].set_ylabel('Consumer, $m_c\cdot m^{-3}$')

    ax[2, 0].plot(x_axis_res, nash_GM_res[:, 1], color =  tableau20[6], linestyle = '-')
    ax[2, 0].plot(x_axis_res, static_values_res[:, 1], color = tableau20[0], linestyle = '-')

    ax[3, 0].set_ylabel('$\\tau_p \cdot \\tau_c$')

    ax[3, 0].fill_between(x_axis_res, strat_nash_GM_res[:, 1]*strat_nash_GM_res[:, 0], y2 = 0, alpha = 0.5, color = tableau20[6], linestyle = '-')
    ax[3, 0].set_ylim((0, 1))

    ax[4, 0].set_ylabel('Predator, $m_c\cdot m^{-3}$')

    ax[4, 0].plot(x_axis_res, nash_GM_res[:, 2], color = tableau20[6], linestyle = '-')
    ax[4, 0].plot(x_axis_res, static_values_res[:, 2], color = tableau20[0], linestyle = '-')

    ax[5, 0].set_ylabel('$\\tau_p$')
    ax[5, 0].set_ylim((0, 1))
    ax[5, 0].fill_between(x_axis_res, strat_nash_GM_res[:, 1], y2 = 0, alpha = 0.5, color = tableau20[6], linestyle = '-')

    #    ax[1, 0].set_ylabel('Resource, $m_c\cdot m^{-3}$')
    ax[-1, 1].set_xlabel('Top predation pressure ($\\xi$) $month^{-1}$')

    ax[0, 1].plot(x_axis_phi0, nash_GM_phi0[:, 0], color=tableau20[6], linestyle='-')
    ax[0, 1].plot(x_axis_phi0, static_values_phi0[:, 0], color=tableau20[0], linestyle='-')

    ax[1, 1].fill_between(x_axis_phi0, strat_nash_GM_phi0[:, 0], y2=0, color=tableau20[6],
                          linestyle='-', alpha=0.5)


    ax[2, 1].plot(x_axis_phi0, nash_GM_phi0[:, 1], color=tableau20[6], linestyle='-')
    ax[2, 1].plot(x_axis_phi0, static_values_phi0[:, 1], color=tableau20[0], linestyle='-')


    ax[3, 1].fill_between(x_axis_phi0, strat_nash_GM_phi0[:, 1] * strat_nash_GM_phi0[:, 0],
                          y2=0, color=tableau20[6], alpha=0.5,
                          linestyle='-')

    ax[4, 1].plot(x_axis_phi0, nash_GM_phi0[:, 2], color=tableau20[6], linestyle='-')
    ax[4, 1].plot(x_axis_phi0, static_values_phi0[:, 2], color=tableau20[0], linestyle='-')

    ax[5, 1].fill_between(x_axis_phi0, strat_nash_GM_phi0[:, 1], y2=0, alpha=0.5,
                          color=tableau20[6], linestyle='-')

    if settings['linear'] is False:
        plt.savefig('sensitivity.pdf')
    else:
        plt.savefig('sensitivity_linear.pdf')


    fig3, ax3 = plt.subplots(1, 2, sharey=True)
    fig3.set_size_inches((16/2.54, 8/2.54))

    #ax3[2].set_title('Ratio of flux compared to static system, top-down control')

    ax3[0].set_ylabel('Production/Static production')

    ax3[0].plot(x_axis_phi0, flux_nash_GM_phi0[:, 0]/flux_static_values_phi0[:, 0], color = tableau20[2], linestyle = '-')

    ax3[0].plot(x_axis_phi0, flux_nash_GM_phi0[:, 1] / flux_static_values_phi0[:, 1], color=tableau20[6], linestyle='-')

    ax3[0].plot(x_axis_phi0, flux_nash_GM_phi0[:, 2] / flux_static_values_phi0[:, 2], color=tableau20[10], linestyle='-')

    ax3[0].plot(x_axis_phi0, flux_static_values_phi0[:, 0]/flux_static_values_phi0[:, 0], color = tableau20[0], linestyle = '-')


    ax3[1].plot(x_axis_res, flux_nash_GM_res[0]/flux_static_values_res[0], color = tableau20[2], linestyle = '-')
    ax3[1].plot(x_axis_res, flux_nash_GM_res[1]/flux_static_values_res[1], color = tableau20[6], linestyle = '-')
    ax3[1].plot(x_axis_res, flux_nash_GM_res[2]/flux_static_values_res[2], color = tableau20[10], linestyle = '-')

    ax3[1].plot(x_axis_res, flux_static_values_res[0]/flux_static_values_res[0], color = tableau20[0], linestyle = '-')

    ax3[0].set_xlabel('Top predation pressure ($\\xi$) $month^{-1}$')

    ax3[1].set_xlabel('Carrying capacity ($\overline{R}$) $m_c\cdot m^{-3}$')



    fig3.tight_layout()
    if settings['linear'] is False:
        plt.savefig('top_down_flux.pdf')
    else:
        plt.savefig('top_down_flux_linear.pdf')



    fig5, ax5 = plt.subplots(2, 2,  sharex='col', sharey = 'row')
    fig5.set_size_inches((16/2.54, 12/2.54))

    ax5[0, 0].plot(x_axis_res, func_nash_GM_res[0], color = tableau20[6], linestyle = '-')
    ax5[0, 0].plot(x_axis_res, func_static_values_res[0], color = tableau20[0], linestyle = '-')
    ax5[-1, 0].set_xlabel('Carrying capacity ($\overline{R}$) $m_c\cdot m^{-3}$')
    ax5[0, 0].set_ylabel("C consumption/Max")

    ax5[1, 0].plot(x_axis_res, func_nash_GM_res[1], color = tableau20[6], linestyle = '-')
    ax5[1, 0].plot(x_axis_res, func_static_values_res[1], color = tableau20[0], linestyle = '-')
    ax5[1, 0].set_ylabel("P consumption/Max")
    ax5[0, 1].plot(x_axis_phi0, func_nash_GM_phi0[:,0], color = tableau20[6], linestyle = '-')
    ax5[0, 1].plot(x_axis_phi0, func_static_values_phi0[:,0], color = tableau20[0], linestyle = '-')
    ax5[-1, 1].set_xlabel('Max predation pressure ($\\xi$) $month^{-1}$')
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
    fig6.set_size_inches((12/2.54, 8/2.54))
    #plt.title(
    #    "Functional response of predator, P " + str(np.round(pred_m, 2)) + " R " + str(np.round(res_m, 2)))

    ax6[0].plot(prey_variation, prey_variation / (prey_variation + params_ext['nu0']),
             label="P consumption, static", color = tableau20[0], linestyle = '-')
    for k in range(fidelity+1):
        ax6[0].plot(prey_variation, frp[:, k, 0] * frp[:, k, 1] * prey_variation / (
                    frp[:, k, 0] * frp[:, k, 1] * prey_variation + params_ext['nu0']), color = tableau20[6], linestyle = '-',
             label="P consumption, optimal", alpha = 0.05) #Changed to relative functional
    ax6[0].set_xlabel("Prey in $m_c\cdot m^{-3}$")
    ax6[0].set_ylabel("Consumption/Max")
    ax6[1].plot(resource_variation, resource_variation / (resource_variation + params_ext['nu0']),
             color = tableau20[0], linestyle = '-', label="C consumption, static")
    for k in range(fidelity+1):
        ax6[1].plot(resource_variation,
                 frc[:, k, 0] * resource_variation / (frc[:, k, 0] * resource_variation + params_ext['nu0']),
                 color = tableau20[6], linestyle = '-', label="C consumption, optimal", alpha = 0.05) #alpha = 0.5
    ax6[1].set_xlabel("Resource in $m_c\cdot m^{-3}$")
    fig6.tight_layout()
    if settings['linear'] is False:
        plt.savefig("Functional_response_consumer.pdf")
    else:
        plt.savefig("Functional_response_consumer_linear.pdf")

    print((frp[:, 0, 0] * frp[:, 0, 1] * prey_variation / (frp[:, 0, 0] * frp[:, 0, 1] * prey_variation + params_ext['nu0']))[25])



    fig8, ax8 = plt.subplots(6, 1, sharex=True, gridspec_kw={'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5]})
    fig8.set_size_inches((8/2.54, 16/2.54))

    #ax8[5].set_title('Population dynamics of optimal populations with bottom-up control')
    ax8[0].set_ylabel('Resource, $m_p\cdot m^{-3}$')
    ax8[-1].set_xlabel('Months')


    ax8[0].plot(tim, sol[0, :], color = tableau20[6], linestyle = '-')
    ax8[0].plot(tim, sol_2[0, :], color = tableau20[0], linestyle = '-')

    ax8[1].set_ylabel('$\\tau_c$')

    ax8[1].fill_between(tim, strat[0, :], y2 = min(strat[0, :]), alpha = 0.5,color = tableau20[6], linestyle = '-')

    ax8[2].set_ylabel('Consumer, $m_c\cdot m^{-3}$')

    ax8[2].plot(tim, sol[1, :], color =  tableau20[6], linestyle = '-')
    ax8[2].plot(tim, sol_2[1, :], color = tableau20[0], linestyle = '-')

    ax8[3].set_ylabel('$\\tau_p \cdot \\tau_c $')

    ax8[3].fill_between(tim, strat[1, :]*strat[0, :], y2 =0, alpha = 0.5, color = tableau20[6], linestyle = '-')
    ax8[3].set_ylim((0, 1))

    ax8[4].set_ylabel('Predator, $m_c\cdot m^{-3}$')

    ax8[4].plot(tim, sol[2, :], color = tableau20[6], linestyle = '-')
    ax8[4].plot(tim, sol_2[2, :], color = tableau20[0], linestyle = '-')

    ax8[5].set_ylabel('$\\tau_p$')
    ax8[5].set_ylim((0, 1))
    ax8[5].fill_between(tim, strat[1, :], y2 = 0, alpha = 0.5, color = tableau20[6], linestyle = '-')

    fig8.tight_layout()
    if settings['linear'] is False:
        plt.savefig('simulation_dynamics.pdf')
    else:
        plt.savefig('simulation_dynamics_linear.pdf')

    heatmap_plotter([grid_data[:,:, 0]], "res_var", [20, x_axis_res[-1], 0.6*params_ext['phi0'], 3*params_ext['phi0']])
    heatmap_plotter([grid_data[:,:,1]], "cons_var", [20, x_axis_res[-1], 0.6*params_ext['phi0'], 3*params_ext['phi0']])
    heatmap_plotter([grid_data[:,:,2]], "pred_var", [20, x_axis_res[-1], 0.6*params_ext['phi0'], 3*params_ext['phi0']])
    heatmap_plotter([grid_data[:,:,3], grid_data[:,:,4]], "strat_var", [20, x_axis_res[-1], 0.6*params_ext['phi0'], 3*params_ext['phi0']])
    #heatmap_plotter(, "taup_var", [20, x_axis_res[-1], 0.6*params_ext['phi0'], 3*params_ext['phi0']])

    #heatmap_plotter([np.vstack([grid_data_down, grid_data])[:,:,3], np.vstack([grid_data_down, grid_data])[:,:,4]], "strat_var_down", [5, 30, 0.6*params_ext['phi0'], 3*params_ext['phi0']])

    print(grid_data.shape)
    print(grid_data[:, 0, 4].shape)
    print(grid_data[:, 0, 3].shape)
    print(strat_nash_GM_phi0[:, 0].shape)
    print(strat_nash_GM_phi0[:, 0] - grid_data[:, 0, 3])
