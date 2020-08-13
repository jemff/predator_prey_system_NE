from common_functions import *
import analytical_numerical_solution as an_sol
import semi_impl_eul as num_sol
import matplotlib.pyplot as plt
import copy as copy

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

settings = {'gen_data': False, 'plot': True}

mass_vector = np.array([1, 1, 100])

cost_of_living, nu, growth_max, lam = parameter_calculator_mass(mass_vector, v = 0.1)
base = 16
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

print(phi0)

params_ext = {'cmax': cmax, 'mu0': mu0, 'mu1': mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0': phi0, 'phi1': phi1,
              'resource': base, 'lam': lam, 'nu0': nu0, 'nu1': nu1}



if settings['gen_data'] is True:


#    params_ext = {'cmax': cmax, 'mu0': mu0, 'mu1': mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0': phi0, 'phi1': phi1,
#                  'resource': base, 'lam': lam, 'nu0': nu0, 'nu1': nu1}


    params_ext['resource'] = 20
    t_end = 5
    init = 0.5 * static_eq_calc(
        params_ext)  # np.array([0.5383403,  0.23503815, 0.00726976]) #np.array([5.753812957581866, 5.490194692112937, 1.626801718856221])#
    # params_ext['resource'] = 16 This si for the low-resource paradigm
    tim, sol, strat = num_sol.semi_implicit_euler(t_end, init, 0.0001, lambda t, y, tn, tp:
    num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp), params_ext, opt_prey=True,
                                                  opt_pred=True)
    tim, sol_2, strat_2 = num_sol.semi_implicit_euler(t_end, init, 0.0001, lambda t, y, tn, tp:
    num_sol.optimal_behavior_trajectories(t, y, params_ext, taun=tn, taup=tp), params_ext,
                                                      opt_prey=False, opt_pred=False)

    init = np.array([0.9500123,  0.4147732,  0.01282899 ])
    params_ext['resource'] = 16
    reverse = True
    start = 5
    stop = 30
    x_axis_res = np.linspace(start, stop, its)

    nash_GM_res, strat_nash_GM_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse)

    start = 0.6*phi0
    stop = 1.2*phi0
    x_axis_phi0 = np.linspace(start, stop, its)

    nash_GM_phi0, strat_nash_GM_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, type = 'phi0')

    with open('bifurcation_data.npy', 'wb') as f:
        np.save(f, x_axis_res)
        np.save(f, nash_GM_res)
        np.save(f, strat_nash_GM_res)

        np.save(f, x_axis_phi0)
        np.save(f, nash_GM_phi0)
        np.save(f, strat_nash_GM_phi0)

    with open('simulate_data.npy', 'wb') as g:
        np.save(g, tim)
        np.save(g, sol)
        np.save(g, strat)

        np.save(g, sol_2)
        np.save(g, strat_2)

elif settings['gen_data'] is False:
    with open('bifurcation_data.npy', 'rb') as f:  # Save time by not generating all the data every time.

        x_axis_res = np.load(f)

        nash_GM_res = np.load(f)
        strat_nash_GM_res = np.load(f)

        x_axis_phi0= np.load(f)

        nash_GM_phi0 = np.load(f)
        strat_nash_GM_phi0 = np.load(f)

    with open('simulate_data.npy', 'rb') as g:  # Save time by not generating all the data every time.
        tim = np.load(g)
        sol = np.load(g)
        strat = np.load(g)

        sol_2 = np.load(g)
        strat_2 = np.load(g)


if settings['plot'] is True:
    # Creates just a figure and only one subplot
    # Creates two subplots and unpacks the output array immediately
    static_values_res = np.zeros((its, 3))
    static_values_phi0 = np.zeros((its, 3))
    flux_nash_GM_phi0 = np.zeros((its, 3))
    flux_nash_Gill_phi0 = np.zeros((its, 3))
    flux_static_values_phi0 = np.zeros((its, 3))

    ones = np.repeat(1, its)

    params_temp_res = copy.deepcopy(params_ext)
    params_temp_phi0 = copy.deepcopy(params_ext)

    func_nash_GM_phi0 = np.zeros((its, 2))
    func_nash_Gill_phi0 = np.zeros((its, 2))
    func_static_values_phi0 = np.zeros((its, 2))

    for i in range(its):
        params_temp_res['resource'] = x_axis_res[i]
        static_values_res[i] = static_eq_calc(params_temp_res)

        params_temp_phi0['phi0'] = x_axis_phi0[i]
        static_values_phi0[i] = static_eq_calc(params_temp_phi0)

        flux_nash_GM_phi0[i] = an_sol.flux_calculator(nash_GM_phi0[i, 0], nash_GM_phi0[i, 1], nash_GM_phi0[i, 2],
                                                      strat_nash_GM_phi0[i, 0], strat_nash_GM_phi0[i, 1], params_temp_phi0)

        flux_static_values_phi0[i] = an_sol.flux_calculator(static_values_phi0[i, 0], static_values_phi0[i, 1],
                                                            static_values_phi0[i, 2], 1, 1, params_temp_phi0)

        func_nash_GM_phi0[i] = an_sol.frp_calc(nash_GM_phi0[i, 0], nash_GM_phi0[i, 1], nash_GM_phi0[i, 2],
                                               strat_nash_GM_phi0[i, 0], strat_nash_GM_phi0[i, 1], params_temp_phi0)
        func_static_values_phi0[i] = an_sol.frp_calc(static_values_phi0[i, 0], static_values_phi0[i, 1],
                                                     static_values_phi0[i, 2], 1, 1, params_temp_phi0)


    flux_nash_GM_res = an_sol.flux_calculator(nash_GM_res[:, 0], nash_GM_res[:, 1], nash_GM_res[:, 2], strat_nash_GM_res[:, 0], strat_nash_GM_res[:, 1], params_ext)
    flux_static_values_res = an_sol.flux_calculator(static_values_res[:, 0], static_values_res[:, 1], static_values_res[:, 2], ones, ones, params_ext)

    func_nash_GM_res = an_sol.frp_calc(nash_GM_res[:, 0], nash_GM_res[:, 1], nash_GM_res[:, 2], strat_nash_GM_res[:, 0], strat_nash_GM_res[:, 1], params_ext)
    func_static_values_res = an_sol.frp_calc(static_values_res[:, 0], static_values_res[:, 1], static_values_res[:, 2], ones, ones, params_ext)




    fig, ax = plt.subplots(6, 1, sharex=True, gridspec_kw={'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5]})
    fig.set_size_inches((8/2.54, 16/2.54))

    #ax[5].set_title('Population dynamics of optimal populations with bottom-up control')
    ax[0].set_ylabel('Resource, $m_p/m^3$')
    ax[-1].set_xlabel('Carrying capacity $(\overline{R})$ $m_c/m^3$')


    ax[0].plot(x_axis_res, nash_GM_res[:, 0], color = tableau20[6], linestyle = '-')
    ax[0].plot(x_axis_res, static_values_res[:, 0], color = tableau20[0], linestyle = '-.')

    ax[1].set_ylabel('$\\tau_c$')

    ax[1].plot(x_axis_res, strat_nash_GM_res[:, 0], color = tableau20[6], linestyle = '-')

    ax[2].set_ylabel('Consumer, $m_c/m^3$')

    ax[2].plot(x_axis_res, nash_GM_res[:, 1], color =  tableau20[6], linestyle = '-')
    ax[2].plot(x_axis_res, static_values_res[:, 1], color = tableau20[0], linestyle = '-.')

    ax[3].set_ylabel('$\\tau_p \cdot \\tau_c$')

    ax[3].plot(x_axis_res, strat_nash_GM_res[:, 1]*strat_nash_GM_res[:, 0], color = tableau20[6], linestyle = '-')
    ax[3].set_ylim((0, 0.4))

    ax[4].set_ylabel('Predator, $m_c/m^3$')

    ax[4].plot(x_axis_res, nash_GM_res[:, 2], color = tableau20[6], linestyle = '-')
    ax[4].plot(x_axis_res, static_values_res[:, 2], color = tableau20[0], linestyle = '-.')

    ax[5].set_ylabel('$\\tau_p$')
    ax[5].set_ylim((0.5, 1))
    ax[5].plot(x_axis_res, strat_nash_GM_res[:, 1], color = tableau20[6], linestyle = '-')

    fig.tight_layout()

    plt.savefig('bottom_up_pop_dyn.pdf')

    fig2, ax2 = plt.subplots(6, 1, sharex=True, gridspec_kw={'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5]})
    fig2.set_size_inches((8/2.54, 16/2.54))

    #ax2[5].set_title('Population dynamics of optimal populations with top-down control')

    ax2[0].set_ylabel('Resource, $m_c/m^3$')
    ax2[-1].set_xlabel('Top predation pressure ($\\xi$) $m_c/(m^3 month)$')

    ax2[0].plot(x_axis_phi0, nash_GM_phi0[:, 0], color = tableau20[6], linestyle = '-')
    ax2[0].plot(x_axis_phi0, static_values_phi0[:, 0], color = tableau20[0], linestyle = '-.')

    ax2[1].set_ylabel('$\\tau_c$')

    ax2[1].plot(x_axis_phi0, strat_nash_GM_phi0[:, 0], color = tableau20[6], linestyle = '-')

    ax2[2].set_ylabel('Consumer, $m_c/m^3$')

    ax2[2].plot(x_axis_phi0, nash_GM_phi0[:, 1], color = tableau20[6], linestyle = '-')
    ax2[2].plot(x_axis_phi0, static_values_phi0[:, 1], color = tableau20[0], linestyle = '-.')

    ax2[3].set_ylabel('$\\tau_c \cdot \\tau_p $')

    ax2[3].plot(x_axis_phi0, strat_nash_GM_phi0[:, 1]*strat_nash_GM_phi0[:, 0], color = tableau20[6], linestyle = '-')

    ax2[4].set_ylabel('Predator, $m_c/m^3$')

    ax2[4].plot(x_axis_phi0, nash_GM_phi0[:, 2], color = tableau20[6], linestyle = '-')
    ax2[4].plot(x_axis_phi0, static_values_phi0[:, 2], color = tableau20[0], linestyle = '-.')

    ax2[5].set_ylabel('$\\tau_p$')

    ax2[5].plot(x_axis_phi0, strat_nash_GM_phi0[:, 1], color = tableau20[6], linestyle = '-')

    fig2.tight_layout()
    plt.savefig('top_down_pop_dyn.pdf')

    fig3, ax3 = plt.subplots(3, 1, sharex=True)
    fig3.set_size_inches((8/2.54, 12/2.54))

    #ax3[2].set_title('Ratio of flux compared to static system, top-down control')

    ax3[0].set_ylabel('R Production')

    ax3[0].plot(x_axis_phi0, flux_nash_GM_phi0[:, 0]/flux_static_values_phi0[:, 0], color = tableau20[6], linestyle = '-')
    ax3[0].plot(x_axis_phi0, flux_static_values_phi0[:, 0]/flux_static_values_phi0[:, 0], color = tableau20[0], linestyle = '-.')


    ax3[1].set_ylabel('C Production')

    ax3[1].plot(x_axis_phi0, flux_nash_GM_phi0[:, 1]/flux_static_values_phi0[:, 1], color = tableau20[6], linestyle = '-')
    ax3[1].plot(x_axis_phi0, flux_static_values_phi0[:, 1]/flux_static_values_phi0[:, 1], color = tableau20[0], linestyle = '-.')

  #  ax3[1].plot(x_axis_phi0, flux_nash_Gill_phi0[:, 1]/flux_static_values_phi0[:, 1],  color = 'Blue', linestyle = '-')

    ax3[2].set_ylabel('P Production')

    ax3[2].plot(x_axis_phi0, flux_nash_GM_phi0[:, 2]/flux_static_values_phi0[:, 2], color = tableau20[6], linestyle = '-')
    ax3[2].plot(x_axis_phi0, flux_static_values_phi0[:, 2]/flux_static_values_phi0[:, 2], color = tableau20[0], linestyle = '-.')

    ax3[-1].set_xlabel('Top predation pressure ($\\xi$) $m_c/(m^3 day)$')

    fig3.tight_layout()
    plt.savefig('top_down_flux.pdf')


    fig4, ax4 = plt.subplots(3, 1, sharex=True)
    fig4.set_size_inches((8/2.54, 12/2.54))
    #ax4[2].set_title('Ratio of flux compared to static system, bottom-up control')

    ax4[0].set_ylabel('R Production')
    ax4[0].plot(x_axis_res, flux_nash_GM_res[0]/flux_static_values_res[0], color = tableau20[6], linestyle = '-')
    ax4[0].plot(x_axis_res, flux_static_values_res[0]/flux_static_values_res[0], color = tableau20[0], linestyle = '-.')

    ax4[1].set_ylabel('C Production')

    ax4[1].plot(x_axis_res, flux_nash_GM_res[1]/flux_static_values_res[1], color = tableau20[6], linestyle = '-')
    ax4[1].plot(x_axis_res, flux_static_values_res[1]/flux_static_values_res[1], color = tableau20[0], linestyle = '-.')

    ax4[2].set_ylabel('P Production')

    ax4[2].plot(x_axis_res, flux_nash_GM_res[2]/flux_static_values_res[2], color = tableau20[6], linestyle = '-')
    ax4[2].plot(x_axis_res, flux_static_values_res[2]/flux_static_values_res[2], color = tableau20[0], linestyle = '-.')


    ax4[-1].set_xlabel('Carrying capacity ($\overline{R}$) $m_c/m^3$')

    fig4.tight_layout()

    plt.savefig('bottom_up_flux.pdf')

    fig5, ax5 = plt.subplots(1, 2, sharex=True)
    fig5.set_size_inches((8/2.54, 12/2.54))

    ax5[0].plot(x_axis_res, func_nash_GM_res[0], color = tableau20[6], linestyle = '-')
    ax5[0].plot(x_axis_res, func_static_values_res[0], color = tableau20[0], linestyle = '-.')
    ax5[-1].set_xlabel('Carrying capacity ($\overline{R}$) $m_c/m^3$')
    ax5[0].set_ylabel("C consumption/Max")

    ax5[1].plot(x_axis_res, func_nash_GM_res[1], color = tableau20[6], linestyle = '-')
    ax5[1].plot(x_axis_res, func_static_values_res[1], color = tableau20[0], linestyle = '-.')
    ax5[1].set_ylabel("P consumption/Max")

    fig5.tight_layout()

    plt.savefig('functional_response_compare.pdf')


    fig5, ax5 = plt.subplots(2, 1, sharex=True)
    fig5.set_size_inches((8/2.54, 12/2.54))

    ax5[0].plot(x_axis_res, func_nash_GM_res[0], color = tableau20[6], linestyle = '-')
    ax5[0].plot(x_axis_res, func_static_values_res[0], color = tableau20[0], linestyle = '-.')
    ax5[-1].set_xlabel('Carrying capacity ($\overline{R}$) $m_c/m^3$')
    ax5[0].set_ylabel("C consumption/Max")

    ax5[1].plot(x_axis_res, func_nash_GM_res[1], color = tableau20[6], linestyle = '-')
    ax5[1].plot(x_axis_res, func_static_values_res[1], color = tableau20[0], linestyle = '-.')
    ax5[1].set_ylabel("P consumption/Max")

    fig5.tight_layout()

    plt.savefig('functional_response_compare.pdf')

    fig7, ax7 = plt.subplots(2, 1, sharex=True)
    fig7.set_size_inches((8/2.54, 12/2.54))

    ax7[0].plot(x_axis_phi0, func_nash_GM_phi0[:,0], color = tableau20[6], linestyle = '-')
    ax7[0].plot(x_axis_phi0, func_static_values_phi0[:,0], color = tableau20[0], linestyle = '-.')
    ax7[-1].set_xlabel('Max predation pressure ($\\xi$) $m_c/m^3$')
    ax7[0].set_ylabel("C consumption/Max")

    ax7[1].plot(x_axis_phi0, func_nash_GM_phi0[:,1], color = tableau20[6], linestyle = '-')
    ax7[1].plot(x_axis_phi0, func_static_values_phi0[:,1], color = tableau20[0], linestyle = '-.')
    ax7[1].set_ylabel("P consumption/Max")

    fig7.tight_layout()

    plt.savefig('functional_response_compare_phi0.pdf')



    fidelity = 100

    res_m = nash_GM_res[500, 0]
    prey_m = nash_GM_res[500, 1]
    pred_m = nash_GM_res[500, 2]

    params_t = copy.deepcopy(params_ext)
    params_t['resource'] = x_axis_res[500]
    resource_variation = np.linspace(0.001*res_m, res_m, fidelity)
    prey_variation = np.linspace(0.001*prey_m, prey_m, fidelity)
    frp = np.zeros((fidelity, 2))
    frc = np.zeros((fidelity, 2))

    prey_variation = prey_variation[::-1]
    resource_variation = resource_variation[::-1]
    frc[0] = strat_nash_GM_res[500]
    frp[0] = strat_nash_GM_res[500]

    for i in range(1, fidelity):
        frp[i] = combined_strat_finder(params_t, np.array([res_m, prey_variation[i], pred_m]), x0 = frp[i-1])
        frc[i] = combined_strat_finder(params_t, np.array([resource_variation[i], prey_m, pred_m]), x0 = frc[i-1])

    frp = frp[::-1]
    frc = frc[::-1]
    prey_variation = prey_variation[::-1]
    resource_variation = resource_variation[::-1]

    fig6, ax6 = plt.subplots(1, 2, sharey=True)
    fig6.set_size_inches((12/2.54, 8/2.54))
    #plt.title(
    #    "Functional response of predator, P " + str(np.round(pred_m, 2)) + " R " + str(np.round(res_m, 2)))
    ax6[0].plot(prey_variation, prey_variation / (prey_variation + params_ext['nu0']),
             label="P consumption, static", color = tableau20[0], linestyle = '-.')
    ax6[0].plot(prey_variation, frp[:, 0] * frp[:, 1] * prey_variation / (
                frp[:, 0] * frp[:, 1] * prey_variation + params_ext['nu0']), color = tableau20[6], linestyle = '-', 
             label="P consumption, optimal") #Changed to relative functional
    ax6[0].set_xlabel("Prey in $m_c/m^3$")
    ax6[0].set_ylabel("Consumption/Max")
#    plt.savefig("Functional_response_predator.pdf")

#    fig7, ax7 = plt.subplots(1, 1, sharex=True)
#    fig7.set_size_inches((8/2.54, 8/2.54))

    ax6[1].plot(resource_variation, resource_variation / (resource_variation + params_ext['nu0']),
             color = tableau20[0], linestyle = '-.', label="C consumption, static")
    ax6[1].plot(resource_variation,
             frc[:, 0] * resource_variation / (frc[:, 0] * resource_variation + params_ext['nu0']),
             color = tableau20[6], linestyle = '-', label="C consumption, optimal") #alpha = 0.5
    ax6[1].set_xlabel("Resource in $m_c/m^3$")
    #ax6[1].set_ylabel("Consumption rate/Max rate")
    fig6.tight_layout()

    plt.savefig("Functional_response_consumer.pdf")




    fig8, ax8 = plt.subplots(6, 1, sharex=True, gridspec_kw={'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5]})
    fig8.set_size_inches((8/2.54, 16/2.54))

    #ax8[5].set_title('Population dynamics of optimal populations with bottom-up control')
    ax8[0].set_ylabel('Resource, $m_p/m^3$')
    ax8[-1].set_xlabel('Months')


    ax8[0].plot(tim, sol[0, :], color = tableau20[6], linestyle = '-')
    ax8[0].plot(tim, sol_2[0, :], color = tableau20[0], linestyle = '-.')

    ax8[1].set_ylabel('$\\tau_c$')

    ax8[1].plot(tim, strat[0, :], color = tableau20[6], linestyle = '-')

    ax8[2].set_ylabel('Consumer, $m_c/m^3$')

    ax8[2].plot(tim, sol[1, :], color =  tableau20[6], linestyle = '-')
    ax8[2].plot(tim, sol_2[1, :], color = tableau20[0], linestyle = '-.')

    ax8[3].set_ylabel('$\\tau_p \cdot \\tau_c $')

    ax8[3].plot(tim, strat[1, :]*strat[0, :], color = tableau20[6], linestyle = '-')
    ax8[3].set_ylim((0, 0.4))

    ax8[4].set_ylabel('Predator, $m_c/m^3$')

    ax8[4].plot(tim, sol[2, :], color = tableau20[6], linestyle = '-')
    ax8[4].plot(tim, sol_2[2, :], color = tableau20[0], linestyle = '-.')

    ax8[5].set_ylabel('$\\tau_p$')
    ax8[5].set_ylim((0.5, 1))
    ax8[5].plot(tim, strat[1, :], color = tableau20[6], linestyle = '-')

    fig8.tight_layout()

    plt.savefig('simulation_dynamics.pdf')
