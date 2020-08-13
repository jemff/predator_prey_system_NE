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
phi1 = cost_of_living[1]#*2/3 #/3 #*2#/2#*5
phi0 = cost_of_living[1]#*4/3 #The problem is that nash equiibrium does not exist in the low levels...
eps = 0.7
epsn = 0.7

cmax, cp = growth_max
mu0 = 0 # cost_of_living[0] #/3 #*2#/2#*5 #*60 #/2
mu1 = cost_of_living[0]#/3 #*2 #*2#/2#*5 #*2 #*10 #/2
nu0 = nu[0] #nu
nu1 = nu[1] #nu

its = 600

params_ext = {'cmax': cmax, 'mu0': mu0, 'mu1': mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0': phi0, 'phi1': phi1, 
              'resource': base, 'lam': lam, 'nu0': nu0, 'nu1': nu1}

if settings['gen_data'] is True:

    init = np.array([0.9500123,  0.4147732,  0.01282899 ])

    reverse = True
    start = 5
    stop = 30
    x_axis_res = np.linspace(start, stop, its)

    nash_GM_res, strat_nash_GM_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse)
    nash_Gill_res, strat_nash_Gill_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, Gill = True)

    stack_GM_res, strat_stack_GM_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, nash = False)
    stack_Gill_res, strat_stack_Gill_res = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, Gill = True, nash = False, root = False)

    start = 0.6*phi0
    stop = 1.1*phi0
    x_axis_phi0 = np.linspace(start, stop, its)

    nash_GM_phi0, strat_nash_GM_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, type = 'phi0')
    nash_Gill_phi0, strat_nash_Gill_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, type = 'phi0', Gill = True)
    stack_GM_phi0, strat_stack_GM_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, type = 'phi0', nash = False)

#    stack_Gill_phi0, strat_stack_Gill_phi0 = an_sol.continuation_func_ODE(an_sol.optimal_behavior_trajectories_version_2, init, params_ext, start, stop, its, reverse = reverse, type = 'phi0', Gill = True, nash = False, root = False)

    with open('bifurcation_data_appendix.npy', 'wb') as f:
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

   #     np.save(f, stack_Gill_phi0)
   #     np.save(f, strat_stack_Gill_phi0)

elif settings['gen_data'] is False:
    with open('bifurcation_data_appendix.npy', 'rb') as f:  # Save time by not generating all the data every time.

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

    #    stack_Gill_phi0 = np.load(f)
    #    strat_stack_Gill_phi0 = np.load(f)


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
        #params_temp_phi0['phi0'] = x_axis_phi0[i]
        #static_values_phi0[i] = static_eq_calc(params_temp_phi0)

        #flux_nash_GM_phi0[i] = an_sol.flux_calculator(nash_GM_phi0[i, 0], nash_GM_phi0[i, 1], nash_GM_phi0[i, 2],
        #                                              strat_nash_GM_phi0[i, 0], strat_nash_GM_phi0[i, 1], params_temp_phi0)

        #flux_nash_Gill_phi0[i] = an_sol.flux_calculator(nash_Gill_phi0[i, 0], nash_Gill_phi0[i, 1], nash_Gill_phi0[i, 2],
        #                                               strat_nash_Gill_phi0[i, 0], strat_nash_Gill_phi0[i, 1],
        #                                                params_temp_phi0)
        #flux_static_values_phi0[i] = an_sol.flux_calculator(static_values_phi0[i, 0], static_values_phi0[i, 1],
        #                                                    static_values_phi0[i, 2], 1, 1, params_temp_phi0)

        #func_nash_GM_phi0[i] = an_sol.frp_calc(nash_GM_phi0[i, 0], nash_GM_phi0[i, 1], nash_GM_phi0[i, 2],
        #                                       strat_nash_GM_phi0[i, 0], strat_nash_GM_phi0[i, 1], params_temp_phi0)
        #func_nash_Gill_phi0[i] = an_sol.frp_calc(nash_Gill_phi0[i, 0], nash_Gill_phi0[i, 1], nash_Gill_phi0[i, 2],
        #                                         strat_nash_Gill_phi0[i, 0], strat_nash_Gill_phi0[i, 1],
        #                                         params_temp_phi0)
        #func_static_values_phi0[i] = an_sol.frp_calc(static_values_phi0[i, 0], static_values_phi0[i, 1],
        #                                             static_values_phi0[i, 2], 1, 1, params_temp_phi0)

        params_temp_res['resource'] = x_axis_res[i]
        static_values_res[i] = static_eq_calc(params_temp_res)

        params_temp_phi0['phi0'] = x_axis_phi0[i]
        static_values_phi0[i] = static_eq_calc(params_temp_phi0)

        flux_nash_GM_phi0[i] = an_sol.flux_calculator(nash_GM_phi0[i, 0], nash_GM_phi0[i, 1], nash_GM_phi0[i, 2],
                                                      strat_nash_GM_phi0[i, 0], strat_nash_GM_phi0[i, 1],
                                                      params_temp_phi0)

        flux_nash_Gill_phi0[i] = an_sol.flux_calculator(nash_Gill_res[i, 0], nash_Gill_res[i, 1], nash_Gill_res[i, 2],
                                                               strat_nash_Gill_res[i, 0], strat_nash_Gill_res[i, 1],
                                                                params_temp_phi0)
        flux_static_values_phi0[i] = an_sol.flux_calculator(static_values_phi0[i, 0], static_values_phi0[i, 1],
                                                            static_values_phi0[i, 2], 1, 1, params_temp_phi0)

        func_nash_GM_phi0[i] = an_sol.frp_calc(nash_GM_phi0[i, 0], nash_GM_phi0[i, 1], nash_GM_phi0[i, 2],
                                               strat_nash_GM_phi0[i, 0], strat_nash_GM_phi0[i, 1], params_temp_phi0)
        func_nash_Gill_phi0[i] = an_sol.frp_calc(nash_Gill_phi0[i, 0], nash_Gill_phi0[i, 1], nash_Gill_phi0[i, 2],
                                                        strat_nash_Gill_phi0[i, 0], strat_nash_Gill_phi0[i, 1],
                                                         params_temp_phi0)
        func_static_values_phi0[i] = an_sol.frp_calc(static_values_phi0[i, 0], static_values_phi0[i, 1],
                                                     static_values_phi0[i, 2], 1, 1, params_temp_phi0)

    flux_nash_GM_res = an_sol.flux_calculator(nash_GM_res[:, 0], nash_GM_res[:, 1], nash_GM_res[:, 2], strat_nash_GM_res[:, 0], strat_nash_GM_res[:, 1], params_ext)
    flux_nash_Gill_res = an_sol.flux_calculator(nash_Gill_res[:, 0], nash_Gill_res[:, 1], nash_Gill_res[:, 2], strat_nash_Gill_res[:, 0], strat_nash_Gill_res[:, 1], params_ext)
    flux_stack_GM_res = an_sol.flux_calculator(stack_GM_res[:, 0], stack_GM_res[:, 1], stack_GM_res[:, 2], strat_stack_GM_res[:, 0], strat_stack_GM_res[:, 1], params_ext)
    flux_static_values_res = an_sol.flux_calculator(static_values_res[:, 0], static_values_res[:, 1], static_values_res[:, 2], ones, ones, params_ext)

    func_nash_GM_res = an_sol.frp_calc(nash_GM_res[:, 0], nash_GM_res[:, 1], nash_GM_res[:, 2], strat_nash_GM_res[:, 0], strat_nash_GM_res[:, 1], params_ext)
    func_nash_Gill_res = an_sol.frp_calc(nash_Gill_res[:, 0], nash_Gill_res[:, 1], nash_Gill_res[:, 2], strat_nash_Gill_res[:, 0], strat_nash_Gill_res[:, 1], params_ext)
    func_static_values_res = an_sol.frp_calc(static_values_res[:, 0], static_values_res[:, 1], static_values_res[:, 2], ones, ones, params_ext)




    fig, ax = plt.subplots(6, 1, sharex=True, gridspec_kw={'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5]})
    fig.set_size_inches((8/2.54, 16/2.54))

    #ax[5].set_title('Population dynamics of optimal populations with bottom-up control')
    ax[0].set_ylabel('Resource, $m_c/m^3$')
    ax[-1].set_xlabel('Carrying capacity $(\overline{R})$ $m_c/m^3$')


    ax[0].plot(x_axis_res, nash_GM_res[:, 0], color = tableau20[6], linestyle = '-')
    ax[0].plot(x_axis_res, nash_Gill_res[:, 0], color = tableau20[10], linestyle = '-')
    ax[0].plot(x_axis_res, stack_GM_res[:, 0], color = tableau20[8], linestyle = 'dotted')
    ax[0].plot(x_axis_res, static_values_res[:, 0], color = tableau20[0], linestyle = '-.')

    ax[1].set_ylabel('$\\tau_c$')

    ax[1].plot(x_axis_res, strat_nash_GM_res[:, 0], color = tableau20[6], linestyle = '-')
    ax[1].plot(x_axis_res, strat_nash_Gill_res[:, 0], color = tableau20[10], linestyle = '-')
    ax[1].plot(x_axis_res, strat_stack_GM_res[:, 0], color = tableau20[8], linestyle = 'dotted')

    ax[2].set_ylabel('Consumer, $m_c/m^3$')
    
    ax[2].plot(x_axis_res, nash_Gill_res[:, 1], color = tableau20[10], linestyle = '-')
    ax[2].plot(x_axis_res, nash_GM_res[:, 1], color =  tableau20[6], linestyle = '-')
    ax[2].plot(x_axis_res, stack_GM_res[:, 1], color = tableau20[8], linestyle = 'dotted')
    ax[2].plot(x_axis_res, static_values_res[:, 1], color = tableau20[0], linestyle = '-.')

    ax[3].set_ylabel('$\\tau_p \cdot \\tau_C  $')

    ax[3].plot(x_axis_res, strat_nash_GM_res[:, 1]*strat_nash_GM_res[:, 0], color = tableau20[6], linestyle = '-')
    ax[3].plot(x_axis_res, strat_nash_Gill_res[:, 1]*strat_nash_GM_res[:, 0], color = tableau20[10], linestyle = '-')
    ax[3].plot(x_axis_res, strat_stack_GM_res[:, 1]*strat_nash_GM_res[:, 0])

    ax[4].set_ylabel('Predator, $m_c/m^3$')

    ax[4].plot(x_axis_res, nash_Gill_res[:, 2], color = tableau20[10], linestyle = '-')
    ax[4].plot(x_axis_res, nash_GM_res[:, 2], color = tableau20[6], linestyle = '-')
    ax[4].plot(x_axis_res, stack_GM_res[:, 2], color = tableau20[8], linestyle = 'dotted')
    ax[4].plot(x_axis_res, static_values_res[:, 2], color = tableau20[0], linestyle = '-.')

    ax[5].set_ylabel('$\\tau_p$')
    ax[5].set_ylim((0.5, 1))
    ax[5].plot(x_axis_res, strat_nash_GM_res[:, 1], color = tableau20[6], linestyle = '-')
    ax[5].plot(x_axis_res, strat_nash_Gill_res[:, 1], color = tableau20[10], linestyle = '-')
    ax[5].plot(x_axis_res, strat_stack_GM_res[:, 1], color = tableau20[6], linestyle = '-')

    fig.tight_layout()

    plt.savefig('bottom_up_pop_dyn_appendix.pdf')

    fig2, ax2 = plt.subplots(6, 1, sharex=True, gridspec_kw={'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5]})
    fig2.set_size_inches((8/2.54, 16/2.54))

    #ax2[5].set_title('Population dynamics of optimal populations with top-down control')

    ax2[0].set_ylabel('Resource, $m_c/m^3$')
    ax2[-1].set_xlabel('Top predation pressure $(\\xi)$ $m_c/(m^3 month)$')

    ax2[0].plot(x_axis_phi0, nash_GM_phi0[:, 0], color = tableau20[6], linestyle = '-')
    ax2[0].plot(x_axis_phi0, nash_Gill_phi0[:, 0], color = tableau20[10], linestyle = '-')
    ax2[0].plot(x_axis_phi0, static_values_phi0[:, 0], color = tableau20[0], linestyle = '-.')

    ax2[1].set_ylabel('$\\tau_c$')

    ax2[1].plot(x_axis_phi0, strat_nash_GM_phi0[:, 0], color = tableau20[6], linestyle = '-')
    ax2[1].plot(x_axis_phi0, strat_nash_Gill_phi0[:, 0], color = tableau20[10], linestyle = '-')

    ax2[2].set_ylabel('Consumer, $m_c/m^3$')

    ax2[2].plot(x_axis_phi0, nash_Gill_phi0[:, 1], color = tableau20[10], linestyle = '-')
    ax2[2].plot(x_axis_phi0, nash_GM_phi0[:, 1], color = tableau20[6], linestyle = '-')
    ax2[2].plot(x_axis_phi0, static_values_phi0[:, 1], color = tableau20[0], linestyle = '-.')

    ax2[3].set_ylabel('$\\tau_c \cdot \\tau_p$')

    ax2[3].plot(x_axis_phi0, strat_nash_GM_phi0[:, 1]*strat_nash_GM_phi0[:, 0], color = tableau20[6], linestyle = '-')
    ax2[3].plot(x_axis_phi0, strat_nash_Gill_phi0[:, 1]*strat_nash_Gill_phi0[:, 0], color = tableau20[10], linestyle = '-')

    ax2[4].set_ylabel('Predator, $m_c/m^3$')

    ax2[4].plot(x_axis_phi0, nash_Gill_phi0[:, 2], color = tableau20[10], linestyle = '-')
    ax2[4].plot(x_axis_phi0, nash_GM_phi0[:, 2], color = tableau20[6], linestyle = '-')
    ax2[4].plot(x_axis_phi0, static_values_phi0[:, 2], color = tableau20[0], linestyle = '-.')

    ax2[5].set_ylabel('$\\tau_p$')

    ax2[5].plot(x_axis_phi0, strat_nash_GM_phi0[:, 1], color = tableau20[6], linestyle = '-')
    ax2[5].plot(x_axis_phi0, strat_nash_Gill_phi0[:, 1], color = tableau20[10], linestyle = '-')

    fig2.tight_layout()
    plt.savefig('top_down_pop_dyn_appendix.pdf')

    fig3, ax3 = plt.subplots(3, 1, sharex=True)
    fig3.set_size_inches((8/2.54, 12/2.54))

    #ax3[2].set_title('Ratio of flux compared to static system, top-down control')

    ax3[0].set_ylabel('R production')

    ax3[0].plot(x_axis_phi0, flux_nash_GM_phi0[:, 0]/flux_static_values_phi0[:, 0], color = tableau20[6], linestyle = '-')

    ax3[0].plot(x_axis_phi0, flux_nash_Gill_phi0[:, 0]/flux_static_values_phi0[:, 0],  color = tableau20[10], linestyle = '-')

    ax3[1].set_ylabel('C Production')

    ax3[1].plot(x_axis_phi0, flux_nash_GM_phi0[:, 1]/flux_static_values_phi0[:, 1], color = tableau20[6], linestyle = '-')
    ax3[1].plot(x_axis_phi0, flux_nash_Gill_phi0[:, 1]/flux_static_values_phi0[:, 1],  color = tableau20[10], linestyle = '-')

    ax3[2].set_ylabel('P Production')

    ax3[2].plot(x_axis_phi0, flux_nash_GM_phi0[:, 0]/flux_static_values_phi0[:, 2], color = tableau20[6], linestyle = '-')
    ax3[2].plot(x_axis_phi0, flux_nash_Gill_phi0[:, 0]/flux_static_values_phi0[:, 2],  color = tableau20[10], linestyle = '-')

    ax3[-1].set_xlabel('Top predation pressure $(\\xi)$ m_c/(m^3 day)$')

    fig3.tight_layout()
    plt.savefig('top_down_flux_appendix.pdf')


    fig4, ax4 = plt.subplots(3, 1, sharex=True)
    fig4.set_size_inches((8/2.54, 12/2.54))
    #ax4[2].set_title('Ratio of flux compared to static system, bottom-up control')

    ax4[0].set_ylabel('R production')
    ax4[0].plot(x_axis_res, flux_nash_GM_res[0]/flux_static_values_res[0], color = tableau20[6], linestyle = '-')
    ax4[0].plot(x_axis_res, flux_nash_Gill_res[0]/flux_static_values_res[0], color = tableau20[10], linestyle = '-')
    ax4[0].plot(x_axis_res, flux_stack_GM_res[0]/flux_static_values_res[0], color = tableau20[8], linestyle = 'dotted')

    ax4[1].set_ylabel('C Production')

    ax4[1].plot(x_axis_res, flux_nash_GM_res[1]/flux_static_values_res[1], color = tableau20[6], linestyle = '-')
    ax4[1].plot(x_axis_res, flux_nash_Gill_res[1]/flux_static_values_res[1],  color = tableau20[10], linestyle = '-')
    ax4[1].plot(x_axis_res, flux_stack_GM_res[1]/flux_static_values_res[1], color = tableau20[8], linestyle = 'dotted')

    ax4[2].set_ylabel('P Production')

    ax4[2].plot(x_axis_res, flux_nash_GM_res[2]/flux_static_values_res[2], color = tableau20[6], linestyle = '-')
    ax4[2].plot(x_axis_res, flux_nash_Gill_res[2]/flux_static_values_res[2], color = tableau20[10], linestyle = '-')
    ax4[2].plot(x_axis_res, flux_stack_GM_res[2]/flux_static_values_res[2], color = tableau20[8], linestyle = 'dotted')


    ax4[-1].set_xlabel('Carrying capacity $(\overline{R})$ $m_c/m^3$')

    fig4.tight_layout()

    plt.savefig('bottom_up_flux_appendix.pdf')

    fig6, ax6 = plt.subplots(1, 1, sharey=True)
    fig6.set_size_inches((8/2.54, 4/2.54))
    ax6.axis('off')
    ax6.plot(np.array([1]), np.array([1]) , color=tableau20[6], linestyle='-', label = 'Growth - Mortality, Nash Equilibrium')
    ax6.plot(np.array([1]), np.array([1]), color=tableau20[10], linestyle='-', label = 'Gilliams Rule, Nash Equilibrium')
    ax6.plot(np.array([1]), np.array([1]), color=tableau20[8], linestyle='dotted', label = 'Growth - Mortality, Stackelberg Equilibrium')
    ax6.plot(np.array([1]), np.array([1]), color=tableau20[0], linestyle='-.', label = 'Model with static behavior')
    ax6.legend(loc='center left')
    fig6.tight_layout()

    plt.savefig('Legends_appendix.pdf')


