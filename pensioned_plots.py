
#    ax3[1].set_ylabel('C Production')

#
#    ax3[1].plot(x_axis_phi0, flux_static_values_phi0[:, 1]/flux_static_values_phi0[:, 1], color = tableau20[0], linestyle = '-')

  #  ax3[1].plot(x_axis_phi0, flux_nash_Gill_phi0[:, 1]/flux_static_values_phi0[:, 1],  color = 'Blue', linestyle = '-')

#    ax3[2].set_ylabel('P Production')

#
#    ax3[2].plot(x_axis_phi0, flux_static_values_phi0[:, 2]/flux_static_values_phi0[:, 2], color = tableau20[0], linestyle = '-')

#    ax3[0].set_xlabel('Top predation pressure ($\\xi$) $1/month$')



#    fig4, ax4 = plt.subplots(3, 1, sharex=True)
#    fig4.set_size_inches((8/2.54, 12/2.54))
    #ax4[2].set_title('Ratio of flux compared to static system, bottom-up control')

#    ax4[0].set_ylabel('R Production')
#    ax4[0].plot(x_axis_res, flux_nash_GM_res[0]/flux_static_values_res[0], color = tableau20[6], linestyle = '-')
#    ax4[0].plot(x_axis_res, flux_static_values_res[0]/flux_static_values_res[0], color = tableau20[0], linestyle = '-')

#    ax4[1].set_ylabel('C Production')

#    ax4[1].plot(x_axis_res, flux_nash_GM_res[1]/flux_static_values_res[1], color = tableau20[6], linestyle = '-')
#    ax4[1].plot(x_axis_res, flux_static_values_res[1]/flux_static_values_res[1], color = tableau20[0], linestyle = '-')

#    ax4[2].set_ylabel('P Production')

#    ax4[2].plot(x_axis_res, flux_nash_GM_res[2]/flux_static_values_res[2], color = tableau20[6], linestyle = '-')
#    ax4[2].plot(x_axis_res, flux_static_values_res[2]/flux_static_values_res[2], color = tableau20[0], linestyle = '-')


#    ax4[-1].set_xlabel('Carrying capacity ($\overline{R}$) $m_c\cdot m^{-3}$')

#    fig4.tight_layout()
#    if settings['linear'] is False:
#        plt.savefig('bottom_up_flux.pdf')
#    else:
#        plt.savefig('bottom_up_flux_linear.pdf')


    #    fig.tight_layout()
#    if settings['linear'] is False:
#        plt.savefig('bottom_up_pop_dyn.pdf')
#    else:
#        plt.savefig('bottom_up_pop_dyn_linear.pdf')

 #   fig2, ax2 = plt.subplots(6, 1, sharex=True, gridspec_kw={'height_ratios': [1, 0.5, 1, 0.5, 1, 0.5]})
 #   fig2.set_size_inches((8/2.54, 16/2.54))

    #ax2[5].set_title('Population dynamics of optimal populations with top-down control')

    ax2[0].set_ylabel('Resource, $m_c\cdot m^{-3}$')
    ax2[-1].set_xlabel('Top predation pressure ($\\xi$) $month^{-1}$')

    ax2[0].plot(x_axis_phi0, nash_GM_phi0[:, 0], color = tableau20[6], linestyle = '-')
    ax2[0].plot(x_axis_phi0, static_values_phi0[:, 0], color = tableau20[0], linestyle = '-')

    ax2[1].set_ylabel('$\\tau_c$')

    ax2[1].fill_between(x_axis_phi0, strat_nash_GM_phi0[:, 0], y2 = min(strat_nash_GM_phi0[:, 0]), color = tableau20[6], linestyle = '-', alpha = 0.5)

    ax2[2].set_ylabel('Consumer, $m_c\cdot m^{-3}$')

    ax2[2].plot(x_axis_phi0, nash_GM_phi0[:, 1], color = tableau20[6], linestyle = '-')
    ax2[2].plot(x_axis_phi0, static_values_phi0[:, 1], color = tableau20[0], linestyle = '-')

    ax2[3].set_ylabel('$\\tau_c \cdot \\tau_p $')

    ax2[3].fill_between(x_axis_phi0, strat_nash_GM_phi0[:, 1]*strat_nash_GM_phi0[:, 0], y2 = min(strat_nash_GM_phi0[:, 1]*strat_nash_GM_phi0[:, 0]), color = tableau20[6], alpha = 0.5, linestyle = '-')

    ax2[4].set_ylabel('Predator, $m_c\cdot m^{-3}$')

    ax2[4].plot(x_axis_phi0, nash_GM_phi0[:, 2], color = tableau20[6], linestyle = '-')
    ax2[4].plot(x_axis_phi0, static_values_phi0[:, 2], color = tableau20[0], linestyle = '-')

    ax2[5].set_ylabel('$\\tau_p$')

    ax2[5].fill_between(x_axis_phi0, strat_nash_GM_phi0[:, 1], y2 = min(strat_nash_GM_phi0[:, 1]), alpha = 0.5, color = tableau20[6], linestyle = '-')

    fig2.tight_layout()

ax5[0].plot(x_axis_res, func_nash_GM_res[0], color=tableau20[6], linestyle='-')
ax5[0].plot(x_axis_res, func_static_values_res[0], color=tableau20[0], linestyle='-')
ax5[-1].set_xlabel('Carrying capacity ($\overline{R}$) $m_c\cdot m^{-3}$')
ax5[0].set_ylabel("C consumption/Max")

ax5[1].plot(x_axis_res, func_nash_GM_res[1], color=tableau20[6], linestyle='-')
ax5[1].plot(x_axis_res, func_static_values_res[1], color=tableau20[0], linestyle='-')
ax5[1].set_ylabel("P consumption/Max")

fig5.tight_layout()
if settings['linear'] is False:
    plt.savefig('functional_response_compare.pdf')
else:
    plt.savefig('functional_response_compare_linear.pdf')

fig7, ax7 = plt.subplots(2, 1, sharex=True)
fig7.set_size_inches((8 / 2.54, 12 / 2.54))

ax7[0].plot(x_axis_phi0, func_nash_GM_phi0[:, 0], color=tableau20[6], linestyle='-')
ax7[0].plot(x_axis_phi0, func_static_values_phi0[:, 0], color=tableau20[0], linestyle='-')
ax7[-1].set_xlabel('Max predation pressure ($\\xi$) $month^{-1}$')
ax7[0].set_ylabel("C consumption/Max")

ax7[1].plot(x_axis_phi0, func_nash_GM_phi0[:, 1], color=tableau20[6], linestyle='-')
ax7[1].plot(x_axis_phi0, func_static_values_phi0[:, 1], color=tableau20[0], linestyle='-')
ax7[1].set_ylabel("P consumption/Max")

fig7.tight_layout()
if settings['linear'] is False:
    plt.savefig('functional_response_compare_phi0.pdf')
else:
    plt.savefig('functional_response_compare_phi0_linear.pdf')
