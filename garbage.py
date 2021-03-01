
manual_max = False
if manual_max is True:
    for i in range(int(base*10)):
        for j in range(int(base*10)):
            for k in range(int(base*10)):
                test_num = np.array([0.1+0.1*i, 0.1+0.1*j, 0.1+0.1*k])
                is_opt = np.abs(optimal_behavior_trajectories(test_num, params_ext))
                if np.sum(is_opt) < 0.5*10**(-1):
                    print(test_num)

    print("I'm done")
its = 0
if its > 0:
    taun_grid = np.zeros((its, its))
    taup_grid = np.zeros((its, its))
    res_nums = np.zeros((its, its))
    prey_nums = np.zeros((its, its))
    pred_nums = np.zeros((its, its))
    start_point = np.copy(sol_3.x)
    x_ext = np.zeros(3)
    x_prev = np.zeros(3)


    for i in range(its):
        params_ext['resource'] = base + step_size*i
        if i is 0:
            x_ext = np.copy(start_point)
        else:
            x_ext = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0=x_ext, method='hybr').x
    #    jac_temp = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext), x_ext, h)
    #    print(np.linalg.eig(jac_temp)[0])
        for j in range(its):
            params_ext['phi0'] = phi0_base+j*step_size_phi
            if j is 0:
                x_prev = np.copy(x_ext)
                sol_temp = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0=x_prev, method='hybr')
                x_prev = sol_temp.x
            else:
                sol_temp = optm.root(lambda y: optimal_behavior_trajectories(y, params_ext), x0=x_prev, method='hybr')
                x_prev = sol_temp.x
 #               if sol_temp.success is False or j is 0:
 #                   print(sol_temp.message)

            jac_temp = jacobian_calculator(lambda y: optimal_behavior_trajectories(y, params_ext), x_prev, h)
            #print(np.linalg.eig(jac_temp)[0], x_prev, params_ext["phi0"], params_ext["resource"])
#            print(np.real(np.linalg.eig(jac_temp)[0].max()))
            eigen_max[i, j] = np.real(np.linalg.eig(jac_temp)[0].max())
            res_nums[i, j] = x_prev[0]
            prey_nums[i, j] = x_prev[1]
            pred_nums[i, j] = x_prev[2]
            taun_grid[i, j], taup_grid[i, j] = strat_finder(x_prev, params_ext)
#            print(taup_grid[i, j], j)
#            if eigen_max[i, j] > 0:
#                print("Oh no!")

#    print(prey_nums[:, 0])
    ran = [base, base+its*step_size, 100*phi0_base, 100*(phi0_base+its*step_size_phi)]
    print("Ran", ran)



    temporary_thingy = np.zeros((its, its))
    temporary_thingy[0,:] = 1
    heatmap_plotter(temporary_thingy, "test", "test", ran)
    heatmap_plotter(res_nums.T, 'Resource g/m^3', "resource_conc", ran)
    heatmap_plotter(prey_nums.T, 'Prey g/m^3', "prey_conc", ran)
    heatmap_plotter(pred_nums.T, 'Predators g/m^3', "pred_conc", ran)
    heatmap_plotter(taun_grid.T, 'Prey foraging intensity', "prey_for", ran)
    heatmap_plotter(taup_grid.T, 'Predator foraging intensity', "pred_for", ran)
    heatmap_plotter(eigen_max.T, 'Eigenvalues', "Eigenvalues", ran)

#    plt.figure()
#    plt.title('Resource kg/m^3')
#    #    plt.colorbar(res_nums, fraction=0.046, pad=0.04)
#    plt.xlabel("Cbar, kg/m^3")
#    plt.ylabel("phi0, kg/(m^3 * week)")
#
#    ax = plt.gca()
#    im = ax.imshow(res_nums, cmap='Reds', extent =ran)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)

#    plt.colorbar(im, cax=cax)

#    plt.savefig("resource_conc.png")
#    plt.show()


manual_max = False
if manual_max is True:
    for i in range(int(base*10)):
        for j in range(int(base*10)):
            for k in range(int(base*10)):
                test_num = np.array([0.1+0.1*i, 0.1+0.1*j, 0.1+0.1*k])
                is_opt = np.abs(optimal_behavior_trajectories(test_num, params_ext))
                if np.sum(is_opt) < 0.5*10**(-1):
                    print(test_num)

    print("I'm done")