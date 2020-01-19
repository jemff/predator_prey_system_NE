import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optm
from multiprocessing import Pool


class PredatorPrey:
    def __init__(self, params, iterations, h, timestep, step_size, opt_prey = True, opt_pred = True):
        self.params = params
        self.iterations = iterations
        self.h = h
        self.timestep = timestep
        self.opt_prey = opt_prey
        self.opt_pred = opt_pred
        self.step_size = step_size

    def optimal_behavior_trajectories_implc(y, self):
        C = y[0]
        N = y[1]
        P = y[2]
        cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = self.params.values()
        taun, taup = self.strat_finder(y)
        Cdot = lam * (cbar - C) - cmax * N * taun * C / (taun * C + cmax)
        Ndot = N * (epsn * cmax * taun * C / (taun * C + cmax) - taup * taun * P * cp * 1 / (
                    taup * taun * N + cp) - mu0 * taun - mu1)
        Pdot = P * (cp * eps * taup * taun * N / (N * taup * taun + cp) - phi0 * taup - phi1)
        return np.array([Cdot.squeeze(), Ndot.squeeze(), Pdot.squeeze()])

    def static_eq_calc(self):
        cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = self.params.values()

        phitild = phi0 + phi1
        mutild = mu0 + mu1
        N_star = phitild * cp / (
                    cp * eps - phitild)  # -(epsn*cmax*lam + cp/(phitild*eps))/(cmax*epsn-cp/phitild-mutild)
        btild = cmax * (1 + N_star / lam) - cbar

        C_star = 1 / 2 * (-btild + np.sqrt(btild ** 2 + 4 * cbar * cmax))

        P_star = (1 / cp + 1 / N_star) * (epsn * lam * (cbar - C_star) - mutild * N_star)
        #    print(np.array([C_star, N_star, P_star]))
        return np.array([C_star, N_star, P_star])

    def jacobian_calculator(f, x, h):
        jac = np.zeros((x.shape[0], x.shape[0]))
        x_m = np.copy(x)
        x_p = np.copy(x)
        for i in range(len((x))):
            x_m[i] -= h
            x_p[i] += h
            jac[:, i] = (f(x_p) - f(x_m)) / (2 * h)
            x_m = np.copy(x)
            x_p = np.copy(x)

        return jac

    def strat_finder(y, self):
        C, N, P = y[0], y[1], y[2]
        taun = 1
        taup = 1
        cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = self.params.values()

        if self.opt_prey is True and self.opt_pred is True:
            taun = min(max(self.opt_taun_find(y), 0), 1)
            taup = min(max(self.opt_taup_find(y, taun), 0), 1)

        elif self.opt_prey is True and self.opt_pred is False:
            taun = min(max(optm.minimize(lambda s_prey: -(cmax * epsn * s_prey * C / (s_prey * C + cmax)
                                                          - cp * taup * s_prey * P / (taup * s_prey * N + cp)
                                                          - mu0 * s_prey - mu1), 0.5).x[0], 0), 1)

        elif self.opt_pred is True and self.opt_prey is False:
            taup = min(max(taup(taun, N), 0), 1)

        return np.array([taun]), np.array([taup])


    def opt_taup_find(y, taun, self):
        C = y[0]
        N = y[1]
        P = y[2]
        cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = self.params.values()
        if taun is 0:
            return 0
        else:
            # taun = np.array([taun])
            # print(np.min(np.concatenate([np.max(np.concatenate([cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun)), np.array([1]) ])), np.array[0]])))
            res = cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun))
            if res < 0 or res > 1:
                tau1 = cp * eps * 1 * taun * N / (N * 1 * taun + cp) - phi0 * 1 - phi1
                tau0 = -phi1
                if tau1 > tau0:
                    res = 1
                else:
                    res = 0
            return res

    def opt_taun_find(y, self):
        C, N, P = y[0], y[1], y[2]
        cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = self.params.values()

        taun_fitness_II = lambda s_prey: \
            epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * self.opt_taup_find(y, s_prey) * s_prey * P / (
                    self.opt_taup_find(y, s_prey) * s_prey * N + cp) - mu0 * s_prey - mu1
        taun_fitness_II_d = lambda s_prey: epsn * cmax ** 2 * C / (s_prey * C + cmax) ** 2 \
                                           - 3 / 2 * (N ** 2 * cp * (eps / (phi0 * N)) ** (1 / 2) * s_prey ** (
                    5 / 2)) ** (-1) - mu0

        linsp = np.linspace(0.001, 1, 100)
        comparison_numbs = (taun_fitness_II_d(linsp))
        if len(np.where(comparison_numbs > 0)[0]) is 0 or len(np.where(comparison_numbs < 0)[0]) is 0:
            t0 = taun_fitness_II(0)
            t1 = taun_fitness_II(1)
            if t0 > t1:
                max_cands = 0
            else:
                max_cands = 1

        else:
            maxi_mill = linsp[np.where(comparison_numbs > 0)[0][-1]]
            max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[0.001, maxi_mill], method='brentq').root

        return max_cands

    def optimal_behavior_trajectories(self, t, y, taun=1, taup=1):
        C = y[0]
        N = y[1]
        P = y[2]
        cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = self.params.values()
        if self.seasons is True:
            Cdot = lam * (cbar + 0.5 * cbar * np.cos(t * np.pi / 180) - C) - N * taun * C / (
                        taun * C + cmax)  # t is one month
        else:
            Cdot = lam * (cbar - C) - cmax * N * taun * C / (taun * C + cmax)
        flux_c_to_n = N * taun * C / (taun * C + cmax)
        flux_n_to_p = N * taup * taun * P * cp * 1 / (taup * taun * N + cp)  # Now these values are calculated twice..
        Ndot = N * (epsn * cmax * taun * C / (taun * C + cmax) - taup * taun * P * cp * 1 / (
                    taup * taun * N + cp) - mu0 * taun - mu1)
        Pdot = P * (cp * eps * taup * taun * N / (N * taup * taun + cp) - phi0 * taup - phi1)
        return np.array([Cdot, Ndot, Pdot, flux_c_to_n, flux_n_to_p])

    def semi_implicit_euler(self, t_final, y0, step_size, f, params):
        solution = np.zeros((y0.shape[0], int(t_final / step_size)))
        flux = np.zeros((2, int(t_final / step_size)))
        strat = np.zeros((2, int(t_final / step_size)))
        t = np.zeros(int(t_final / step_size))
        solution[:, 0] = y0
        for i in range(1, int(t_final / step_size)):
            taun_best, taup_best = self.strat_finder(solution[:, i - 1], params)
            strat[:, i] = taun_best, taup_best

            fwd = f(t[i], solution[:, i - 1], taun_best, taup_best)
            t[i] = t[i - 1] + step_size
            solution[:, i] = solution[:, i - 1] + step_size * fwd[0:3]
            flux[:, i] = fwd[3:]
        return t, solution, flux, strat

    def dynamic_pred_prey(self):
        t_end = 120

        init = np.array([0.8, 0.5, 0.5])
        time_b, sol_basic, flux_bas, strat_bas = self.semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
        self.optimal_behavior_trajectories(t, y, taun=tn, taup=tp), self.params)
        base_case = np.array([sol_basic[0, -1], sol_basic[1, -1], sol_basic[2, -1]])

        tim, sol, flux, strats = self.semi_implicit_euler(t_end, base_case, 0.001, lambda t, y, tn, tp:
        self.optimal_behavior_trajectories(t, y, taun=tn, taup=tp))

        strats = np.zeros((self.its, 2))
        fluxes = np.zeros((self.its, 2))
        pops = np.zeros((self.its, 3))
        t_end = 100
        params = copy.deepcopy(self.params)
        for i in range(0, its):
            params['resource'] = self.base + self.step_size * i
            init = np.array([sol[0, -1], sol[1, -1], sol[2, -1]])
            tim, sol, flux, strat = self.semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
            self.optimal_behavior_trajectories(t, y, taun=tn, taup=tp))

            tim_OG, sol_OG, flux_OG, strat_OG = self.semi_implicit_euler(t_end, init, 0.001, lambda t, y, tn, tp:
            self.optimal_behavior_trajectories(t, y, taun=tn, taup=tp))
            pops[i] = np.sum((sol-sol_OG)*0.001, axis = 1) #sol[:,-1] - sol_OG[:,-1]
            strats[i] = strat[:, -1]
            if strats[i, 0] is 0 or strats[i, 1] is 0:
                print(strat)
            fluxes[i] = np.sum((flux - flux_OG) * 0.001, axis=1)
            # print(fluxes[i], pops[i], phi0_dyn, base+step_size*i)
        return np.hstack([strats, pops, fluxes])


    base = 14
    its = 1
    step_size = 1
    step_size_phi = 0.05
    cbar = base
    phi0_base = 0.4

    cmax = 2
    mu0 = 0.2
    mu1 = 0.3
    eps = 0.7
    epsn = 0.7
    cp = 2
    phi0 = phi0_base
    phi1 = 0.2
    lam = 0.5

    opt_prey = True
    opt_pred = True
    # Currently the affinity is set to 1!!

    params_ext = {'cmax': cmax, 'mu0': mu0, 'mu1': mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0': phi0, 'phi1': phi1,
                  'resource': base, 'lam': lam}
