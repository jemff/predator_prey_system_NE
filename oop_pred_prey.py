import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optm
from multiprocessing import Pool



def num_derr(f, x, h):
    x_m = float(np.copy(x))
    x_p = float(np.copy(x))
    x_m -= h
    x_p += h
    derr = (f(x_p) - f(x_m))/(2*h)
#    print(derr)
    return derr

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




class PredatorPrey:
    def __init__(self, mass_vector, base, ivp, opt_prey = True, opt_pred = True, nash = True, metabolism = True):
        cost_of_living, nu, growth_max, lam = self.parameter_calculator_mass(mass_vector, v=0.1)
        base = base
        phi1 = cost_of_living[1]  # *2/3 #/3 #*2#/2#*5
        phi0 = cost_of_living[1]  # *4/3 #The problem is that nash equiibrium does not exist in the low levels...
        eps = 0.7
        epsn = 0.7

        cmax, cp = growth_max
        #cmax = 1/5 *cmax
        mu0 = 0 # cost_of_living[0]/2 #*2  # cost_of_living[0] #/3 #*2#/2#*5 #*60 #/2 #6/7 was not eough to provoke an effect... There seemed to be an effect when using least-squares with 9/10 around 0.07, but might have been an artifact
        mu1 = cost_of_living[0] # /3 #*2 #*2#/2#*5 #*2 #*10 #/2
        nu0 = nu[0]  # nu
        nu1 = nu[1]  # nu

        self.params =  {'cmax' : cmax, 'mu0' : mu0, 'mu1' : mu1, 'eps': eps, 'epsn': epsn, 'cp': cp, 'phi0':phi0, 'phi1': phi1,
          'resource': base, 'lam':lam, 'nu0':nu0, 'nu1': nu1}
        self.opt_prey = opt_prey
        self.opt_pred = opt_pred
        self.strat = np.array([0.5, 0.5])
        self.population = ivp
        self.metabolism = metabolism

    def parameter_calculator_mass(self, mass_vector, alpha=15, b=330 / 12, v=0.05):
        # alpha = 15
        # b = 330/12
        # v = 0.1 #/12
        maximum_consumption_rate = alpha * mass_vector[1:] ** (0.75)

        ci = v * maximum_consumption_rate
        ci[0] = ci[0]
        # ci[-1] = ci[-1]*0.1
        # print(maximum_consumption_rate)
        r0 = 0.1
        nu = alpha / b * mass_vector[1:] ** (0)
        # print(ci)

        return ci, nu, maximum_consumption_rate, r0

    def update_pop(self, time_step = 0.0005):
        self.population += time_step*self.optimal_behavior_trajectories()


    def taun_linear(self, taup):
        root_object = optm.root(lambda strat: num_derr(
            lambda s_prey: self.taun_fitness_II_linear(s_prey, taup), strat, 0.00001),
                                x0=self.strat[0])

        val_max = max(min(root_object.x, 1), 0)
        strategy_selection = np.array([0, root_object.x, 1])
        return val_max #Should use strategy selection algorithm

    def static_eq_calc(params):
        cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

        phitild = phi0 + phi1
        mutild = mu0 + mu1
        C_star = phitild * nu1 / (eps * cp - phitild)
        gam = nu0 - cbar + (cmax / lam) * C_star
        #    print(gam, gam**2, 4*cbar*nu0, np.sqrt(gam**2+4*cbar*nu0))
        R_star = (-gam + np.sqrt(gam ** 2 + 4 * cbar * nu0)) / 2
        P_star = (epsn * C_star * R_star * cmax / (R_star + nu0) - mutild * C_star) / (cp * C_star / (C_star + nu1))

        #    print(cp*C_star/(C_star+nu1), epsn * C_star*R_star*cmax/(R_star+nu0))
        if P_star < 0 or C_star < 0:
            R_star = nu0 * mutild / (epsn * cmax + mutild)
            C_star = lam * (cbar - R_star) * (R_star + nu0) / (cmax * R_star)
            P_star = 0

        if C_star < 0:
            R_star = cbar
            P_star = 0
            C_star = 0

        return np.array([R_star, C_star, P_star])


    def optimal_behavior_trajectories(self):
        C = self.population[0]
        N = self.population[1]
        P = self.population[2]
        taun = self.strat[0]
        taup = self.strat[1]

        cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = self.params.values()
        Cdot = lam * (cbar - C) - cmax * N * taun * C / (taun * C + nu0)
        Ndot = N * (epsn * cmax * taun * C / (taun * C + nu0) - taup * taun * P * cp / (
                    taup * taun * N + nu1) - mu0 * taun - mu1)
        Pdot = P * (cp * eps * taup * taun * N / (
                    N * taup * taun + nu1) - phi0 * taup ** 2 - phi1)  # Square cost removed

#        print(C, N, taun, "Optimal behavior trajectories", mu0, self.params['mu0'])
        return np.array([Cdot.squeeze(), Ndot.squeeze(), Pdot.squeeze()])


    def taun_fitness_II_linear(self, s_prey, s_pred):
        R, C, P = self.population[0], self.population[1], self.population[2]

        return self.params['epsn'] * self.params['cmax'] * s_prey * R / (s_prey * R + self.params['nu0']) - \
               self.params['cp'] * s_pred * s_prey * P / (s_pred * s_prey * C + self.params['nu1']) - self.params['mu1'] - self.params['mu0'] * s_prey

    def opt_taun_analytical(self, y, taup, s, eps, gamma, params=None):
        R, C, P = self.population[0], self.population[1], self.population[2]

        eta = (taup * P * s ** (3 / 4) * (eps * R) ** (-1)) ** (1 / 2)

        tauc = gamma * (1 - eta) / (R * eta - C * taup)

        tauc = np.array([tauc])
        if len(tauc.shape) > 1:
            tauc = np.squeeze(tauc)

        tauc[tauc > 1] = 1

        tauc[tauc < 0] = 0.000001
        return tauc

    def nash_eq_find(self):
        y = self.population
        if self.metabolism is False:
            testing_numbers = np.linspace(0.0000005, 1, 100)
            x0 = testing_numbers[(self.opt_taun_analytical(self.opt_taup_find(testing_numbers, self.params), 100, self.params['eps'],
                                                      self.params['nu0']) - testing_numbers) < 0]
            x0 = x0[0]
            # optimal_strategy = optm.fixed_point(lambda strat:  opt_taun_analytical(y, opt_taup_find(y, strat, params)[0], 100, params['eps'], params['nu0']), x0 = x0)

            optimal_strategy = optm.root_scalar(
                    lambda strat: self.opt_taun_analytical(y, self.opt_taup_find(strat)[0], 100, self.params['eps'],
                                                      self.params['nu0']) - strat, bracket=[0.0000005, x0], xtol=10 ** (-7))
            taun = np.array([optimal_strategy.root])
            taup = self.opt_taup_find(taun)
                # print(optimal_strategy.root, taun, taup)
            if (np.isnan(taup) and taun != 0):
                testing_numbers = np.linspace(0.000001, 1, 1000)
                optimal_coordinate = np.argmax(self.params['cp'] * self.params['eps'] * testing_numbers * 1 * y[1] / (
                            y[1] * testing_numbers * 1 + self.params['nu1']) - self.params['phi0'] * testing_numbers ** 2 -
                                               self.params['phi1'])
                taup = testing_numbers[optimal_coordinate]
                taun = np.array([1])
            elif (np.isnan(taup) and taun[0] == 0):
                taup = np.array([0])
            self.strat[0] = taun
            self.strat[1] = taup
        else:
            if y[-1]>10**(-8):
                taun = optm.root(lambda strat: self.taun_linear(strat) - strat, x0 = self.strat[0]).x
                if taun > 1:
                    taun = np.array([1])
                taup = self.opt_taup_find(taun)
                if (np.isnan(taup) and taun != 0):
                    taup = np.array([0])
                self.strat[0] = taun[0]
                self.strat[1] = taup[0]
            else:
                #taun = optm.root(lambda strat: num_derr(lambda s_prey: self.taun_fitness_II_linear(s_prey, 0), strat, 0.000005), x0 = self.strat[0]).x
                #print(self.taun_fitness_II_linear(taun, 0), "Before division")
                taun = np.sqrt(self.params['cmax']*self.params['nu0']*self.params['eps'] / (self.params['mu0'] * y[0])) - self.params['nu0'] / y[0]

                #print(taun, "TAUN")
                #print(self.taun_fitness_II_linear(taun, 0), self.taun_fitness_II_linear(taun / 10, 0), self.taun_fitness_II_linear(1, 0), taun)

                if taun > 1:
                    taun = np.array([1])
                self.strat[0] = taun
                self.strat[1] = np.array([0])

    def opt_taup_find(self, s_prey):
        y = self.population
        k = s_prey * y[1] / self.params['nu1']
        c = self.params['cp'] / self.params['nu1'] * self.params['eps'] * s_prey * y[1] / self.params['phi0']
        x = 1 / 3 * (2 ** (2 / 3) / (
                3 * np.sqrt(3) * np.sqrt(27 * c ** 2 * k ** 8 + 8 * c * k ** 7) + 27 * c * k ** 4 + 4 * k ** 3) ** (
                                 1 / 3)
                     + (3 * np.sqrt(3) * np.sqrt(
                    27 * c ** 2 * k ** 8 + 8 * c * k ** 7) + 27 * c * k ** 4 + 4 * k ** 3) ** (
                             1 / 3) / (2 ** (2 / 3) * k ** 2) - 2 / k)  # Why was WA not included!?!?
        x = np.array([x])
        if max(x.shape) > 1:
            x = np.squeeze(x)
            x[x > 1] = 1
            # print("Alarm!")
        else:
            if x[0] > 1:
                x[0] = 1
        x[x < 0] = 0
        return x

    def strat_setter(self, strategy):
        self.strat = strategy

    def pop_setter(self, population):
        self.population = population

    def resource_setter(self, resource):
        self.params['resource'] = resource


    def gilliam_nash_find(self):
        gill_strat = optm.root(self.gilliam_nash, x0=self.strat).x
        gill_strat[gill_strat < 0] = 0
        gill_strat[gill_strat > 1] = 1

        self.strat = gill_strat

    def optimizer(self, pop, gilliam = False):
        self.population = pop
        if gilliam is False:
            self.nash_eq_find()
        else:
            self.ibr_gill_nash()
            #self.gilliam_nash_find()

        return self.optimal_behavior_trajectories()

    def ibr_gill_nash(self):
        R, C, P = self.population[0], self.population[1], self.population[2]

        prey_gill = lambda prey_s, s_pred :-(self.params['epsn'] * self.params['cmax'] * prey_s * R / (prey_s * R + self.params['nu0'])
                     - self.params['mu0'] * prey_s - self.params['mu1'])/(self.params['cp'] * s_pred * prey_s * P / (s_pred * prey_s * C + self.params['nu1']))
        pred_gill = lambda pred_s, s_prey: -(self.params['cp'] * self.params['eps'] * s_prey * pred_s * C / (C * s_prey * pred_s + self.params['nu1']) - self.params['phi1'])/(self.params['phi0'] * pred_s ** 2)
        error = 1
        its = 0
        s = np.zeros(2)
        while error > 10**(-8):
            s[0] = optm.minimize(lambda x: prey_gill(x, self.strat[1]), x0 = self.strat[0], bounds = [(0.00000001, 1)]).x
            s[1] = optm.minimize(lambda x: pred_gill(x, self.strat[0]), x0 = self.strat[1], bounds = [(0.00000001, 1)]).x
            error = max(np.abs(s - self.strat))
            self.strat = np.copy(s)
            #print(error, its, )
            its += 1
            if its > 100:
                error = 0
                self.strat = np.array([1, 1])
                print("AAAAAAAAAAAAAH")


    def gilliam_nash(self, strat_vec):
        R, C, P = self.population[0], self.population[1], self.population[2]
        s_prey, s_pred = strat_vec[0], strat_vec[1]
        prey_gill = lambda prey_s :-(self.params['epsn'] * self.params['cmax'] * prey_s * R / (prey_s * R + self.params['nu0'])
                     - self.params['mu0'] * prey_s - self.params['mu1'])/(self.params['cp'] * s_pred * prey_s * P / (s_pred * prey_s * C + self.params['nu1']))
        pred_gill =lambda pred_s: -(self.params['cp'] * self.params['eps'] * s_prey * pred_s * C / (C * s_prey*pred_s + self.params['nu1']) - self.params['phi1'])/(self.params['phi0'] * pred_s ** 2)

        der_prey = num_derr(prey_gill, s_prey, 0.00000001)
        der_pred = num_derr(pred_gill, s_pred, 0.00000001)

        return der_prey, der_pred

    def no_coexist_steady_state(self):

        const1 = self.params['cmax']*self.params['eps']*self.params['nu0']
        const2 = (const1/self.params['mu0'])**0.5
        const3 = (const2*(self.params['cmax']*self.params['eps']-self.params['mu1']))
        const4 = const1/const3

        const5 = self.params['nu0']*(self.params['cmax']*self.params['eps']*self.params['mu1'])**(0.5)
        x = const4 + const5/const3

        Rstar = x**2

        tauc = np.sqrt(self.params['cmax']*self.params['nu0']*self.params['eps'] / (self.params['mu0'] * Rstar)) - self.params['nu0'] / Rstar
        Nstar = self.params['lam']*(self.params['resource']-Rstar)*(tauc*Rstar+self.params['nu0'])/(self.params['cmax']*Rstar*tauc)

        self.pop_setter(np.array([Rstar, Nstar, 0]))
        self.strat_setter(np.array([tauc, 0]))
        #print(Rstar, tauc, Nstar, self.optimal_behavior_trajectories())

