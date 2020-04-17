import numpy as np
import scipy.optimize as optm

def nash_eq_find(y, params, opt_prey = True, opt_pred = True):
    if opt_pred and opt_prey is True:
        testing_numbers = np.linspace(0.01, 1, 100)
        valid_responses = opt_taun_analytical(y, testing_numbers, 100, 0.7, 0.545454)
        if np.min(valid_responses)>1: #fix the magic numbers
            taun = 1
            taup = 1 #opt_taup_find(y, taun, params)[0]
        elif np.max(valid_responses)<0:
            taun = 0
            taup = 0
        else:
            taun = optm.root(lambda strat: opt_taun_analytical(y, opt_taup_find(y, strat, params)[0], 100, 0.7, 0.54545454)-strat, x0 = np.array([0.5])).x
            taup = opt_taup_find(y, taun, params)[0]

        if taun>1:
            taun = 1
            taup = opt_taup_find(y, taun, params)[0]

    else: #Should add the other two cases.
        taun = 1
        taup = 1

    return taun, taup

def opt_taup_find(y, s_prey, params):
    #print(params)
    k = s_prey * y[1] / params['nu1']
    c = params['cp']/params['nu1']*params['eps'] * s_prey * y[1] / params['phi0']
    x = 1 / 3 * (2 ** (2 / 3) / (
                3 * np.sqrt(3) * np.sqrt(27 * c ** 2 * k ** 8 + 8 * c * k ** 7) + 27 * c * k ** 4 + 4 * k ** 3) ** (1 / 3)
                 + (3 * np.sqrt(3) * np.sqrt(27 * c ** 2 * k ** 8 + 8 * c * k ** 7) + 27 * c * k ** 4 + 4 * k ** 3) ** (
                             1 / 3) / (2 ** (2 / 3) * k ** 2) - 2 / k) #Why was WA not included!?!?
    x = np.array([x])
    if max(x.shape) > 1:
        x = np.squeeze(x)
        x[x > 1] = 1
        #print("Alarm!")
    else:
        if x[0] > 1:
            x[0] = 1
            #print("Alarm, single!", s_prey)
    return x


def opt_taun_find(y, params, taun_old):
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    taun_fitness_II = lambda s_prey: epsn * cmax * s_prey * C / (s_prey * C + nu0) - cp * taup(s_prey) * s_prey * P / (
                    taup(s_prey) * s_prey * N + nu1) - mu0 * s_prey**2 - mu1
    p_term = lambda s_prey : (N*s_prey*taup(s_prey)+nu1)

    taup = lambda s_prey: opt_taup_find(y, s_prey, params) #cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey)) -cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey))

    taup_prime = lambda s_prey: (opt_taup_find(y, s_prey+0.00001, params)-opt_taup_find(y, s_prey-0.00001, params))/(2*0.00001) #cp*(1/(N*s_prey**2) - 1/2*np.sqrt(eps/(phi0*N))*s_prey**(-3/2))

    taun_fitness_II_d = lambda s_prey: epsn*(cmax*nu0)*C/((s_prey*C+nu0)**2) - 2*s_prey*mu0 \
                                            - (nu1*cp)*P*((s_prey*taup_prime(s_prey)+taup(s_prey))/(p_term(s_prey))**2)

    linsp = np.linspace(0.001 , 1, 100)
    comparison_numbs = taun_fitness_II_d(linsp)
    alt_max_cand = linsp[np.argmax(taun_fitness_II(linsp))]

    #print(comparison_numbs)
    if len(np.where(comparison_numbs > 0)[0]) is 0 or len(np.where(comparison_numbs < 0)[0]) is 0:
        t0 = taun_fitness_II(0.001)
        t1 = taun_fitness_II(1)
        if t0 > t1:
            max_cands = 0.001
        else:
            max_cands = 1
      #  print("dong dong")

    else:
        maxi_mill = linsp[np.where(comparison_numbs > 0)[0][0]]
        mini_mill = linsp[np.where(comparison_numbs < 0)[0][0]]
        max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[mini_mill, maxi_mill], method='brentq').root
        #print("ding ding", taun_fitness_II(max_cands), np.max(taun_fitness_II(linsp)))
   # print(num_derr(taun_fitness_II, max_cands, 0.0001), taun_fitness_II_d(max_cands), max_cands)
   # print(np.max(taun_fitness_II(linsp)), taun_fitness_II(max_cands))
    if taun_fitness_II(max_cands)<=taun_fitness_II(alt_max_cand):
        max_cands = alt_max_cand
        #print("Ding dong sling slong")
    #if taun_fitness_II(max_cands)<0:
        #print(taun_fitness_II(0.001), taun_fitness_II(1), taun_fitness_II(alt_max_cand), taun_fitness_II(max_cands))
    #print(taun_fitness_II(taun_old), taun_fitness_II(max_cands))
    return max_cands #max_cands_two[np.argmax(taun_fitness_II(max_cands_two))]

def static_eq_calc(params):
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    phitild = phi0+phi1
    mutild = mu0 + mu1
    C_star = phitild*nu1/(eps*cp-phitild)
    gam = nu0-cbar+(cmax/lam)*C_star
    print(gam, gam**2, 4*cbar*nu0, np.sqrt(gam**2+4*cbar*nu0))
    R_star = (-gam + np.sqrt(gam**2+4*cbar*nu0))/2
    P_star = (epsn * C_star*R_star*cmax/(R_star+nu0)-mutild*C_star)/(cp*C_star/(C_star+nu1))

    print(cp*C_star/(C_star+nu1), epsn * C_star*R_star*cmax/(R_star+nu0))
    if P_star<0 or C_star<0:
        R_star = nu0*mutild/(epsn*cmax+mutild)
        C_star = lam*(cbar-R_star)*(R_star+nu0)/(cmax*R_star)
        P_star = 0
    if C_star<0:
        R_star = cbar
        P_star = 0
        C_star = 0
    return np.array([R_star, C_star, P_star])



def parameter_calculator_mass(mass_vector, alpha = 15, b = 330/12, v = 0.1):
    #alpha = 15
    #b = 330/12
    #v = 0.1 #/12
    maximum_consumption_rate = alpha * mass_vector[1:]**(0.75)

    ci = v*maximum_consumption_rate
    ci[0] = ci[0]
    #ci[-1] = ci[-1]*0.1
    #print(maximum_consumption_rate)
    r0  = 0.1
    nu = alpha/b*mass_vector[1:]**(0)
    #print(ci)
    return ci, nu, maximum_consumption_rate, r0


def taun_fitness_II(s_prey, params, R, C, P):
    y = np.array([R, C, P])
    return params['epsn'] * params['cmax'] * s_prey * R / (s_prey * R + params['nu0']) - params['cp'] * opt_taup_find(y, s_prey, params) * s_prey * P / (
                opt_taup_find(y, s_prey, params) * s_prey * C + params['nu1']) - params['mu1']


def strat_finder(y, params, opt_prey = True, opt_pred = True, taun_old = 1):
    C, N, P = y[0], y[1], y[2]
    taun = 1
    taup = 1
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    if opt_prey is True and opt_pred is True:
        taun = min(max(opt_taun_find(y, params, taun_old), 0), 1)
        taup = min(max(opt_taup_find(y, taun, params), 0), 1)

    elif opt_prey is True and opt_pred is False:

        taun_fitness_II = lambda s_prey: epsn * cmax * s_prey * C / (s_prey * C + cmax) - cp * 1 * s_prey * P / (1 * s_prey * N + nu1) - mu0 * s_prey ** 2 - mu1

        p_term = lambda s_prey: (N * s_prey + nu1)

        taun_fitness_II_d = lambda s_prey: epsn * (cmax * nu0) * C / ((s_prey * C + nu0) ** 2) - 2*s_prey*mu0 \
                                           - (cp * nu1) * P * ((s_prey + 1)) / (p_term(s_prey)) ** 2

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
            #print(comparison_numbs[np.where(comparison_numbs < 0)[0]])
            #print(taun_fitness_II_d(comparison_numbs[np.where(comparison_numbs < 0)[0]]))
            maxi_mill = linsp[np.where(comparison_numbs > 0)[0][0]]
            mini_mill = linsp[np.where(comparison_numbs < 0)[0][0]]
            #print(taun_fitness_II_d(0.0001), taun_fitness_II_d(maxi_mill))
            max_cands = optm.root_scalar(taun_fitness_II_d, bracket=[mini_mill, maxi_mill], method='brentq').root

        taun = min(max(max_cands, 0.0001), 1)
    elif opt_pred is True and opt_prey is False:
        taup = min(max(opt_taup_find(y, 1, params), 0),1)

    return taun, taup



def opt_taup_find_linear(y, taun, params):
    N = y[1]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam = params.values()
    if taun is 0:
        return 0
    else:
        res1 = cp * (np.sqrt(eps / (phi0 * taun * N)) - 1 / (N * taun))
        res2 = -res1

        res1 = max(min(res1, 1),0)
        res2 = max(min(res2, 1),0)

        r1 = cp * eps * res1 * taun * N / (N * res1 * taun + cp) - phi0 * res1 - phi1
        r2 = cp * eps * res2 * taun * N / (N * res2 * taun + cp) - phi0 * res2 - phi1

#        print(r1, "r1", r2, "r2", res1, res2,  cp * eps * 1 * taun * N / (N * 1 * taun + cp) - phi0 * 1 - phi1, -phi1)
#        res = res1
#        res[r2>r1] = res[r2 > r1]
        if r1 > r2:
            res = res1
        else:
            res = res2
#            print("Ding ding")
#        res = res2
#        print(res, taun)
        return res

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
    for i in range(len((x))) :
        x_m[i] -= h
        x_p[i] += h
        jac[:, i] = (f(x_p) - f(x_m))/(2*h)
        x_m = np.copy(x)
        x_p = np.copy(x)

    return jac

def opt_taun_analytical(y, taup, s, eps, gamma):
    R, C, P = y[0], y[1], y[2]

    eta = (taup*P*s**(3/4)*(eps*R)**(-1))**(1/2)

    tauc = gamma*(1-eta)/(R*eta-C*taup)

    return tauc