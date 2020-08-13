import numpy as np
import scipy.optimize as optm

def opt_taun_linear(y, taup, params, v = 0.1, s=100, eps = 0.14, nu = 0.545454545):
    R, C, P = y[0], y[1], y[2]


    k_1 = 1/v*eps*nu*R
    k_2 = C*taup
    k_3 = nu
    k_4 = 1/v*s**(3/4)*taup*nu
    k_5 = R

    # The problem can be rewritten as
    #-k_2^2 k_5^2 x^4 - 2 k_2 k_3 k_5^2 x^3 - 2 k_2^2 k_3 k_5 x^3 + k_1 k_2^2 x^2 - k_2^2 k_3^2 x^2
    # - k_3^2 k_5^2 x^2 - k_4 k_5^2 x^2 - 4 k_2 k_3^2 k_5 x^2 - 2 k_2 k_3^3 x + 2 k_1 k_2 k_3 x - 2 k_3^3 k_5 x
    # - 2 k_3 k_4 k_5 x - k_3^4 + k_1 k_3^2 - k_3^2 k_4

    c1 = -k_2**2*k_5
    c2 = -( 2*k_2*k_3*(k_5)**2+2*(k_2)**2*k_3*k_5)
    c3 = k_1*k_2**2 - k_2**2*k_3**2 - k_3**2*k_5**2 - k_4*k_5 - 4*k_2*k_3**2*k_5
    c4 =-2*k_2*k_3**3+2*k_1*k_2*k_3 - 2*k_3**3*k_5 - 2*k_3*k_4*k_5
    c5 = -k_3**4 + k_1*k_3**2 - k_3**2*k_4
    print(np.sqrt((2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                        c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3))
    solution_1 = \
    -c2 / (4 * c1) - 1 / 2 * np.sqrt(c2 ** 2 / (4 * c1 ** 2) - (2 * c3) / (3 * c1) + (
                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
            (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                        c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) + (
                                                 2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
            (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                        c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3))) - 1 / 2 * np.sqrt(
        c2 ** 2 / (2 * c1 ** 2) - (4 * c3) / (3 * c1) - (
                    2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                (
                            2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                            c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) - (
                    2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                    2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                (
                            2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                            c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3)) - (
                    -c2 ** 3 / c1 ** 3 + (4 * c3 * c2) / c1 ** 2 - (8 * c4) / c1) / (4 * np.sqrt(
            c2 ** 2 / (4 * c1 ** 2) - (2 * c3) / (3 * c1) + (
                        2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                    (
                                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                                c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) + (
                        2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                        2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                    (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                                c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3)))))

    solution_2 = -c2 / (4 * c1) - 1 / 2 * np.sqrt(c2 ** 2 / (4 * c1 ** 2) - (2 * c3) / (3 * c1) + (
                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
            (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                        c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) + (
                                                 2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
            (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                        c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3))) + 1 / 2 * np.sqrt(
        c2 ** 2 / (2 * c1 ** 2) - (4 * c3) / (3 * c1) - (
                    2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                (
                            2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                            c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) - (
                    2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                    2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                (
                            2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                            c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3)) - (
                    -c2 ** 3 / c1 ** 3 + (4 * c3 * c2) / c1 ** 2 - (8 * c4) / c1) / (4 * np.sqrt(
            c2 ** 2 / (4 * c1 ** 2) - (2 * c3) / (3 * c1) + (
                        2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                    (
                                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                                c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) + (
                        2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                        2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                    (
                                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                                c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3)))))

    solution_3 = -c2 / (4 * c1) + 1 / 2 * np.sqrt(c2 ** 2 / (4 * c1 ** 2) - (2 * c3) / (3 * c1) + (
                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
            (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                        c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) + (
                                                 2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
            (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                        c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3))) - 1 / 2 * np.sqrt(
        c2 ** 2 / (2 * c1 ** 2) - (4 * c3) / (3 * c1) - (
                    2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                            c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) - (
                    2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                    2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                (
                            2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                            c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3)) - (
                    + c2 ** 3 / c1 ** 3 + (4 * c3 * c2) / c1 ** 2 - (8 * c4) / c1) / (4 * np.sqrt(
            c2 ** 2 / (4 * c1 ** 2) - (2 * c3) / (3 * c1) + (
                        2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                    (
                                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                                c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) + (
                        2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                        2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                    (
                                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                                c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3)))))
    solution_4 = -c2 / (4 * c1) + 1 / 2 * np.sqrt(c2 ** 2 / (4 * c1 ** 2) - (2 * c3) / (3 * c1) + (
                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
            (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                        c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) + (
                                                 2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
            (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                        c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3))) + 1 / 2 * np.sqrt(
        c2 ** 2 / (2 * c1 ** 2) - (4 * c3) / (3 * c1) - (
                    2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                (2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                            c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) - (
                    2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                    2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                (
                            2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                            c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3)) - (
                    + c2 ** 3 / c1 ** 3 + (4 * c3 * c2) / c1 ** 2 - (8 * c4) / c1) / (4 * np.sqrt(
            c2 ** 2 / (4 * c1 ** 2) - (2 * c3) / (3 * c1) + (
                        2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                    (
                                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                                c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3) / (3 * 2 ** (1 / 3) * c1) + (
                        2 ** (1 / 3) * (c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5)) / (3 * c1 * (
                        2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5 + np.sqrt(
                    (
                                2 * c3 ** 3 - 9 * c2 * c4 * c3 - 72 * c1 * c5 * c3 + 27 * c1 * c4 ** 2 + 27 * c2 ** 2 * c5) ** 2 - 4 * (
                                c3 ** 2 - 3 * c2 * c4 + 12 * c1 * c5) ** 3)) ** (1 / 3)))))


    print(taun_fitness_II_linear(solution_1,taup, params, R, C, P) , taun_fitness_II_linear(solution_1,taup, params, R, C, P) , taun_fitness_II_linear(solution_1,taup, params, R, C, P) , taun_fitness_II_linear(solution_1,taup, params, R, C, P))

    solution_1[solution_1 < 0] = 0
    solution_1[solution_1 > 1] = 1
    solution_2[solution_2 < 0] = 0
    solution_2[solution_2 > 1] = 1
    solution_3[solution_3 < 0] = 0
    solution_4[solution_4 > 1] =1
    return solution_3


def taun_linear(y, taup, params):
    root_object = optm.root(lambda strat: num_derr(lambda s_prey: taun_fitness_II_linear(s_prey, taup, params, y[0], y[1], y[2]), strat, 0.00001), x0 = 1)
    return max(root_object.x, np.array([1]))

def multi_objective_root(y, taun, taup, params):
    return np.array([taun_linear(y, taup, params)-taun, opt_taup_find(y, taun, params)[0] - taup])

def nash_eq_find(y, params, opt_prey = True, opt_pred = True):

    if opt_pred is True and opt_prey is True:
        testing_numbers = np.linspace(0.0000005, 1, 100)
        x0 = testing_numbers[(opt_taun_analytical(y, opt_taup_find(y, testing_numbers, params), 100, params['eps'], params['nu0'], params = params) - testing_numbers) < 0]
        if len(x0)<1:
            taun, taup = working_nash_eq_find(y, params, opt_prey = True, opt_pred = True)
        else:
            x0 = x0[0]
            #optimal_strategy = optm.fixed_point(lambda strat:  opt_taun_analytical(y, opt_taup_find(y, strat, params)[0], 100, params['eps'], params['nu0']), x0 = x0)


            optimal_strategy = optm.root_scalar(lambda strat:  opt_taun_analytical(y, opt_taup_find(y, strat, params)[0], 100, params['eps'], params['nu0'], params = params)-strat, bracket = [0.0000005, x0], xtol = 10**(-7))
            taun = np.array([optimal_strategy.root])
            taup = opt_taup_find(y, taun, params)
            #print("Here I am", optimal_strategy)
            #print(optimal_strategy.root, taun, taup)
        if (np.isnan(taup) and taun != 0):
            testing_numbers = np.linspace(0.000001, 1, 1000)
            optimal_coordinate = np.argmax(params['cp'] * params['eps'] * testing_numbers * 1 * y[1] / (y[1] * testing_numbers * 1 + params['nu1']) - params['phi0'] * testing_numbers ** 2 - params['phi1'])
            taup = testing_numbers[optimal_coordinate]
            taun = np.array([1])
            #print(np.max(params['cp'] * params['eps'] * testing_numbers * 1 * y[1] / (y[1] * taup * taun + params['nu1']) - params['phi0'] * testing_numbers ** 2 - params['phi1']), "This is it?",

            #      testing_numbers[optimal_coordinate], params['cp'] * params['eps'] * 1 * 1 * y[1] / (y[1] * taup * taun + params['nu1']) - params['phi0'] * 1 ** 2 - params['phi1'], optimal_coordinate)
        elif (np.isnan(taup) and taun[0] == 0):
            taup = np.array([0])
            print(taup, taun, y)
    else:
        taun = np.array([1])
        taup = np.array([1])
    return taun, taup



def working_nash_eq_find(y, params, taun_previous = np.array([1]), opt_prey = True, opt_pred = True):
    if opt_pred and opt_prey is True:
        #testing_numbers = np.linspace(0.01, 1, 100)
        #valid_responses = opt_taun_analytical(y, testing_numbers, 100, params['eps'], params['nu0'])
        #if np.min(valid_responses)>1: #fix the magic numbers
        #    taun = np.array([1])
        #    taup = opt_taup_find(y, taun, params)[0]
        #elif np.max(valid_responses)<0:
        #    taun = 0
        #    taup = 0

        #else:
        root_obj = optm.root(lambda strat: opt_taun_analytical(y, opt_taup_find(y, strat, params)[0], 100, params['eps'], params['nu0'], params = params, taun_previous = taun_previous)-strat, x0 = taun_previous)
#        print(root_obj.x, least_sq_obj.x)
        taun = root_obj.x
        if root_obj.success is False:
            least_sq_obj = optm.root(
                lambda strat: opt_taun_analytical(y, opt_taup_find(y, strat, params)[0], 100, params['eps'],
                                                  params['nu0'], params=params, taun_previous=np.array([1])) - strat,
                x0=np.array([1]))

            taun = least_sq_obj.x
#            print(taun_previous, root_obj.x)
        taup = opt_taup_find(y, taun, params)[0]
#        print(taun, taup, root_obj.message, "Outer root")

        if taun>1:
            taun = np.array([1])
            taup = opt_taup_find(y, taun, params)[0]

    else: #Should add the other two cases.
        taun = 1
        taup = 1
#    print(opt_taup_find(y, 0.5, params))

    return taun, taup

def opt_taun_find_Gill(y, params, taun_old):
    C, N, P = y[0], y[1], y[2]
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    taun_fitness_II = lambda s_prey: epsn * cmax * s_prey * C / (s_prey * C + nu0) - cp * taup(s_prey) * s_prey * P / (
                    taup(s_prey) * s_prey * N + nu1) - mu0 * s_prey**2 - mu1
    p_term = lambda s_prey : (N*s_prey*taup(s_prey)+nu1)

    taup = lambda s_prey: opt_taup_find(y, s_prey, params) #cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey)) -cp * (np.sqrt(eps / (phi0 * s_prey * N)) - 1 / (N * s_prey))

    taup_prime = lambda s_prey: (gill_opt_taup(y, s_prey+0.00001, params)-gill_opt_taup(y, s_prey-0.00001, params))/(2*0.00001) #cp*(1/(N*s_prey**2) - 1/2*np.sqrt(eps/(phi0*N))*s_prey**(-3/2))

    taun_fitness_II_d = lambda s_prey: epsn*(cmax*nu0)*C/((s_prey*C+nu0)**2) - 2*s_prey*mu0 \
                                            - (nu1*cp)*P*((s_prey*taup_prime(s_prey)+taup(s_prey))/(p_term(s_prey))**2)

    #Eeeeh. Not worth it.
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

    return max_cands


def nash_eq_find_old(y, params, opt_prey = True, opt_pred = True):
    if opt_pred and opt_prey is True:
        testing_numbers = np.linspace(0.000001, 1, 100)
        valid_responses = opt_taun_analytical(y, testing_numbers, 100, params['eps'], params['nu0'])
        if np.min(valid_responses)>1: #fix the magic numbers
            #print(valid_responses-testing_numbers)
            taun = np.array([1])
            taup = opt_taup_find(y, taun, params)[0]
            #print("wut", taun, taup, np.min(valid_responses), np.max(valid_responses), y)
            if np.isnan(taup[0]):
                taup = np.array([1]) #np.array([1]) #opt_taup_find(y, taun, params)[0]
        elif np.max(valid_responses)<0:
            taun = np.array([0])
            taup = np.array([0])
        else:
            print(valid_responses)
            taun = optm.root_scalar(lambda strat: opt_taun_analytical(y, opt_taup_find(y, strat, params)[0], 100, params['eps'], params['nu0'])-strat, method = 'brentq', bracket = [0.000001, 1])
            #print(taun)
            taun = taun.root

            taun = np.array([taun])
            taup = opt_taup_find(y, taun, params)[0]
            if np.isnan(taup[0]):
                taup = np.array([0]) #np.array([1]) #opt_taup_find(y, taun, params)[0]

            #print("wut2", taun, taup)
        if taun>1:
            taun = np.array([1])
            taup = opt_taup_find(y, taun, params)[0]

    else: #Should add the other two cases.
        taun = 1
        taup = 1

    return taun, taup

def opt_taup_find_quadratic(y, s_prey, params):
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
        x[np.isnan(x)] = 0.78 #eeeh
        #print("Alarm!")
    else:
        if x[0] > 1:
            x[0] = 1
        x[np.isnan(x)] = 0.78
    x[x<0] = 0
    return x

def opt_taup_find(y, s_prey, params, linear = False):
    if linear is False:
        x = opt_taup_find_quadratic(y, s_prey, params)
    elif linear is True:
        x = opt_taup_find_linear(y, s_prey, params)
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

    return max_cands

import semi_impl_eul as num_sol
def static_eq_calc(params):
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()

    phitild = phi0 + phi1
    mutild = mu0 + mu1

    C_star = phitild * nu1 / (eps * cp - phitild)

    gam = nu0 - cbar + (cmax / lam) * C_star

    R_star = (-gam + np.sqrt(gam ** 2 + 4 * cbar * nu0)) / 2

    P_star = (epsn * C_star * R_star * cmax / (R_star + nu0) - mutild * C_star) / (cp * C_star / (C_star + nu1))

    #    print(cp*C_star/(C_star+nu1), epsn * C_star*R_star*cmax/(R_star+nu0))
    if P_star < 0 or C_star < 0:
        R_star = mutild*nu0/(eps*cmax)*((1-mutild/(eps*cmax))**(-1)) #nu0 * mutild / (epsn * cmax + mutild)
        C_star = lam * (cbar - R_star) * (R_star + nu0) / (cmax * R_star)
        P_star = 0

        print(num_sol.optimal_behavior_trajectories(0, np.array([R_star, C_star, P_star]), params)[0:3],
              np.array([R_star, C_star, P_star]))

    if C_star < 0:
        R_star = cbar
        P_star = 0
        C_star = 0
    return np.array([R_star, C_star, P_star])



def parameter_calculator_mass(mass_vector, alpha = 15, b = 330/12, v = 0.05):
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


def taun_fitness_II_linear(s_prey, s_pred, params, R, C, P):
    return params['epsn'] * params['cmax'] * s_prey * R / (s_prey * R + params['nu0']) - params['cp'] * s_pred * s_prey * P / (
                s_pred * s_prey * C + params['nu1']) - params['mu1'] - params['mu0']*s_prey #This does have linear cost???

def taun_fitness_II(s_prey, params, R, C, P):
    y = np.array([R, C, P])
    return params['epsn'] * params['cmax'] * s_prey * R / (s_prey * R + params['nu0']) - params['cp'] * opt_taup_find(y, s_prey, params) * s_prey * P / (
                opt_taup_find(y, s_prey, params) * s_prey * C + params['nu1']) - params['mu1']



def strat_finder_Gill(y, params, opt_prey = True, opt_pred = True, taun_old = 1):

    taun = min(max(opt_taun_find(y, params, taun_old), 0), 1)
    taup = min(max(opt_taup_find(y, taun, params), 0), 1)

    return taun, taup


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



def opt_taup_find_linear(y, taun, params): #THIS IS THE LIENAR VERSION !!!!!!!!!!!!!!!!!!!!!!!
    N = y[1]

    taun = np.array([taun])
    if taun.shape == (1,1):
        taun = taun[0]
    elif np.sum(taun.shape) > 2:
        taun = np.squeeze(taun)
    cmax, mu0, mu1, eps, epsn, cp, phi0, phi1, cbar, lam, nu0, nu1 = params.values()
    res = (np.sqrt(cp*nu0*eps / (phi0 * taun * N)) - nu0 / (N * taun))

    res[res>1] = 1
    res[res<0] = 0

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

def opt_taun_analytical_old(y, taup, s, eps, gamma, params = None):
    R, C, P = y[0], y[1], y[2]

    eta = (taup*P*s**(3/4)*(eps*R)**(-1))**(1/2)

    tauc = gamma*(1-eta)/(R*eta-C*taup)

    tauc = np.array([tauc])
    if len(tauc.shape)>1:
        tauc = np.squeeze(tauc)


    tauc[tauc>1] = 1
#    if len(tauc[tauc<0]) != 0:
#        tauc = taun_linear(y, taup, params)

    tauc[tauc<0] = 0.000001

    return tauc

def opt_taun_analytical(y, taup, s, eps, gamma, params = None, taun_previous = np.array([0.5])):

    a=eps*y[0]*15
    b=y[0]
    c=gamma
    d=s**(3/4)*15*taup*y[2]
    e=y[1]*taup

    tauc = ((b**2*d-a*e**2)*np.sqrt(a*c**2*(d*(b-e)**2)/(b**2*d-a*e**2)**2)+a*c*e-b*c*d)/(b**2*d-a*e**2)
#    tauc_alt = opt_taun_analytical_old(y, taup, s, eps, gamma)



    tauc = np.array([tauc])
    if len(tauc.shape)>1:
        tauc = np.squeeze(tauc)


    tauc[tauc>1] = 1
#    if len(tauc[tauc<0]) != 0:
#        tauc = taun_linear(y, taup, params)

    tauc[tauc<0] = 0.000001
    tauc_alt = np.copy(tauc)
#    this = prey_GM(tauc, taup, params, y)
#    that = prey_GM(1, taup, params, y)
#    if that < this:
#        tauc = np.array([1])
    if max(np.array([taup]).shape)<2:
        tauc = optm.minimize(lambda x: prey_GM(x, taup, params, y), x0 = taun_previous, bounds = [(0.00000001, 1)]).x #Used to be 0.5
        #print(prey_GM(tauc, taup, params, y), prey_GM(tauc_alt, taup, params, y))
    return tauc


def prey_gill(s_prey, s_pred, params, y):
    R, C, P = y[0], y[1], y[2]

    return -(params['cmax'] * s_prey * R / (s_prey * R + params['nu0']) - params['mu0'] * s_prey - params['mu1']) / (
                params['cp'] * s_pred * s_prey * P / (s_pred * s_prey * C + params['nu1']))

def pred_gill(s_prey, s_pred, params, y):
    R, C, P = y[0], y[1], y[2]

    return -(params['cp'] * params['eps'] * s_prey * s_pred * C / (C * s_prey * s_pred + params['nu1']) - params['phi1']) / (
                params['phi0'] * s_pred ** 2)

def prey_GM(s_prey, s_pred, params, y):
    R, C, P = y[0], y[1], y[2]

    return -((params['cmax'] * s_prey * R / (s_prey * R + params['nu0']) - params['mu0'] * s_prey - params['mu1']) - (
            params['cp'] * s_pred * s_prey * P / (s_pred * s_prey * C + params['nu1'])))

def pred_GM(s_prey, s_pred, params, y, linear = False):
    R, C, P = y[0], y[1], y[2]
    if linear is False:
        return -((params['cp'] * params['eps'] * s_prey * s_pred * C / (C * s_prey * s_pred + params['nu1']) - params['phi1']) - (
                params['phi0'] * s_pred ** 2))
    else:
        return -((params['cp'] * params['eps'] * s_prey * s_pred * C / (C * s_prey * s_pred + params['nu1']) - params['phi1']) - (
                params['phi0'] * s_pred))

def chooser_payoff(choice = 'Gill'):
    if choice is 'Gill':
        return [prey_gill, pred_gill]
    else:
        return [prey_GM, pred_GM]

def gilliam_nash(y, params, strat = np.array([0.5, 0.5])):
    s_prey, s_pred = strat[0], strat[1]

    der_prey = num_derr(prey_gill(s_prey, s_pred, params, y), s_prey, 0.00000001)
    der_pred = num_derr(prey_gill(s_prey, s_pred, params, y), s_pred, 0.00000001)

    return der_prey, der_pred

def gilliam_nash_find(y, params, strat = np.array([0.5, 0.5])):
    gill_strat = optm.root(lambda x: gilliam_nash(y, params, strat = x), x0=strat).x
    gill_strat[gill_strat < 0] = 0
    gill_strat[gill_strat > 1] = 1

    return gill_strat

def combined_strat_finder(params, y, stackelberg = False, x0=None, Gill = False):
    error = 1
    its = 0
    s = np.zeros(2)
    strat = np.copy(x0)

    if stackelberg is True and Gill is True:
        s[0] = optm.minimize(lambda x: prey_gill(x, gill_opt_taup(y, x, params), params, y), x0 = strat[0], bounds = [(0.00000001, 1)]).x
        s[1] = gill_opt_taup(y, s[0], params)

        strat = np.copy(s)

    elif stackelberg is True and Gill is False:
        tauc, taup = strat_finder(y, params)
        strat[0] = tauc
        strat[1] = taup

    elif stackelberg is False and Gill is False:
        s = np.zeros(2)
        while error > 10 ** (-8):
            s[0] = optm.minimize(lambda x: prey_GM(x, strat[1], params, y), x0 = strat[0], bounds = [(0.00000001, 1)]).x
            s[1] = optm.minimize(lambda x: pred_GM(strat[0], x, params, y), x0 = strat[1], bounds = [(0.00000001, 1)]).x
            error = max(np.abs(s - strat))
            strat = np.copy(s)

            its += 1
            if its > 100:
                error = 0
                tauc, taup = working_nash_eq_find(y, params, taun_previous = x0[0])
                strat[0] = tauc
                strat[1] = taup
        #if its<100:
        #    print(working_nash_eq_find(y, params), strat, y)

    elif stackelberg is False and Gill is True:
        s = np.zeros(2)
        while error > 10 ** (-8):
            s[0] = optm.minimize(lambda x: prey_gill(x, strat[1], params, y), x0 = strat[0], bounds = [(0.00000001, 1)]).x
            s[1] = optm.minimize(lambda x: pred_gill(strat[0], x, params, y), x0 = strat[1], bounds = [(0.00000001, 1)]).x
            error = max(np.abs(s - strat))
            strat = np.copy(s)
            its += 1
            if its > 300:
                error = 0
                #print(s, x0)
                strat[0], strat[1] = s #nash_eq_find_Gill(y, params, strat[0])


    return strat[0], strat[1]

def gill_opt_taup(y, tauc, params):
    k_1 = params['eps']*y[1]*tauc*params['cp']/params['phi0']
    k_2 = y[1]*tauc
    k_3 = params['nu1']
    k_4 = params['phi1']/params['phi0']

    x = (k_4 * np.sqrt((k_1 * k_3 ** 2 * (k_1 + 8 * k_2 * k_4)) / (
                k_2 ** 2 * (k_2 * k_4 - k_1) ** 2)) * k_2 ** 2 - 4 * k_3 * k_4 * k_2 - k_1 * np.sqrt(
        (k_1 * k_3 ** 2 * (k_1 + 8 * k_2 * k_4)) / (k_2 ** 2 * (k_2 * k_4 - k_1) ** 2)) * k_2 + k_1 * k_3) / (
                    4 * k_2 * (k_2 * k_4 - k_1))

    x = np.array([x])
    if max(x.shape) > 1:
        x = np.squeeze(x)
        x[x > 1] = 1
        x[np.isnan(x)] = 0.78 #eeeh
        #print("Alarm!")
    else:
        if x[0] > 1:
            x[0] = 1
        x[np.isnan(x)] = 0.78
    x[x<0] = 0

    return x

def opt_taun_analytical_Gill(y, s_pred, params):

    k_1 = y[0] * params['cmax'] * params['eps']
    k_2 = y[0]
    k_3 = params['nu0']
    k_4 = params['mu1']
    k_5 = s_pred * y[2] * params['cp']
    k_6 = s_pred * y[1]


    x = (k_4 * np.sqrt((k_1 * k_3 ** 2 * k_4 * (k_2 - k_6)) / (
                k_4 * k_2 ** 2 - k_1 * k_2 + k_1 * k_6) ** 2) * k_2 ** 2 - k_3 * k_4 * k_2 - k_1 * np.sqrt(
        (k_1 * k_3 ** 2 * k_4 * (k_2 - k_6)) / (
                    k_4 * k_2 ** 2 - k_1 * k_2 + k_1 * k_6) ** 2) * k_2 + k_1 * k_6 * np.sqrt(
        (k_1 * k_3 ** 2 * k_4 * (k_2 - k_6)) / (k_4 * k_2 ** 2 - k_1 * k_2 + k_1 * k_6) ** 2)) / (
                    k_4 * k_2 ** 2 - k_1 * k_2 + k_1 * k_6)

    x = np.array([x])
    if max(x.shape) > 1:
        x = np.squeeze(x)
        x[x > 1] = 1
        x[np.isnan(x)] = 0.78 #eeeh
        #print("Alarm!")
    else:
        if x[0] > 1:
            x[0] = 1
        x[np.isnan(x)] = 0.78
    x[x<0] = 0

    return x


def nash_eq_find_Gill(y, params, s_prey0):
#    print(opt_taun_analytical_Gill(y, gill_opt_taup(y, 0.5, params)[0], params), s_prey0)
    taun = optm.root(lambda x: opt_taun_analytical_Gill(y, gill_opt_taup(y, x, params)[0], params)[0]-x, x0 = 1).x
    taup = gill_opt_taup(y, taun, params)

    return taun, taup
