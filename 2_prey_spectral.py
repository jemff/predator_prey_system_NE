import matlab.engine
import numpy as np
import scipy.special as scp
import itertools
eng = matlab.engine.start_matlab()

JacobiGL = lambda x, y, z: eng.JacobiGL(float(x), float(y), float(z), nargout=1)
JacobiGQ = lambda x, y, z: eng.JacobiGQ(float(x), float(y), float(z), nargout=2)


def JacobiP(x, alpha, beta, n):
    P_n = np.zeros((n, x.shape[0]))
    P_n[0] = 1
    P_n[1] = 0.5 * (alpha - beta + (alpha + beta + 2) * x)
    for i in range(1, n - 1):
        an1n = 2 * (i + alpha) * (i + beta) / ((2 * i + alpha + beta + 1) * (2 * i + alpha + beta))
        ann = (alpha ** 2 - beta ** 2) / ((2 * i + alpha + beta + 2) * (2 * i + alpha + beta))
        anp1n = 2 * (i + 1) * (i + alpha + beta + 1) / ((2 * i + alpha + beta + 2) * (2 * i + alpha + beta + 1))

        P_n[i + 1] = ((ann + x) * P_n[i] - an1n * P_n[i - 1]) / anp1n
        # print(np.min(P_n[i]))
    return P_n


def JacobiP_n(x, alpha, beta, n):
    P_n = JacobiP(x, alpha, beta, n)
    if alpha == 1 and beta == 1:
        gamma = lambda alpha, beta, m: 2 ** (3) * (m + 1) / (m + 2) * 1 / ((2 * m + alpha + beta + 1))
    elif alpha == 0 and beta == 0:
        gamma = lambda alpha, beta, m: 2 / ((2 * m + alpha + beta + 1))
    elif alpha == -1 / 2 and beta == - 1 / 2:
        gamma = lambda alpha, beta, m: 2 * scp.math.factorial(m) / ((2 * m + alpha + beta + 1) * scp.gamma(m + 1 / 2))

    for i in range(n):
        d = np.sqrt(gamma(alpha, beta, i))
        P_n[i] = P_n[i] / d

    return P_n


def GradJacobi_n(x, alpha, beta, n):
    P_diff = np.zeros((n, x.shape[0]))
    JacobiPnorma = JacobiP_n(x, alpha + 1, beta + 1, n)
    for i in range(1, n):
        P_diff[i] = JacobiPnorma[i - 1] * np.sqrt(i * (i + alpha + beta + 1))
    return P_diff


def vandermonde_calculator(n):
    x = np.array(list(itertools.chain(JacobiGL(0, 0, n))))
    x = np.reshape(x, x.shape[0])
    return (JacobiP_n(x, 0, 0, n + 1))


def vandermonde_dx(n):
    x = np.array(list(itertools.chain(JacobiGL(0, 0, n))))
    x = np.reshape(x, x.shape[0])
    return (GradJacobi_n(x, 0, 0, n + 1))


matrix_derivative_tt = lambda n: np.matmul(np.transpose(vandermonde_dx(n)),
                                           np.linalg.inv(np.transpose(vandermonde_calculator(n))))

mass_matrix = lambda n: np.dot(np.linalg.inv(vandermonde_calculator(n)),
                               np.linalg.inv(np.transpose(vandermonde_calculator(n))))


