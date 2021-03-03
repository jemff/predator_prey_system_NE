import numpy as np
import scipy as scp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from io import StringIO
from scipy import optimize as optm
import scipy.integrate
from multiprocessing import Pool
from mpl_toolkits.axes_grid1 import make_axes_locatable

class PredatorPrey:
    def __init__(self, params, end_time = 50, timestep = 0.005, opt_prey = True, opt_pred = True):
        self.params = params
        self.timestep = timestep
        self.opt_prey = opt_prey
        self.opt_pred = opt_pred
        self.end_time = end_time



    def opt_taup(self, y, taun):
        #based on finding roots of (c - 2 x (k x + 1)^2)
        #which comes from optimizing c*x/(k*x+1)-x^2
        #Thus c = eps*taun*N/phi0, k = taun*N/cp

        k = taun * y[1] / self.params['cp']
        c = self.params['eps'] * taun * y[1] / self.params['phi0']
        x = 1 / 3*(2 ** (2 / 3) / (3* np.sqrt(3)*np.sqrt(27*c ^ 2* k ^ 8 + 8*c*k ^ 7) + 27*c*k ^ 4 + 4*k ^ 3) ** (1 / 3)
                  + (3 *np.sqrt(3)* np.sqrt(27*c**2*k ^ 8 + 8*c*k ^ 7) + 27*c*k ^ 4 + 4*k ^ 3) ** (1 / 3) / (2 ** (2 / 3) *k ** 2) - 2 / k)