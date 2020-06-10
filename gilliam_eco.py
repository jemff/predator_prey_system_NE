import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optm
from multiprocessing import Pool
from oop_pred_prey import *

settings = {'simulate' : True, 'resource_bifurcation': False, 'coexistence': True, 'gilliam' : True}

if settings['gilliam'] is True:
    init = np.array([10,  0.3,  0.3]) #np.array([9.09090909e-02, 2.08090909e-01, 3.30211723e-15]) #
    mass_vector = np.array([1, 1, 10])
    pred_prey = PredatorPrey(mass_vector, 20, init)


    pred_prey.resource_setter(30)
    pred_prey.ibr_gill_nash()
    print(pred_prey.strat)
    #x_pop = optm.least_squares(pred_prey.optimizer, x0=init, bounds = ([0, np.inf])).x
    #pred_prey.gilliam_nash_find()

    #Non-coexistence ecosystem can be solved analytically, ish. Do so.

    t = 0
    while t<1000:
        pred_prey.ibr_gill_nash()
        #    print(pred_prey.strat)
        pred_prey.update_pop(time_step = 0.0001)
        t+= 0.0001
        print(pred_prey.population)

