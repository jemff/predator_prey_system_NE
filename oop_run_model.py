import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optm
from multiprocessing import Pool
from oop_pred_prey import *

settings = {'simulate' : False, 'resource_bifurcation': False, 'coexistence': False}


if settings['resource_bifurcation'] is True:
    init = np.array([0.9500123,  0.4147732,  0.01282899]) #np.array([9.09090909e-02, 2.08090909e-01, 3.30211723e-15]) #
    mass_vector = np.array([1, 1, 100])
    pred_prey = PredatorPrey(mass_vector, 30, init)
    #pred_prey.strat_setter(np.array([1, 0.78891139]))


    lowest_res = 4/100
    increasing_fidelity = (lambda x: 1/x)(np.linspace(1/30, 100/(3.9), 1)) #(lambda x: 1/x)(np.arange(1/30, int(1/lowest_res), 1/50))

    for i in range(len(increasing_fidelity)):
        bnds = (0, None)
        optm_obj = optm.least_squares(pred_prey.optimizer, x0=init, bounds = ([0, np.inf])) #optm.root(pred_prey.optimizer, x0=init)

    #    print(optm_obj)
        x_pop = optm_obj.x
        if x_pop[-1] < 10**(-8):
            init[-1] = 0
            x_pop = optm.least_squares(pred_prey.optimizer, x0=init, bounds = ([0, np.inf])).x
        pred_prey.pop_setter(x_pop)
        pred_prey.nash_eq_find()
        #print(optm.least_squares(pred_prey.gilliam_nash, x0 = pred_prey.strat), "Gilliam")
        #print(optm.least_squares(pred_prey.gilliam_nash, x0 = pred_prey.strat, bounds=([0,1])))
    #    print(pred_prey.strat, increasing_fidelity[i], pred_prey.population, pred_prey.optimal_behavior_trajectories(), "G")
        init = np.copy(x_pop)
        pred_prey.resource_setter(increasing_fidelity[i])

    #    print(optm.root(pred_prey.optimizer, x0 = init, method = 'hybr'))
    #    print(pred_prey.population, pred_prey.strat, pred_prey.optimal_behavior_trajectories())
if settings['simulate'] is True:
    pred_prey.resource_setter(0.09)
    x_pop = optm.least_squares(pred_prey.optimizer, x0=init, bounds = ([0, np.inf])).x
    pred_prey.nash_eq_find()

    #Non-coexistence ecosystem can be solved analytically, ish. Do so.

    t = 0
    while t<1000:
        pred_prey.nash_eq_find()
    #    print(pred_prey.strat)
        pred_prey.update_pop(time_step = 0.0001)
        t+= 0.0001

    print(pred_prey.population, pred_prey.strat, pred_prey.optimal_behavior_trajectories())

if settings['coexistence'] is False:
    init = np.array([0.9500123, 0.4147732, 0.01282899])  # np.array([9.09090909e-02, 2.08090909e-01, 3.30211723e-15]) #
    mass_vector = np.array([1, 1, 100])
    pred_prey = PredatorPrey(mass_vector, 10, init)
    pred_prey_2 = PredatorPrey(mass_vector, 8, init)
    pred_prey.no_coexist_steady_state()
    pred_prey_2.no_coexist_steady_state()

    print(pred_prey.strat, pred_prey_2.strat, pred_prey_2.population, pred_prey.population)

if settings['coexistence'] is True:
    init = np.array([9.20423327e+00, 6.19141899e-01, 1.86784697e-04])
    mass_vector = np.array([1, 1, 100])
    pred_prey = PredatorPrey(mass_vector, 27, init)
    root_obj =  optm.root(pred_prey.optimizer, x0=init)
    print(root_obj)
    x_pop = root_obj.x
    pred_prey.pop_setter(x_pop)
    pred_prey.nash_eq_find()

    print(pred_prey.strat, pred_prey.population, pred_prey.optimal_behavior_trajectories())