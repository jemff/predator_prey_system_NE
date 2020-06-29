import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optm
from multiprocessing import Pool
from oop_pred_prey import *

settings = {'simulate' : True, 'resource_bifurcation': False, 'coexistence': True, 'gilliam' : True, 'top_down': True}

if settings['gilliam'] is True:
    init = np.array([0.9,  0.3,  0.3]) #np.array([9.09090909e-02, 2.08090909e-01, 3.30211723e-15]) #
    mass_vector = np.array([1, 1, 100])
    pred_prey = PredatorPrey(mass_vector, 30, init)


    pred_prey.resource_setter(10)
    pred_prey.gilliam_nash_find()
    print(pred_prey.strat)
    #x_pop = optm.least_squares(pred_prey.optimizer, x0=init, bounds = ([0, np.inf])).x
    #pred_prey.gilliam_nash_find()

    #Non-coexistence ecosystem can be solved analytically, ish. Do so.

    t = 1000
    while t<1000:
        #pred_prey.gilliam_nash_find()
        pred_prey.ibr_gill_nash()
        print(pred_prey.strat)
        pred_prey.update_pop(time_step = 0.0001)
        t+= 0.0001
        print(pred_prey.population)

if settings['resource_bifurcation'] is True:
    init = np.array([0.9500123,  0.4147732,  0.01282899]) #np.array([9.09090909e-02, 2.08090909e-01, 3.30211723e-15]) #
    mass_vector = np.array([1, 1, 100])
    pred_prey = PredatorPrey(mass_vector, 25, init)

    lowest_res = 10
    increasing_fidelity = np.linspace(12, 30, 160)#(lambda x: 1/x)(np.linspace(1/30, 100/(3.9), 5)) #(lambda x: 1/x)(np.arange(1/30, int(1/lowest_res), 1/50))
    increasing_fidelity = increasing_fidelity[::-1] #Now it's decreasing, hehehe
    PredPrey_G = PredatorPrey(mass_vector, 30, init)
    optm_obj2 = optm.least_squares(PredPrey_G.optimizer, x0=init, bounds = ([0, np.inf])) #optm.root(pred_prey.optimizer, x0=init)

    #    print(optm_obj)$
    x_pop2 = optm_obj2.x
    pred_prey.pop_setter(x_pop2)
    pred_prey.nash_eq_find()
    PredPrey_G = PredatorPrey(mass_vector, 30, init)
    populations_G = np.zeros((160, 3))
    strategies_G = np.zeros((160, 2))
    populations_Gilliam = np.zeros((160, 3))
    strategies_Gilliam = np.zeros((160, 2))
    for i in range(len(increasing_fidelity)):
        bnds = (0, None)
        pred_prey.resource_setter(increasing_fidelity[i])
        PredPrey_G.resource_setter(increasing_fidelity[i])

        optm_obj = optm.root(lambda x: pred_prey.optimizer(x, gilliam = True), x0=init) #optm.root(pred_prey.optimizer, x0=init)

        #print(optm_obj, i)
        x_pop = optm_obj.x
        pred_prey.pop_setter(x_pop)
        pred_prey.ibr_gill_nash()
        #pred_prey.gilliam_nash_find()
        init = np.copy(x_pop)

        optm_obj2 = optm.root(PredPrey_G.optimizer, x0=init,
                                       )  # optm.root(pred_prey.optimizer, x0=init)

        #print(optm_obj2)
        x_pop2 = optm_obj2.x
        PredPrey_G.pop_setter(x_pop2)
        PredPrey_G.nash_eq_find()
        strategies_G[i] = PredPrey_G.strat
        populations_G[i] = x_pop2
        populations_Gilliam[i] = x_pop
        strategies_Gilliam[i] = pred_prey.strat
        print(x_pop, x_pop2, optm_obj.message, optm_obj2.message)

    for k in range(3):
        plt.figure()
        plt.plot(increasing_fidelity, populations_Gilliam[:, k])
        plt.plot(increasing_fidelity, populations_G[:, k])
        plt.show()

    for k in range(2):
        plt.figure()
        plt.plot(increasing_fidelity, strategies_Gilliam[:, k])
        plt.plot(increasing_fidelity, strategies_G[:, k])
        plt.show()

if settings['top_down'] is True:
    init = np.array([0.9500123,  0.4147732,  0.01282899]) #np.array([9.09090909e-02, 2.08090909e-01, 3.30211723e-15]) #
    mass_vector = np.array([1, 1, 100])
    pred_prey = PredatorPrey(mass_vector, 15, init)

    lowest_res = 10
    increasing_fidelity = np.linspace(0.3, 2, 100)#(lambda x: 1/x)(np.linspace(1/30, 100/(3.9), 5)) #(lambda x: 1/x)(np.arange(1/30, int(1/lowest_res), 1/50))
    increasing_fidelity = increasing_fidelity[::-1] #Now it's decreasing, hehehe
    original_phi0 = np.copy(pred_prey.params['phi0'])

    PredPrey_G = PredatorPrey(mass_vector, 15, init)
    optm_obj2 = optm.least_squares(PredPrey_G.optimizer, x0=init, bounds = ([0, np.inf])) #optm.root(pred_prey.optimizer, x0=init)

    #    print(optm_obj)$
    x_pop2 = optm_obj2.x
    pred_prey.pop_setter(x_pop2)
    pred_prey.nash_eq_find()
    PredPrey_G = PredatorPrey(mass_vector, 15, init)
    populations_G = np.zeros((100, 3))
    strategies_G = np.zeros((100, 2))
    populations_Gilliam = np.zeros((100, 3))
    strategies_Gilliam = np.zeros((100, 2))

    for i in range(len(increasing_fidelity)):
        bnds = (0, None)
        pred_prey.params['phi0'] = increasing_fidelity[i]*original_phi0
        PredPrey_G.params['phi0'] = increasing_fidelity[i]*original_phi0

        optm_obj = optm.root(lambda x: pred_prey.optimizer(x, gilliam = True), x0=init) #optm.root(pred_prey.optimizer, x0=init)

        x_pop = optm_obj.x
        pred_prey.pop_setter(x_pop)
        pred_prey.ibr_gill_nash()
        #pred_prey.gilliam_nash_find()
        init = np.copy(x_pop)

        optm_obj2 = optm.root(PredPrey_G.optimizer, x0=init)  # optm.root(pred_prey.optimizer, x0=init)

        #print(optm_obj2)
        x_pop2 = optm_obj2.x
        PredPrey_G.pop_setter(x_pop2)
        PredPrey_G.nash_eq_find()
        strategies_G[i] = PredPrey_G.strat
        populations_G[i] = x_pop2
        populations_Gilliam[i] = x_pop
        strategies_Gilliam[i] = pred_prey.strat
        print(x_pop, x_pop2, optm_obj.message, optm_obj2.message, i)

    for k in range(3):
        plt.figure()
        plt.plot(increasing_fidelity, populations_Gilliam[:, k])
        plt.plot(increasing_fidelity, populations_G[:, k])
        plt.show()

    for k in range(2):
        plt.figure()
        plt.plot(increasing_fidelity, strategies_Gilliam[:, k])
        plt.plot(increasing_fidelity, strategies_G[:, k])
        plt.show()
