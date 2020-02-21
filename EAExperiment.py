import numpy as np
import pyswarms as ps
from pyswarms.utils.search import RandomSearch
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import numpy
from deap import algorithms
import random
from deap import base
from deap import creator
from deap import tools
from sklearn.model_selection import KFold
import itertools
from GAModule import GAModule

class EAExperiment(object):

    # particles, pso_option_bounds, iters, max_hidden_layers, n_inputs, n_outputs
    def __init__(self, evaluation_function):

        self.evaluation_function = evaluation_function

    def objective_function(self, x):

        fitness = []
        with Pool(12) as p:
            fitness = p.map(self.evaluation_function, np.array_split(x, 12))
            
        fitness = [item for sublist in fitness for item in sublist]
        return fitness
    
    def start_search_ga(self, ngen, pop_size,  window_size, data):
        gam = GAModule()
        
        return gam.start_search(ngen, pop_size, data, self.objective_function, window_size)
        
