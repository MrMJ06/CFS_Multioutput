from deap import algorithms
import random
from deap import base
from deap import creator
from deap import tools
import numpy as np

        
class GAModule(object):
    
    def __init__(self):
        self.toolbox = base.Toolbox()
            
    def start_search(self, ngen, pop_size, data, objective_function, window):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox.register("evaluate", objective_function)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox = base.Toolbox()
        # Attribute generator 
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("attr_int", random.randint, 0, window)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                     (*[self.toolbox.attr_bool for i in range(len(data.columns))], toolbox.attr_int),n=1)
        
        self.toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
            
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        
        fitnesses, features = list(self.toolbox.evaluate(pop))
        for ind, fit in zip(pop, fitnesses, features):
            ind.fitness.values = fit
            ind.features = features
            
        CXPB, MUTPB = 0.5, 0.2
        fits = [ind.fitness.values[0] for ind in pop]
        g = 0

        # Begin the evolution
        while max(fits) < 100 and g < 1000:
            # A new generation
            g = g + 1
            print("-- Generation %i --" % g)
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.evaluate(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]
        
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

        return pop
