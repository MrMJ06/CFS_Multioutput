from deap import algorithms
import random
from deap import base
from deap import creator
from deap import tools
import numpy as np

        
class GAModule(object):
    
    #def __init__(self):
        #self.toolbox = base.Toolbox()
            
    def start_search(self, ngen, pop_size, data, objective_function, window):
        toolbox = base.Toolbox()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        #self.toolbox.register("evaluate", objective_function)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox = base.Toolbox()
        # Attribute generator 
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("attr_int", random.randint, 1, window)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                     (*[toolbox.attr_bool for i in range(len(data.columns))], toolbox.attr_int),n=1)
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
            
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        fitnesses = objective_function(pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit,
            
        CXPB, MUTPB = 0.5, 0.2
        fits = [ind.fitness.values[0] for ind in pop]
        g = 0

        # Begin the evolution
        while g < ngen:
            # A new generation
            g = g + 1
            print("-- Generation %i --" % g)
            # Select the next generation individuals
            offspring = tools.selTournament(pop, len(pop), tournsize=3 )
            #offspring = toolbox.select(pop, len(pop), tournsize=3)
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    tools.cxTwoPoint(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    tools.mutFlipBit(mutant, indpb=0.05)
                    del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = objective_function(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit,
                
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

        return pop, fits
