import numpy as np
import datetime
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

import errorfunctions as errf

logger = logging.getLogger(__name__)

def ask_error_function():
    error_function = input('Enter ErrorFunction (sphere/rosenbrock/ackley/rastrigin/schwefel): ')
    # Sphere x* = (0,0,...,0)
    if error_function == 'sphere':
        lower_bound = -5.12 # Domain of the target function
        upper_bound = 5.12 # Domain of the target function
        return errf.sphere, lower_bound, upper_bound, error_function
    # Rosenbrock x* = (1,1,...,1)
    elif error_function == 'rosenbrock':
        lower_bound = -2.048 # Domain of the target function
        upper_bound = 2.048 # Domain of the target function
        return errf.rosenbrock, lower_bound, upper_bound, error_function
    # Ackley x* = (0,0,...,0)
    elif error_function == 'ackley':
        lower_bound = -32.768 # Domain of the target function
        upper_bound = 32.768 # Domain of the target function
        return errf.ackley, lower_bound, upper_bound, error_function
    # Rastrigin x* = (0,0,...,0)
    elif error_function == 'rastrigin':
        lower_bound = -5.12 # Domain of the target function
        upper_bound = 5.12 # Domain of the target function
        return errf.rastrigin, lower_bound, upper_bound, error_function
    # Schwefel x* = (420.9687,420.9687,...,420.9687)
    elif error_function == 'schwefel':
        lower_bound = -500. # Domain of the target function
        upper_bound = 500. # Domain of the target function
        return errf.schwefel, lower_bound, upper_bound, error_function

def create_individual(lower_bound, upper_bound):
    '''
    ABLE TO CREATE A NEW INDIVIDUAL.

    lower_bound float. Domain's lower bound.
    upper_bound float. Domain's upper bound.

    RETURNS.
    individual numpyarray. A individual with continous random M gens
    '''
    return np.random.uniform(lower_bound, upper_bound, M_gens)

def create_population():
    '''
    ABLE TO CREATE A POPULATION.

    RETURNS.
    population list. A list with N individuals
    '''
    return [create_individual(lower_bound, upper_bound) for _ in range(N_individuals)]

def evaluate(population, errf=errf):
    '''
    ABLE TO EVALUATE EVERY INDIVIDUAL IN THE ERROR FUNCTION.

    population list. A list with N individuals
    errf function. The target function to minimize

    NOTICE THAT errf.trans_sphere NEEDS A ORIGIN

    RETURNS.
    _evaluated list. A sorted (ASC) list of ordered pairs like (float, numpy.array)
    '''
    evaluated = [( errf(indiv), indiv ) for indiv in population]
    return sorted(evaluated, key=lambda tup:tup[0])

def termination_criteria(errors, epsilon, iteration, itermax):
    '''
    THE ALGORITHM TERMINATES IF THE CRITERIA IS TRUE.

    errors list. A sorted (ASC) list.
    epsilon float. Termination criteria
    iteration int. Current iteration
    itermax int. Termination criteria

    RETURNS.
    boolean
    '''
    if abs(np.max(errors) - np.min(errors)) <= epsilon:
        logger.info('\nTerminationCriteria: delta<=epsilon\n')
        return True
    elif iteration >= itermax:
        logger.info('\nTerminationCriteria: iteration>=itermax\n')
        return True


def crossover(evaluated_population, cutoff_point):
    '''
    ABLE TO EXCHANGE BESTS INDIVIDUALS' GENETIC MATERIAL IN ORDER TO CREATE CHILDS

    evaluated_population list. A sorted list of numpy_arrays
    cutoff_point int. The cutoff point of the genetic material

    RETURNS.
    childs list. A list of numpy arrays with the new individuals.
    '''
    half = N_individuals//2
    parents = evaluated_population[:half]
    childs = [np.zeros(M_gens) for _ in range(half)]
    for i in range(0, len(parents), 2):
        if i+1 <= len(parents):
            childs[i][:cutoff_point] = parents[i][:cutoff_point]
            childs[i][cutoff_point:] = parents[i+1][cutoff_point:]
            childs[i+1][:cutoff_point] = parents[i+1][:cutoff_point]
            childs[i+1][cutoff_point:] = parents[i][cutoff_point:]

    return childs

def mutation(childs, p_1=5., p_2=10.):
    '''
    ABLE TO MODIFY 10% OF THE GENS OF THE 5% OF THE POPULATION.

    childs list. A list of numpy arrays
    p_1 float. Chance of affecting p_1 of the population
    p_2 float. Chance of affecting p_2 of the gens of an individual.

    RETURNS.
    childs list. Modified childs
    '''
    for child in childs:
        if np.random.uniform(0,100) <= p_1:
            for gen in range(M_gens):
                if np.random.uniform(0,100) <= p_2:
                    child[gen] = np.random.uniform(lower_bound, upper_bound)


    return childs


def selection_replacement(evaluated_population, childs):
    '''
    ABLE TO SORT THE POPULATION AND SELECT THE BEST N_individuals

    evaluated_population list. The current population
    childs list. The childs  of the bests individuals.

    RETRUNS.
    population list. The new sorted population
    '''
    _current_population = evaluated_population + childs
    _current_eval_pop = evaluate(_current_population, errf)
    _selected = _current_eval_pop[:N_individuals]
    return [indiv[1] for indiv in _selected]

def local_technique(population):
    '''
    ABLE TO CREATE A TEST INDIVIDUAL AND COMPARE ITS PERFORMANCE VS WORST INDIVIDUAL PERFORMANCE

    population list. A sorted list of N_individuals

    RETURNS
    population list. A new population.
    '''
    foreing_individual = create_individual(lower_bound, upper_bound)
    worst_individual = population[-1]
    evaluated_foreing_indiv = evaluate([foreing_individual], errf)
    evaluated_worst_indiv = evaluate([worst_individual], errf)

    if evaluated_foreing_indiv[0][0] <= evaluated_worst_indiv[0][0]:
        population[-1] = foreing_individual
        logger.info('\nLocalTechniqueApplied\n')

    return population

def plot_errors(generations, means, bests, save=False):
    fig, ax = plt.subplots()
    fig.suptitle(f'Optimizing {str_error_function} in R^{n_dim} with GA')

    ax.plot(generations, means, label='Mean Error')
    ax.plot(generations, bests,'g', label="Elite Error")

    ax.set_ylabel('Error')
    ax.set_xlabel('Generation')
    ax.legend()

    if save==True:
        now = datetime.datetime.now().strftime('%Y_%m_%d')
        plt.savefig(f'{str_error_function}{n_dim}_{now}.png')

    plt.show()

def main():

    generation = 0
    finished = False

    logger.info('\nInitializePopulation\n')

    # Logs
    generations = []
    bests_individual_per_generation = []
    worsts_individual_per_generation = []
    mean_individual_error_per_generation = []
    bests_individual_error_per_generation = []
    worsts_individual_error_per_generation = []

    # Initialize population
    population = create_population()

    while not finished:
        # Evaluate
        _evaluated = evaluate(population, errf)
        population_errors = [indiv[0] for indiv in _evaluated]
        evaluated_population = [indiv[1] for indiv in _evaluated]
        # Logs
        bests_individual_per_generation.append(evaluated_population[0])
        worsts_individual_per_generation.append(evaluated_population[-1])
        mean_individual_error_per_generation.append( np.mean(population_errors) )
        bests_individual_error_per_generation.append( np.min(population_errors) )
        worsts_individual_error_per_generation.append( np.max(population_errors) )

        logger.info(f'\n\t\t\t Generation: {generation}\t| BestError: {bests_individual_error_per_generation[generation]}\t| WorstError: {worsts_individual_error_per_generation[generation]}\t| Delta: {worsts_individual_error_per_generation[generation]-bests_individual_error_per_generation[generation]}\n')
        # Termination criteria
        finished = termination_criteria(population_errors, epsilon, generation, itermax)
        # Crossover
        childs = crossover(evaluated_population,cutoff_point=(M_gens//2))
        # Mutation
        childs = mutation(childs, p_1, p_2)
        # Selection/Replacement
        population = selection_replacement(evaluated_population, childs)
        # Local Technique
        population = local_technique(population)
        # New Generation
        generations.append(generation)
        generation += 1

    logger.info(f'\nElite:\n{bests_individual_per_generation[-1]}')
    # Plots
    plot_errors(generations, mean_individual_error_per_generation, bests_individual_error_per_generation, save=True)

if __name__ == '__main__':

    errf, lower_bound, upper_bound, str_error_function = ask_error_function()
    n_dims = [10,20,50] # Dimension of the solutions

    for n_dim in n_dims:
        N_individuals = 20 * n_dim # Population size
        M_gens = 17 * n_dim # Individual size

        p_1 = 5. # Chance of an individual to be mutated
        p_2 = 10. # Cance of a gen to be mutated

        epsilon = M_gens/N_individuals
        itermax = 500

        main()
