'''
Domingo 7 Marzo 2021
Author: Omar A. Bracamontes Z.
'''
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def create_individual(lower_bound, upper_bound, M_gens):
    '''
    ABLE TO CREATE A NEW INDIVIDUAL.

    lower_bound float. Domain's lower bound.
    upper_bound float. Domain's upper bound.

    RETURNS.
    individual numpyarray. A individual with continous random M gens
    '''
    return np.random.uniform(lower_bound, upper_bound, M_gens)

def create_population(lower_bound, upper_bound, M_gens, N_individuals):
    '''
    ABLE TO CREATE A POPULATION.

    RETURNS.
    population list. A list with N individuals
    '''
    return [create_individual(lower_bound, upper_bound, M_gens) for _ in range(N_individuals)]

def evaluate(population, errf):
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
    if iteration >= itermax:
        logger.info('\nTerminationCriteria: iteration>=itermax\n')
        return True
    else:
        return False


def crossover(evaluated_population, cutoff_point, M_gens, N_individuals):
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

def mutation(childs, p_1, p_2, M_gens, lower_bound, upper_bound):
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


def selection_replacement(evaluated_population, childs, errf, N_individuals):
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

def local_technique(population, M_gens, lower_bound, upper_bound, errf):
    '''
    ABLE TO CREATE A TEST INDIVIDUAL AND COMPARE ITS PERFORMANCE VS WORST INDIVIDUAL PERFORMANCE

    population list. A sorted list of N_individuals

    RETURNS
    population list. A new population.
    '''
    foreing_individual = create_individual(lower_bound, upper_bound, M_gens)
    worst_individual = population[-1]
    evaluated_foreing_indiv = evaluate([foreing_individual], errf)
    evaluated_worst_indiv = evaluate([worst_individual], errf)

    if evaluated_foreing_indiv[0][0] <= evaluated_worst_indiv[0][0]:
        population[-1] = foreing_individual

    return population

def plot_errors(generations, means, bests, worsts, str_error_function, n_dim, N, save=False):
    fig, ax = plt.subplots()
    fig.suptitle(f'Optimizing {str_error_function} in R^{n_dim} with GA\n N={N}\nBestError: {bests[-1]}')

    ax.plot(generations, means,'b', alpha=0.8, label='Mean Error')
    ax.plot(generations, bests,'g', alpha=1.0, label="Elite Error")
    #ax.plot(generations, worsts,'r', alpha=0.6, label="Dreg Error")

    ax.grid(alpha=0.5)
    ax.set_ylabel(r'$Energy/\epsilon$')
    ax.set_xlabel('Generation')
    ax.legend()

    if save==True:
        plt.savefig(f'/home/omar/Documentos/Modulares/M_1/figs/{N}N_E{bests[-1]}.png')
    #plt.show()

def save_results_csv(bests_individual_per_generation,bests_individual_error_per_generation,N,itermax,N_individuals,save):
    if save==True:
        final_best_dict = {'positions':bests_individual_per_generation[-1]}
        final_best = pd.DataFrame(final_best_dict)
        final_best.to_csv('/home/omar/Documentos/Modulares/M_1/logs/{}N_E{}_g{}_p{}.csv'.format(N,bests_individual_error_per_generation[-1],itermax,N_individuals), sep = ' ', index = False)
