'''
Domingo 7 Marzo 2021
Author: Omar A. Bracamontes Z.
'''

# Imports
import geneticAlgo as ga # genetic.py with the gentic algorithm
import time

from numpy import sqrt

# Functions for Lennard-Jonnes potential
def distance(r_1, r_2):
    '''
    ABLE TO CALCULATE EUCLIDEAN DISTANCE
    '''
    return sqrt(sum((r_2 - r_1)**2))

def lennard_jones(individual):
    '''
    ABLE TO CALCULATE LENNARD-JONES POTENTIAL OF A MOLECULE WITH N PARTICLES
    '''
    LJs = [] #Sumando de E_i
    E_i = [] # Energía de la particula i-esima
    for i in range(0,M_gens,3):
        r_i = individual[i:i+3]
        for j in range(0,M_gens,3):
            if i != j:
                r_j = individual[j:j+3]
                r_ij = distance(r_i, r_j)
                LJ = (1/r_ij)**12 - (1/r_ij)**6 #habia*2enelsegundoterminonucpq
                LJs.append(LJ)
        particle_energy = 1/2 * 4 * sum(LJs) # El 1/2 es para eliminar cuando r_ij = r_ji
        E_i.append(particle_energy)

    return 1/2 * sum(E_i)


if __name__ == '__main__':

    # Genetic Algorithm parameters
    errf = lennard_jones
    str_error_function = 'Lennard-Jones Potential'

    lower_bound = -1. #1 x10^(-14) meters
    upper_bound = 1. #1 x10^(-14) meters
    p_1 = 5. # Chance of an individual to be mutated (5.)
    p_2 = 10. # Cance of a gen to be mutated (10.)


    n_dim = 3 #Dimension of every particle: in R^3
    particles_num = [2]#[x for x in range(2,5)] # Number of particles in the molecule [molecule=individual]

    N_individuals = 20 * n_dim # Population size (number of individuals)
    itermax = 300

    save = True

    # For case of a molecule with N particles
    for N in particles_num:
        start = time.time()# starting time

        M_gens = n_dim * N # Individual size
        ga.geneticAlgorithm(errf, str_error_function, lower_bound, upper_bound, p_1, p_2, n_dim, N, N_individuals, M_gens, itermax, save)

        end = time.time()# end time
        print(f"Runtime of the program is {end - start}")# total time taken
