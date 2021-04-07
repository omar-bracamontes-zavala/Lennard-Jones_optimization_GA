'''
Domingo 7 Marzo 2021
Author: Omar A. Bracamontes Z.
'''

# Imports
import geneticAlgo as ga # genetic.py with the gentic algorithm
from geneticFuncs import distance
#import time

from numpy import sqrt

# Functions for Lennard-Jonnes potential
def lennard_jones(individual):
    '''
    ABLE TO CALCULATE LENNARD-JONES POTENTIAL OF A MOLECULE WITH N PARTICLES
    '''
    E_i = [] # Energía de la particula i-esima
    contador = [] # para evitar repetir  r_ij = r_ji
    for i in range(0,M_gens,3):
        LJs = [] #Sumando de E_i
        r_i = individual[i:i+3] #Posicion particula i

        for j in range(0,M_gens,3):
            if i != j:
                r_j = individual[j:j+3] #posicion particula j
                r_ij = distance(r_i, r_j) #distancia entre particula i y j
                LJ = (1/r_ij)**12 - (1/r_ij)**6
                LJs.append(LJ)
        particle_energy = 4 * sum(LJs) # Energia de la particula i respecto a las demas
        E_i.append(particle_energy)

    return 1/2*sum(E_i)


if __name__ == '__main__':

    # Genetic Algorithm parameters
    errf = lennard_jones
    str_error_function = 'Lennard-Jones Potential'

    lower_bound = -1. #1 x10^(-14) meters
    upper_bound = 1. #1 x10^(-14) meters
    p_1 = 5. # Chance of an individual to be mutated (5.)
    p_2 = 10. # Chance of a gen to be mutated (10.)


    n_dim = 3 #Dimension of every particle: in R^3
    particles_num = [x for x in range(2,38+1)] # Number of particles in the molecule [molecule=individual]

    N_individuals = 40 * n_dim # Population size (number of individuals)
    itermax = 10000

    save = True

    # For case of a molecule with N particles
    for N in particles_num:

        #start = time.time()# starting time

        M_gens = n_dim * N # Individual size
        ga.geneticAlgorithm(errf, str_error_function, lower_bound, upper_bound, p_1, p_2, n_dim, N, N_individuals, M_gens, itermax, save)


        #end = time.time()# end time

        #print(f"Runtime of the program is {end - start}")# total time taken
