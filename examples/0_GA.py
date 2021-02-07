import random

modelo = [1,1,1,1,1,1,1,1,1,1] # A donde queremos llegar
largo = 10 # Cantidad de material genetico por individuo
num = 10 # Numero de individuos que formaran la poblacion inicial
pressure = 3 # Numero de individuos que se seleccionan para la reproduccion
mutacion_chance = 0.2 # La probabilidad de que haya una mutacion


def individual(cota_min, cota_max):
    '''
    Generar ADN de un individuo aleatorio
    '''
    return [random.randint(cota_min, cota_max) for _ in range(largo)]

def crear_poblacion():
    '''
    Genera la poblacion inicial
    '''
    return [individual(1,9) for _ in range(num)]

def calcular_fitness(individual):
    '''
    Calcula siel individuo se parece al modelo
    '''
    fitness = 0
    for i in range(len(individual)):
        if individual[i] == modelo [i]:
            fitness += 1

    return fitness

def seleccion_y_reproduccion(population):
    '''
    Calcula el fitness de cada individuo, ordena y reemplaza la poblacion anterior.

    '''
    puntuados = [(calcular_fitness(i), i) for i in population]
    puntuados = [i[1] for i in sorted(puntuados)]
    population = puntuados
    
    selected = puntuados[(len(puntuados)-pressure):]
    for i in range(len(population) - pressure):
        #se elige el punto donde se hara el cruce
        punto = random.randint(1, largo-1)
        padre = random.sample(selected, 2)

        population[i][:punto] = padre[0][:punto]
        population[i][punto:] = padre[1][punto:]

        return population

def mutacion(population):
    for i in range(len(population)-pressure):
        if random.random() <= mutacion_chance:
            punto = random.randint(0,largo-1)
            nuevo_valor = random.randint(1,9)

            while nuevo_valor == population[i][punto]:
                nuevo_valor = random.randint(1,9)

            population[i][punto] = nuevo_valor
    return population


def main():
    print("\n\Modelo: %s\n"%(modelo))
    population = crear_poblacion()
    print('Poblacion inicial:\n%s'%(population))

    for i in range(100):
        population = seleccion_y_reproduccion(population)
        population = mutacion(population)

    print('\nPoblacion Final:\n%s'%(population))
    print('\n\n')

if __name__ == '__main__':
    main()
