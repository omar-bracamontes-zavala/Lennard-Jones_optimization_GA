from numpy import exp, sin, cos, sqrt, pi
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def sphere(individual):
    return sum(individual**2)


def trans_sphere(individual, origin):
    try:
        return sum((individual - origin)**2)
    except ValueError as e:
        logger.warning('Numpy Array dimensions must agree')


def rosenbrock(individual):
    result = 0
    _dim = len(individual)

    for element in range(_dim - 1):
        result += (individual[element+1]-individual[element]**2)**2 +(individual[element] - 1)**2

    return result


def ackley(individual, a=20, b=0.2, c=2*pi):
    _dim = len(individual)

    return -a * exp(-b * sqrt(1/_dim * sphere(individual))) - exp(1/_dim * sum(cos(c*individual))) + a + exp(1)


def rastrigin(individual):
    _dim = len(individual)

    return 10*_dim + sum( individual**2 - 10*cos(2*pi*individual) )


def schwefel(individual,k=418.9829):
    _dim = len(individual)

    return k*_dim - sum( individual*sin( sqrt(abs(individual) ) ) )
