# NN_optimization_GA
This is a repository for a Lennard-Jones potential optimization with Genetic Algorithm

## Check list
- [x] 1_example
	- This folder has a [modified genetic algorithm](https://www.sciencedirect.com/science/article/abs/pii/S0096300308002907) for optimizing real functions like:
		- [Sphere function](http://benchmarkfcns.xyz/benchmarkfcns/spherefcn.html)
		- [Rosenbrock function](https://www.sfu.ca/~ssurjano/rosen.html)
		- [Ackley function](https://www.sfu.ca/~ssurjano/ackley.html)
		- [Rastrigin function](https://www.sfu.ca/~ssurjano/rastr.html)
		- [Schwefel function](https://www.sfu.ca/~ssurjano/schwef.html) 
- [x] geneticFuncs.py
	- This script holds the basics functions to build a genetic algorithm
- [x] geneticAlgo.py
	- This script holds the genetic algorithm logic using geneticFuncs.py
- [x] main.py
	- This script holds the Lennard-Jones potential optimization using geneticAlgo.py
