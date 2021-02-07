# Replicating [Modifications of real code genetic algorithm for global optimization (2008)](https://www.sciencedirect.com/science/article/abs/pii/S0096300308002907) paper 

The tested error functions are:
- [Sphere function](http://benchmarkfcns.xyz/benchmarkfcns/spherefcn.html)
- [Rosenbrock function](https://www.sfu.ca/~ssurjano/rosen.html)
- [Ackley function](https://www.sfu.ca/~ssurjano/ackley.html)
- [Rastrigin function](https://www.sfu.ca/~ssurjano/rastr.html)
- [Schwefel function](https://www.sfu.ca/~ssurjano/schwef.html)

##FlowChart

```flow
st=>start: Generate initial population
op1=>operation: Evaluate
cond1=>condition: Done Yes or No? (TerminationCriteria)
op2=>operation: Crossover
op3=>operation: Mutation
cond2=>condition: Successful individual?
op4=>operation: Selection
op5=>operation: Replacement
op6=>operation: Local technique
e=>end: Stop and plot

st->op1->cond1
cond1(no)->op2->op3->cond2
cond1(yes)->e
cond2(yes)->op4->op5->op6->op1
cond2(no)->e

```
 
