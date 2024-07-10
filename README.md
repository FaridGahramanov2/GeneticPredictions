
April 23, 2024

1  Introduction

You are expected to solve a Linear Regression problem with a genetic algorithm. Five different datasets are provided, and you should imple- ment a genetic algorithm approach to find a promising solution for each dataset. Also, you will analyze the effect of the algorithm’s parameters on the perfor- mance in terms of the objective function and computational time.

2  Problem Description

For this assignment, we provide five different datasets consisting of two inde- pendent features (x1, x2) and one dependent feature (y). You are expected to construct a linear function to predict the target value, as seen in Equation (1). Your ultimate goal is to minimize the objective value, which is the mean squared error between the actual and estimated target values, as seen in Equation (2).

<a name="_page0_x229.05_y466.47"></a>f (x1,x2) = a × x1 + b× x2 + c = yˆ (1)

<a name="_page0_x254.03_y503.61"></a>z = min 1 (y − yˆ)2 (2)

N

N

i=1

To construct the linear function, you need to carefully determine the param- eters (e.g., a, b, and c) to minimize the estimation error. For this purpose, you are expected to implement a genetic algorithm to optimize the parameters of the linear function for each given dataset.

Moreover, the genetic algorithm has several hyper-parameters impacting the algorithm’s performance in terms of runtime and outcome. Therefore, you need to analyze the effect of each parameter on the algorithm’s performance and then specify the most promising parameters. This process is called “Parameter Tunning”.

3  Genetic Algorithm

Genetic Algorithm is a [meta-heuristic ap](https://en.wikipedia.org/wiki/Metaheuristic)proach inspired by the theory of evo- lution. Its population-based search strategy may provide a promising solution for an optimization or machine-learning problem. However, it does not guaran- tee the finding of a global optimum. A solution for the given problem should be represented as a set of genes (i.e., chromosomes) to apply the Genetic Al- gorithm. Besides, a fitness function should be defined to determine how good a chromosome is for the task. The Algorithm starts with a population of po- tential chromosomes. Iteratively, it applies a series of genetic operators to the population to create new generations of solutions. Each iteration tries to find fitter chromosomes by eliminating some chromosomes (i.e., replacement) and re- combining the existing ones (i.e., reproduction). Figure 1 [displa](#_page1_x227.04_y554.42)ys an example process of a Genetic Algorithm.

![](Aspose.Words.3b2de8d0-30a0-4bdb-b999-65a033923b65.001.png)

Figure 1:<a name="_page1_x227.04_y554.42"></a> An Example Flowchart for Genetic Algorithm

As seen in the figure, Genetic Algorithm can obtain several components as listed below:
![Aspose Words 3b2de8d0-30a0-4bdb-b999-65a033923b65 002](https://github.com/FaridGahramanov2/GeneticPredictions/assets/153610282/0cc71076-fde1-40ea-877b-22aadabf5295)
![Aspose Words 3b2de8d0-30a0-4bdb-b999-65a033923b65 001](https://github.com/FaridGahramanov2/GeneticPredictions/assets/153610282/decff6cd-e6fc-4dcc-987f-fd98e7144f7a)

- Initiate Population : Generate an initial population of potential solu- tions randomly or based on prior knowledge. Each possible solution is represented as a chromosome, usually a binary string, a real-valued vec- tor, a permutation, or any other data structure that can be manipulated. Note that the size of the population is a pre-defined parameter.
- Elitism : A strategy that preserves the best individuals in the population across generations. It involves selecting a pre-defined portion (i.e., Elitism Rate) of top-performing individuals and copying them directly to the next generation without undergoing any genetic operations.
- Selection : The process of selecting individuals from the current popula- tion for breeding and creating the next generation. The selection aims to increase the probability of selecting fitter individuals for breeding, as they are more likely to produce better offspring. Various selection techniques, such as Roulette Wheel Selection, Tournament Selection, Rank Selection, or Random Selection, can be used. Note that each selection approach has own advantages and disadvantages.
- Crossover : Combining genetic material from two parent chromosomes (chosen by selection method) to create a new individual for the next gen- eration. In the literature, there are various strategies for combining two parent chromosomes with their own advantages and disadvantages.
- Mutation : The algorithm can get stuck in local optima if the gene pool does not contain the necessary genes to lead to the global optima. Mu- tation randomly alters the genetic material of a chromosome based on a mutation rate parameter to create a new solution. Thus, the gene pool in the current population is diverse. Various mutation methods can be implemented depending on the task and the chromosome structure.
- Stopping Condition : The algorithm iterates over generation until the stopping condition is met. The literature has several stopping conditions like maximum iteration, time limit, or stopping after convergence. You can try different strategies to analyze which one gives a better outcome.
- Intensification : The strategic focus on promising regions of the solution space by using selection, elitism, and exploitation of the best solutions. This ensures convergence towards optimal or near-optimal solutions by emphasizing exploration in the most promising areas while maintaining diversity for effective search. You can use different strategies to explore the most promising regions deeply.
- Diversification : Maintaining diversity within the population to prevent premature convergence to sub-optimal solutions. This is achieved via mu- tation, crossover, and introducing new individuals to explore different re- gions of the solution space. Diversification ensures that the algorithm continues to explore a wide range of potential solutions, improving the likelihood of finding the global optimum. You can use different strategies to cover more regions to get better solutions.

The Genetic Algorithm’s components can vary depending on the implemen- tation design. The methods of components (e.g., selection, crossover, chromo- somes, initial population, mutation, stopping condition, etc.) are up to the students. However, the performance of the chosen implementation is significant for the grade. Thus, the students can implement several methods and then com- pare their performance in their report. Moreover, the students are expected to evaluate the effect of the parameters (e.g., mutation rate, elitism rate, popula- tion size, maximum iteration, etc.) on the algorithm’s performance in terms of computational time and objective function. An example graph for the parame- ter analysis on the algorithm’s performance is demonstrated in Figure 2 [where ](#_page3_x233.66_y394.63)the task is a maximization problem. Note that it is just a representation; your graphs can differ from the example.

![](Aspose.Words.3b2de8d0-30a0-4bdb-b999-65a033923b65.002.png)

Figure 2:<a name="_page3_x233.66_y394.63"></a> An Example Graph for Parameter Analyze

4  Implementation

In this text, the instructions for a programming assignment are given. The assignment involves implementing a Genetic Algorithm for the given problem in Python. As in the first assignment, the students are expected to implement their approach by utilizing the given code base. Students must utilize Python programming language with version 3.10, and Jupyter is not allowed. Students who need to learn how to code with Python can learn from available sources on the internet. Some helpful links to learn Python programming language from scratch are shared at the end of the document (Section 6). [Inside](#_page6_x133.77_y269.27) the code base:

- data~~ generator.py : This Python script has two free functions to ran- domly generate new dataset and load a dataset from the file respectively.
- GeneticSolver.py : GeneticSolver class employs the Genetic Algorithm. For Assignment 2, you have to implement your Genetic Algorithm al- gorithm in solve method. You are free to make changes (i.e., defining variables and methods) in this class. However, you cannot change the name of the class, the constructor of the class and the definition of solve method.
- Main.py : This Main script is provided for your implementation. You are free to make any changes in this Python file. Note that, we will test your algorithm in our “Main” file. To use this Python script, all you need is defining a proper data file path in “FILE~~ PATH “ variable. Moreover, Genetic Algorithm is a non-deterministic approach that the [random seed ](https://en.wikipedia.org/wiki/Random_seed)can change the performance of the algorithm. Therefore, you need to also evaluate your algorithm with different random seeds.

For the evaluation, five distinct example datasets are provided. In each dataset (e.g., “dataset1.pkl”, “dataset2.pkl”, “dataset3.pkl”, “dataset4.pkl”, and “dataset5.pkl”), both dependent (y) and independent (x1 and x2) feature matrices are provided as [Pickle ](https://docs.python.org/3.10/library/pickle.html)format. The matrices are 2D [NumPy array ](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)objects. Additionally, “dataset~~ generator.py” Python script is provided to gen- erate more datasets for the evaluation.

