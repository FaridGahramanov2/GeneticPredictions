# Farid Gahramanov S023378 - CS 451 - Homework 2

import numpy as np
import random
from random import Random


class GeneticSolver:
    """
        This class employs your genetic algorithm implementations.
    """
    seed: int    #: Specified Random Seed. You need to set specified random seed
    a: float     #: **a** parameter in the linear function
    b: float     #: **b** parameter in the linear function
    c: float     #: **c** parameter in the linear function
    rnd: Random  #: Example random object. You can use it

    def __init__(self, seed: int):
        """
            Construction of the class.

            :param seed: Specified Random Seed
        """
        self.seed = seed

        # Setting random seed globally
        self.rnd = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.population_size = 50  # Example size, adjust as needed
        self.generations = 100  # Example generation count
        self.mutation_rate = 0.1  # Example mutation rate
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0

    def initialize_population(self):
        return np.random.uniform(-10, 10, (self.population_size, 3))

    def calculate_fitness(self, population, x1, x2, y):
        return np.array([np.mean((y - (individual[0] * x1 + individual[1] * x2 + individual[2])) ** 2) for individual in population])

    def selection(self, population, fitness):
        selection_count = self.population_size // 2
        selected_indices = [np.random.choice(self.population_size, 2, replace=False) for _ in range(selection_count)]
        return np.array([population[i] if fitness[i] < fitness[j] else population[j] for i, j in selected_indices])


    # def selection_case_two_roulett_wheel(self, population, fitness):
    #     max_fitness = np.max(fitness)
    #     adjusted_fitness = max_fitness - fitness + 1
    #     probabilities = adjusted_fitness / np.sum(adjusted_fitness)
    #     selected_indices = np.random.choice(range(self.population_size), size=self.population_size // 2, replace=True, p=probabilities)
    #     return population[selected_indices]

    def crossover(self, selected):
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            cross_point = np.random.randint(1, 3)
            offspring.append(np.concatenate([selected[i][:cross_point], selected[i+1][cross_point:]]))
            offspring.append(np.concatenate([selected[i+1][:cross_point], selected[i][cross_point:]]))
        return np.array(offspring)

    # def uniform_crossover_case_two(self, selected):
    #     offspring = []
    #     num_parents = len(selected)
    #     if num_parents % 2 != 0:
    #         num_parents -= 1
    #     for i in range(0, num_parents, 2):
    #         parent1, parent2 = selected[i], selected[i + 1]
    #         mask = np.random.randint(0, 2, size=3)
    #         child1 = np.where(mask, parent1, parent2)
    #         child2 = np.where(mask, parent2, parent1)
    #         offspring.extend([child1, child2])
    #     return np.array(offspring)

    def mutate(self, offspring):
        for individual in offspring:
            if np.random.rand() < self.mutation_rate:
                mutation_adjustment = np.random.normal(0, 1, individual.shape)
                individual += mutation_adjustment
        return offspring

    # def mutate_case_two(self, offspring, mutation_rate=0.2):
    #     for individual in offspring:
    #         if np.random.rand() < mutation_rate:
    #             individual += np.random.normal(0, 0.5, size=individual.shape)
    #     return offspring

    def solve(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
        """
            This method runs your genetic algorithm for the given dataset

            **Note**: You need specify the parameters (a,b,c) at the end of the solve method

            :param x1: First independent features from the dataset as a NumPy array
            :param x2: Second independent features from the dataset as a NumPy array
            :param y: Target dependent values from the dataset as a NumPy array
        """
        population = self.initialize_population()
        best_params = None
        lowest_fitness = float('inf')

        for _ in range(self.generations):
            fitness_scores = self.calculate_fitness(population, x1, x2, y)
            best_current_idx = np.argmin(fitness_scores)

            if fitness_scores[best_current_idx] < lowest_fitness:
                lowest_fitness = fitness_scores[best_current_idx]
                best_params = population[best_current_idx]

            if lowest_fitness < 0.01:
                break

            selected = self.selection(population, fitness_scores)
            offspring = self.crossover(selected)
            mutated = self.mutate(offspring)

            if len(mutated) < self.population_size:
                additional = self.initialize_population()[:(self.population_size - len(mutated))]
                mutated = np.vstack([mutated, additional])

            population = mutated
            self.a, self.b, self.c = best_params if best_params is not None else (0, 0, 0)

    # def solve_case_two(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
    #     population = self.initialize_population()
    #     for _ in range(self.generations):
    #         fitness = self.calculate_fitness(population, x1, x2, y)
    #         selected = self.selection_case_two_roulett_wheel(population, fitness)
    #         if len(selected) < 2:
    #             continue
    #
    #         offspring = self.uniform_crossover_case_two(selected)
    #
    #         if len(offspring) < len(population):
    #             additional_offspring = self.initialize_population()[:len(population) - len(offspring)]
    #             offspring = np.vstack([offspring, additional_offspring])
    #
    #         mutated = self.mutate_case_two(offspring, mutation_rate=0.2)
    #
    #         if len(mutated) != self.population_size:
    #             mutated = mutated[:self.population_size]
    #
    #         population = mutated
    #         best_idx = np.argmin(fitness)
    #         self.a, self.b, self.c = population[best_idx]


    def calculate_objective(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
        """
            This method calculates objective function value (i.e., mean squared error)
        :param x1: First independent features from the dataset as a NumPy array
        :param x2: Second independent features from the dataset as a NumPy array
        :param y: Target dependent values from the dataset as a NumPy array
        :return: Prediction error
        """
        return np.mean(np.power(x1 * self.a + x2 * self.b + self.c - y, 2.))
