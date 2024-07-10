import time

from GeneticSolver import GeneticSolver
from data_generator import load_dataset


if __name__ == "__main__":
    FILE_PATH = "dataset1.pkl"
    SEED = 123  # You need to evaluate your approach with various seed due to the non-deterministic behavior of the GA

    dataset = load_dataset(FILE_PATH)

    x1 = dataset["x1"]
    x2 = dataset["x2"]
    y = dataset["y"]

    solver = GeneticSolver(SEED)

    start_time = time.time()
    solver.solve_case_two(x1, x2, y)
    end_time = time.time()

    print("Objective Value:", solver.calculate_objective(x1, x2, y), sep="\t")
    print("Elapsed Time (ms):", end_time - start_time, sep="\t")
    print("a:", solver.a, sep="\t")
    print("b:", solver.b, sep="\t")
    print("c:", solver.c, sep="\t")
