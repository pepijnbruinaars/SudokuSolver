import numpy as np
import time

from utils import print_sudoku


def init_population(population_size, puzzle):
    population = np.array([puzzle.copy() for _ in range(population_size)])
    for individual in population:
        for i in range(9):
            for j in range(9):
                if puzzle[i, j] == 0:
                    individual[i, j] = np.random.randint(1, 10)
    return population


def fitness(solution):
    row_fitness = sum(9 - len(np.unique(solution[i, :])) for i in range(9))
    col_fitness = sum(9 - len(np.unique(solution[:, j])) for j in range(9))
    subgrid_fitness = sum(
        9 - len(np.unique(solution[i : i + 3, j : j + 3]))
        for i in range(0, 9, 3)
        for j in range(0, 9, 3)
    )
    return 9 * 9 * 3 - (row_fitness + col_fitness + subgrid_fitness)


def mutate_individual(individual, fixed_cells):
    mutated_individual = individual.copy()
    row = np.random.randint(0, 9)
    col = np.random.randint(0, 9)

    while fixed_cells[row, col] == 1:
        row = np.random.randint(0, 9)
        col = np.random.randint(0, 9)

    row_choices = [i for i in range(9) if fixed_cells[row, i] == 0]
    col_choices = [i for i in range(9) if fixed_cells[i, col] == 0]
    subgrid_choices = [
        (r, c)
        for r in range(row // 3 * 3, row // 3 * 3 + 3)
        for c in range(col // 3 * 3, col // 3 * 3 + 3)
        if fixed_cells[r, c] == 0
    ]

    if np.random.random() > 0.33:
        swap_row = np.random.choice(row_choices)
        mutated_individual[row, col], mutated_individual[row, swap_row] = (
            mutated_individual[row, swap_row],
            mutated_individual[row, col],
        )
    elif np.random.random() > 0.5:
        swap_col = np.random.choice(col_choices)
        mutated_individual[row, col], mutated_individual[swap_col, col] = (
            mutated_individual[swap_col, col],
            mutated_individual[row, col],
        )
    else:
        swap_r, swap_c = subgrid_choices[np.random.randint(len(subgrid_choices))]
        mutated_individual[row, col], mutated_individual[swap_r, swap_c] = (
            mutated_individual[swap_r, swap_c],
            mutated_individual[row, col],
        )

    return mutated_individual


def crossover(individual1, individual2, fixed_cells):
    offspring = individual1.copy()
    for row in range(9):
        for col in range(9):
            if np.random.random() > 0.5 and fixed_cells[row, col] == 0:
                offspring[row, col] = individual2[row, col]
    return offspring


def select_individuals(population, fitnesses):
    selected = []
    for _ in range(len(population) // 2):
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        selected.append(
            population[idx1] if fitnesses[idx1] > fitnesses[idx2] else population[idx2]
        )
    return selected


def genetic_algorithm(puzzle: np.ndarray, pop_size: int, max_generations: int = 10000):
    """Genetic algorithm to solve a sudoku puzzle.

    Args:
        puzzle (np.ndarray): The puzzle to solve.
        pop_size (int): The population size.
        max_generations (int, optional): The maximum number of generations the GA is allowed to run for. Defaults to 10000.

    Returns:
        np.ndarray: The solution to the puzzle.
    """
    start_time = time.time()
    fixed_cells = np.where(puzzle != 0, 1, 0)
    population = init_population(pop_size, puzzle)
    best_fitness = 0
    best_individual: np.ndarray | None = None
    generations_with_no_change = 0

    for i in range(max_generations):
        fitnesses = [fitness(ind) for ind in population]
        max_fitness = max(fitnesses)
        if max_fitness == (9 * 9 * 3):
            best_individual = population[np.argmax(fitnesses)]
            break

        selected = select_individuals(population, fitnesses)
        new_population = []
        for j in range(0, len(selected), 2):
            offspring1 = crossover(selected[j], selected[j + 1], fixed_cells)
            offspring2 = crossover(selected[j + 1], selected[j], fixed_cells)
            new_population.extend(
                [selected[j], selected[j + 1], offspring1, offspring2]
            )

        new_population = [mutate_individual(ind, fixed_cells) for ind in new_population]
        population = new_population[: len(population)]

        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_individual = population[np.argmax(fitnesses)]
            generations_with_no_change = 0
        else:
            generations_with_no_change += 1

        if i % 100 == 0:
            print(
                f"\nGeneration: {i}, Best Fitness: {best_fitness} \nCurrent best solution: \n"
            )
            if best_individual is not None:
                print_sudoku(best_individual)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f}s")
    return best_individual
