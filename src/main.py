import numpy as np
import pandas as pd

from utils import init_sudoku, print_sudoku, verify_sudoku
from ga import genetic_algorithm

test_puzzle = (
    "070000043040009610800634900094052000358460020000800530080070091902100005007040802"
)
test_solution = (
    "679518243543729618821634957794352186358461729216897534485276391962183475137945862"
)

if __name__ == "__main__":
    df = pd.read_csv("sudoku_data.csv")
    idx = np.random.randint(0, df.shape[0])

    puzzle = init_sudoku(df["puzzle"][idx])
    solution = init_sudoku(df["solution"][idx])
    print("Puzzle:")
    print_sudoku(puzzle)
    print("Solution:")
    print_sudoku(solution)
    print(verify_sudoku(solution))

    ga_solution = genetic_algorithm(puzzle, 200, 100000)
    print("GA Solution:")
    print_sudoku(ga_solution)
    print("Puzzle:")
    print_sudoku(puzzle)
    print("Solution:")
    print_sudoku(solution)

    print("Solution is correct: ", verify_sudoku(ga_solution))
    print("GA Solution == Solution: ", np.array_equal(solution, ga_solution))
