import numpy as np


def init_sudoku() -> np.ndarray:
    # Generate a sudoku puzzle
    return np.zeros((9, 9), dtype=int)
