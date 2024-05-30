import numpy as np


def init_sudoku(input: str) -> np.ndarray:
    """Function that parses a sudoku string into a 2x2 numpy array.
    The string input is a 81-character string representing the sudoku puzzle. The order of the characters is from left to right, top to bottom.

    Args:
        input (str): The string input representing the sudoku puzzle.

    Returns:
        np.ndarray: A 2x2 numpy array representing the sudoku puzzle.
    """
    sudoku = np.array([int(char) for char in input])
    sudoku = sudoku.reshape((9, 9))
    return sudoku


def print_sudoku(sudoku: np.ndarray) -> None:
    """Print the sudoku puzzle in a readable format.

    Args:
        sudoku (np.ndarray): The sudoku puzzle.
    """
    for row in range(9):
        if row % 3 == 0 and row != 0:
            print("-" * 21)
        for col in range(9):
            if col % 3 == 0 and col != 0:
                print("|", end=" ")
            print(sudoku[row, col], end=" ")
        print()


def verify_sudoku(solution: np.ndarray) -> bool:
    """Verify the solution of a sudoku puzzle.

    Args:
        solution (np.ndarray): The solution of the sudoku puzzle.

    Returns:
        bool: True if the solution is correct, False otherwise.
    """
    # Check rows
    for row in range(9):
        if len(set(solution[row])) != 9:
            return False

    # Check columns
    for col in range(9):
        if len(set(solution[:, col])) != 9:
            return False

    # Check 3x3 squares
    for row in range(0, 9, 3):
        for col in range(0, 9, 3):
            if len(set(solution[row : row + 3, col : col + 3].flatten())) != 9:
                return False

    return True
