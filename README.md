# Conjugate Gradient Descent Implementation

This project is about numerically checking a conjecture in matrix theory, here referred to as Kaustav's conjecture. The conjecture arises from the problem of maximizing orbital correlations in so-called fermionic 'free' quantum states over unitary rotations of the underlying orbital space. We use a gradient descent method.

## Project Structure

- `notebooks/`: Contains Jupyter notebook to explain the conjecture
- `src/kaustav_conj/`: Main source code module
  - `core.py`: Core conjugate gradient algorithm implementation
  - `utils.py`: Utility functions (matrix operations, convergence checks, etc.)
- `python_sampling_scripts/`: Python scripts for systematically checking the conjecture
- `tests/`: Unit tests for the implementation

## Installation

```bash
pip install -r requirements.txt
```