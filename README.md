

# Asynchronous-Decentralized-Constrained-Optimization

This repository contains the Python code and Jupyter notebooks associated with the experiments presented in our paper. The code is organized into four main Python scripts (`exp1.py`, `exp2.py`, `exp3.py`, `exp4.py`) and one Jupyter notebook that includes the appendix experiments. Below, you will find a brief description of each experiment, how to run them, and the required dependencies.

## Repository Structure

- `utilities/`: Contains all utility functions used across experiments.
- `problems/`: Includes problem definitions, gradients, and projections.
- `data/`: Stores the MNIST dataset used in some of the experiments.
- `optimizers/`: Contains implementations of centralized and decentralized algorithms.
- `graph/`: Holds scripts for generating gossip matrices. `graph_pg_extra.py` is a more general script that provides additional matrices necessary for the PG-EXTRA algorithm.
- `analysis/`: Contains functions for computing metrics such as the optimality gap and feasibility gap, which are used to compare algorithm performance.

## Experiments

### exp1.py
Solves a constrained optimization problem and compares the performance to DAGP and its throttled variant. To run this experiment, execute:
```bash
python exp1.py
```

### exp2.py
Addresses an unconstrained logistic regression problem and makes comparisons to APPG and ASY-SPA. Run this experiment with:
```bash
python exp2.py
```

### exp3.py
Focuses on a constrained problem over undirected graphs, comparing the results to ASY-PG-EXTRA. Execute the experiment using:
```bash
python exp3.py
```

### exp4.py
Investigates the robustness to message losses with a communication failure probability of \(p\). This experiment simulates asynchronous setups based on a global iteration counter, which advances whenever at least one agent computes. To run exp4, use:
```bash
python exp4.py
```

The asynchronous setups are generated by creating computation times for nodes (`TV_nodes` parameter) and computing all unique times when at least one node is activated (`T_active` parameter). Algorithms update their variables at each iteration only for those nodes activated at that specific time, for a total of `max_iter` iterations.

### Jupyter Notebook
Contains experiments found in the appendix of the paper. These can be run by opening the notebook in a Jupyter environment and executing the cells in order.

## Dependencies

To run the experiments in this repository, you will need to install several dependencies. You can install all required libraries using the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should list all the necessary libraries, such as numpy, scipy, matplotlib, jupyter, etc. Ensure you create this file and list all dependencies your project needs.
