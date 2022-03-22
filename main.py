import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from algorithms import policy_iteration, value_iteration

# In our assignment problem, there is a 6x6 grid (36 states) and there are 3 actions for each state (left, up, right)


# RUNNING THE ALGORITHMS

def main():
    choice = input("Enter v for value iteration and p for policy iteration: ")
    if choice == 'P' or choice == 'p':
        policy_iteration()
    elif choice == 'V' or choice == 'v':
        value_iteration()
    else:
        print('Invalid choice')


if __name__ == "__main__":
    main()
