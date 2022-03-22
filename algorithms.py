from helper import *
from environment import initialiseEnvironment, nextState
import numpy as np

# Get data
environment = initialiseEnvironment()

# ALGORITHMS

# VALUE ITERATION ALGORITHM


def calculateValue(current_state):
    updated_value = np.zeros(4)
    for action in range(4):  # for all actions

        for next_state, probability in nextState(current_state, environment[8][
            action]).items():  # for all possible future states from an action
            if probability > 0:
                # converting next state from string to int indexes to get the x and y coordinates of the next state
                next_state_num = stringToInt(next_state)

                # calculating the value for this particular state from this action
                updated_value[action] += probability * (
                        environment[1][next_state] + environment[9] * environment[2][next_state_num[0]][next_state_num[1]])

    # how much has the value changed since the last iteration
    environment[4][current_state[0]][current_state[1]] = abs(
        environment[2][current_state[0]][current_state[1]] - max(updated_value))

    # update the new values
    environment[2][current_state[0]][current_state[1]] = max(updated_value)
    environment[3][current_state[0]][current_state[1]] = int(np.argmax(updated_value))

    printEachIteration(current_state, environment[2], environment[8], environment[3])


# POLICY ITERATION ALGORITHM

# As the q-table is initialised with all 0s, and action[0] is up, we begin with a default policy of always going up

def calculatePolicy(current_state):
    # POLICY EVALUATION
    # calculate the value of each state based on the policy
    action = environment[3][current_state[0]][current_state[1]]
    total_evaluation = 0
    for next_state, probability in nextState(current_state, environment[8][
        int(action)]).items():  # for all possible future states from an action
        if probability > 0:
            # converting next state from string to int indexes to get the x and y coordinates of the next state
            next_state_num = stringToInt(next_state)

            # calculating the value for this particular state from this action
            total_evaluation += probability * (
                    environment[1][next_state] + environment[9] * environment[2][next_state_num[0]][next_state_num[1]])

    environment[2][current_state[0]][current_state[1]] = total_evaluation

    # POLICY IMPROVEMENT
    # use the calculated values to get the best policy

    updated_value = np.zeros(4)

    for action in range(4):  # for all actions
        for next_state, probability in nextState(current_state, environment[8][
            action]).items():  # for all possible future states from an action
            if probability > 0:
                # converting next state from string to int indexes to get the x and y coordinates of the next state
                next_state_num = stringToInt(next_state)

                # calculating the value for this particular state from this action
                updated_value[action] += probability * (
                        environment[1][next_state] + environment[9] * environment[2][next_state_num[0]][next_state_num[1]])

    # calculating change in policy
    if environment[3][current_state[0]][current_state[1]] == int(np.argmax(updated_value)):  # policy did not change
        environment[5][current_state[0]][current_state[1]] = 0
    else:  # policy did change
        environment[5][current_state[0]][current_state[1]] = 1

    # updating the policy
    environment[3][current_state[0]][current_state[1]] = int(np.argmax(updated_value))

    printEachIteration(current_state, environment[2], environment[8], environment[3])


# CONTROLLING ALGORITHMS


def oneIterationValue():
    for i in range(6):
        for j in range(6):
            calculateValue(environment[0][i][j])

    valueOverTime(environment[6], environment[2])

    # check if convergence has happened
    if valueConverged(environment[4], environment[10]) is True:
        return False
    else:
        return True


def value_iteration():
    print('Value Iteration...')
    i = 1
    run = True
    while run:
        run = oneIterationValue()
        if run is True:
            print('Iteration Number: ', i)
            pass

        else:
            print('Policy Converged at', i, '\n')
        i = i + 1

    plotOptimalPolicy(environment[0], environment[8], environment[3])
    plotOptimalValue(environment[0], environment[2])
    # printOptimalValue()
    plotValueOverTime(environment[6], 'Value Iteration')


def oneIterationPolicy():
    for i in range(6):
        for j in range(6):
            calculatePolicy(environment[0][i][j])

    valueOverTime(environment[7], environment[2])

    if policyConverged(environment[5]) is True:
        return False
    else:
        return True


def policy_iteration():
    print('Policy Iteration...')
    i = 1
    run = True
    while run:
        run = oneIterationPolicy()
        if run is True:
            pass
            # print('Iteration Number: ', i)
        else:
            print('Policy Converged at ', i, '\n')
        i = i + 1

    plotOptimalPolicy(environment[0], environment[8], environment[3])
    plotOptimalValue(environment[0], environment[2])
    plotValueOverTime(environment[7], 'Policy Iteration')