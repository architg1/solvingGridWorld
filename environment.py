import numpy as np

# BUILDING THE ENVIRONMENT

# In a 6x6 grid with 36 states, the transition function should return the probability of going to each state
# From a state (say s_x), we can only go to 4 different states at most — take a left, take a right, go up, go down
# We define a function to tell us which state it will go to and with what probability

# STATE IS A 2D ARRAY
# VALUE IS A 2D ARRAY

value = np.zeros((6, 6))
q_table = np.zeros((6, 6))
value_change = np.zeros((6, 6))
policy_change = np.ones((6, 6))

value_over_time_vi = {'00': [], '01': [], '02': [], '03': [], '04': [], '05': [],
                      '10': [], '11': [], '12': [], '13': [], '14': [], '15': [],
                      '20': [], '21': [], '22': [], '23': [], '24': [], '25': [],
                      '30': [], '31': [], '32': [], '33': [], '34': [], '35': [],
                      '40': [], '41': [], '42': [], '43': [], '44': [], '45': [],
                      '50': [], '51': [], '52': [], '53': [], '54': [], '55': []}

value_over_time_pi = {'00': [], '01': [], '02': [], '03': [], '04': [], '05': [],
                      '10': [], '11': [], '12': [], '13': [], '14': [], '15': [],
                      '20': [], '21': [], '22': [], '23': [], '24': [], '25': [],
                      '30': [], '31': [], '32': [], '33': [], '34': [], '35': [],
                      '40': [], '41': [], '42': [], '43': [], '44': [], '45': [],
                      '50': [], '51': [], '52': [], '53': [], '54': [], '55': []}

actions = ('up', 'right', 'down', 'left')
gamma = 0.99  # discount factor
delta = 0.01  # convergence factor

reward_function = {'00': -0.04, '01': -0.04, '02': -0.04, '03': -0.04, '04': -0.04, '05': -0.04,
                   '10': -0.04, '11': -100, '12': -100, '13': -100, '14': -1, '15': -0.04,
                   '20': -0.04, '21': -0.04, '22': -0.04, '23': -1, '24': -0.04, '25': 1,
                   '30': -0.04, '31': -0.04, '32': -1, '33': -0.04, '34': 1, '35': -0.04,
                   '40': -0.04, '41': -1, '42': -0.04, '43': 1, '44': -100, '45': -1,
                   '50': 1, '51': -100, '52': 1, '53': -0.04, '54': -0.04, '55': 1}

state_00 = [0, 0]
state_01 = [0, 1]
state_02 = [0, 2]
state_03 = [0, 3]
state_04 = [0, 4]
state_05 = [0, 5]

state_10 = [1, 0]
state_11 = [1, 1]
state_12 = [1, 2]
state_13 = [1, 3]
state_14 = [1, 4]
state_15 = [1, 5]

state_20 = [2, 0]
state_21 = [2, 1]
state_22 = [2, 2]
state_23 = [2, 3]
state_24 = [2, 4]
state_25 = [2, 5]

state_30 = [3, 0]
state_31 = [3, 1]
state_32 = [3, 2]
state_33 = [3, 3]
state_34 = [3, 4]
state_35 = [3, 5]

state_40 = [4, 0]
state_41 = [4, 1]
state_42 = [4, 2]
state_43 = [4, 3]
state_44 = [4, 4]
state_45 = [4, 5]

state_50 = [5, 0]
state_51 = [5, 1]
state_52 = [5, 2]
state_53 = [5, 3]
state_54 = [5, 4]
state_55 = [5, 5]

state = list()
state0 = list()
state1 = list()
state2 = list()
state3 = list()
state4 = list()
state5 = list()

state0.append(state_00)
state0.append(state_01)
state0.append(state_02)
state0.append(state_03)
state0.append(state_04)
state0.append(state_05)

state1.append(state_10)
state1.append(state_11)
state1.append(state_12)
state1.append(state_13)
state1.append(state_14)
state1.append(state_15)

state2.append(state_20)
state2.append(state_21)
state2.append(state_22)
state2.append(state_23)
state2.append(state_24)
state2.append(state_25)

state3.append(state_30)
state3.append(state_31)
state3.append(state_32)
state3.append(state_33)
state3.append(state_34)
state3.append(state_35)

state4.append(state_40)
state4.append(state_41)
state4.append(state_42)
state4.append(state_43)
state4.append(state_44)
state4.append(state_45)

state5.append(state_50)
state5.append(state_51)
state5.append(state_52)
state5.append(state_53)
state5.append(state_54)
state5.append(state_55)

state.append(state0)
state.append(state1)
state.append(state2)
state.append(state3)
state.append(state4)
state.append(state5)


def isLeftEdge(cur_state):
    if cur_state[1] == 0:
        return True
    else:
        return False


def isRightEdge(cur_state):
    if cur_state[1] == 5:
        return True
    else:
        return False


def isUpEdge(cur_state):
    if cur_state[0] == 5:
        return True
    else:
        return False


def isDownEdge(cur_state):
    if cur_state[0] == 0:
        return True
    else:
        return False


def isLeftWall(cur_state):  # there is a wall to the left
    if cur_state[0] == 1 and cur_state[1] == 4 or cur_state[0] == 4 and cur_state[1] == 5 or cur_state[0] == 5 and \
            cur_state[1] == 2:
        return True
    else:
        return False


def isRightWall(cur_state):  # there is a wall to the right
    if cur_state[0] == 1 and cur_state[1] == 0 or cur_state[0] == 5 and cur_state[1] == 0 or cur_state[0] == 4 and \
            cur_state[1] == 3:
        return True
    else:
        return False


def isUpWall(cur_state):  # there is a wall up
    if cur_state[0] == 0 and cur_state[1] == 1 or cur_state[0] == 0 and cur_state[1] == 2 or cur_state[0] == 0 and \
            cur_state[1] == 3 or cur_state[
        0] == 4 and cur_state[1] == 1 or cur_state[0] == 3 and cur_state[1] == 4:
        return True
    else:
        return False


def isDownWall(cur_state):  # there is a wall down
    if cur_state[0] == 2 and cur_state[1] == 1 or cur_state[0] == 2 and cur_state[1] == 2 or cur_state[0] == 2 and \
            cur_state[1] == 3 or cur_state[
        0] == 5 and cur_state[1] == 4:
        return True
    else:
        return False


def nextState(current_state, action):
    next_state = {'00': 0, '01': 0, '02': 0, '03': 0, '04': 0, '05': 0,
                  '10': 0, '11': 0, '12': 0, '13': 0, '14': 0, '15': 0,
                  '20': 0, '21': 0, '22': 0, '23': 0, '24': 0, '25': 0,
                  '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0,
                  '40': 0, '41': 0, '42': 0, '43': 0, '44': 0, '45': 0,
                  '50': 0, '51': 0, '52': 0, '53': 0, '54': 0, '55': 0}

    if action == 'up':

        if isUpEdge(current_state) is False and isUpWall(current_state) is False:
            next_state[str(current_state[0] + 1) + str(current_state[1])] = 0.8
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.8

        if isLeftEdge(current_state) is False and isLeftWall(current_state) is False:
            next_state[str(current_state[0]) + str(current_state[1] - 1)] = 0.1
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.1

        if isRightEdge(current_state) is False and isRightWall(current_state) is False:
            next_state[str(current_state[0]) + str(current_state[1] + 1)] = 0.1
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.1

    if action == 'right':

        if isRightEdge(current_state) is False and isRightWall(current_state) is False:
            next_state[str(current_state[0]) + str(current_state[1] + 1)] = 0.8
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.8

        if isUpEdge(current_state) is False and isUpWall(current_state) is False:
            next_state[str(current_state[0] + 1) + str(current_state[1])] = 0.1
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.1

        if isDownEdge(current_state) is False and isDownWall(current_state) is False:
            next_state[str(current_state[0] - 1) + str(current_state[1])] = 0.1
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.1

    if action == 'down':
        if isDownEdge(current_state) is False and isDownWall(current_state) is False:
            next_state[str(current_state[0] - 1) + str(current_state[1])] = 0.8
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.8

        if isLeftEdge(current_state) is False and isLeftWall(current_state) is False:
            next_state[str(current_state[0]) + str(current_state[1] - 1)] = 0.1
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.1

        if isRightEdge(current_state) is False and isRightWall(current_state) is False:
            next_state[str(current_state[0]) + str(current_state[1] + 1)] = 0.1
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.1

    if action == 'left':
        if isLeftEdge(current_state) is False and isLeftWall(current_state) is False:
            next_state[str(current_state[0]) + str(current_state[1] - 1)] = 0.8
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.8

        if isUpEdge(current_state) is False and isUpWall(current_state) is False:
            next_state[str(current_state[0] + 1) + str(current_state[1])] = 0.1
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.1

        if isDownEdge(current_state) is False and isDownWall(current_state) is False:
            next_state[str(current_state[0] - 1) + str(current_state[1])] = 0.1
        else:
            next_state[str(current_state[0]) + str(current_state[1])] = 0.1

    return next_state


def initialiseEnvironment():
    environment = (
        state, reward_function, value, q_table, value_change, policy_change, value_over_time_vi, value_over_time_pi,
        actions, gamma, delta)
    return environment
