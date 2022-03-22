import matplotlib.pyplot as plt
import seaborn as sns


def stringToInt(s):
    state_int = int(s)
    num1 = int(state_int / 10)
    num2 = state_int % 10
    return [num1, num2]


def intToString(s):
    our_state = str(s[0]) + str(s[1])
    return our_state


def printEachIteration(current_state, value, actions, q_table):
    print('State:', current_state)
    print('Value:', value[current_state[0]][current_state[1]])
    print('Optimal Action:', actions[int(q_table[current_state[0]][current_state[1]])])
    print('\n')


def valueOverTime(value_over_time, value):
    for i in range(6):
        for j in range(6):
            cur_state = intToString([i, j])
            value_over_time[cur_state].append(value[i][j])
    return 0


def valueConverged(value_change, delta):
    for i in range(6):
        for j in range(6):
            if value_change[i][j] > delta:
                return False

    return True


def policyConverged(policy_change):
    for i in range(6):
        for j in range(6):
            if policy_change[i][j] == 1:
                return False

    return True


def plotOptimalPolicy(state, actions, q_table):
    for i in range(6):
        for j in range(6):
            current_state = state[5 - i][j - 6]
            if current_state[0] == 5 and current_state[1] == 1 or current_state[0] == 4 and current_state[1] == 4 or \
                    current_state[0] == 1 and current_state[1] == 1 or current_state[0] == 1 and current_state[
                1] == 2 or current_state[0] == 1 and current_state[1] == 3:
                print('W', end=' | ')
            else:
                print(actions[int(q_table[current_state[0]][current_state[1]])][0], end=" | ")
        print('\n')


def plotOptimalValue(state, value):
    for i in range(6):
        for j in range(6):
            current_state = state[5 - i][j - 6]
            rounded_value = round(value[current_state[0]][current_state[1]], 2)
            print(rounded_value, end=' | ')

        print('\n')


def printOptimalValue():
    for i in range(6):
        for j in range(6):
            if i == 5 and j == 1 or i == 4 and j == 4 or i == 1 and j == 1 or i == 1 and j == 2 or i == 1 and j == 3:
                pass
            else:
                current_state = state[i][j]
                rounded_value = round(value[current_state[0]][current_state[1]], 2)
                print(current_state, rounded_value)


def plotValueOverTime(value_over_time, policy):
    fig, ax = plt.subplots()
    # print(value_over_time['21'])

    for i in range(6):
        for j in range(6):
            if i == 5 and j == 1 or i == 4 and j == 4 or i == 1 and j == 1 or i == 1 and j == 2 or i == 1 and j == 3:
                pass
            else:
                ax = sns.lineplot(data=value_over_time[str(str(i) + str(j))])

    ax.set(xlabel='Iteration', ylabel='Value', title=policy)
    plt.legend(labels=['1, 1', '1, 2', '1, 3', '1, 4', '1, 5', '1, 6', '2, 1', '2, 5', '2, 6',
                       '3, 1', '3, 2', '3, 3', '3, 4', '3, 5', '3, 6', '4, 1', '4, 2', '4, 3', '4, 4', '4, 5', '4, 6',
                       '5, 1', '5, 2', '5, 3', '5, 4', '5, 6', '6, 1', '6, 3', '6, 4', '6, 5', '6, 6'],
               fontsize='small',
               title="State", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    pass
