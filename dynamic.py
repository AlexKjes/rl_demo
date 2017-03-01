import mrp
import numpy as np

gamma = 1   # aka reward discount factor


# calculates rewards for entire state space
def value_iteration(state_space):
    for s in state_space:
        s.value = s.reward + gamma*(neighbour_value_m(s))


# calculates state reward from neighbour average
def neighbour_value_e(state):
    ret = 0
    n_neighbours = len(state.neighbours)
    for n in state.neighbours:
        ret += n.value * (1/n_neighbours)
    return ret


# calculates state reward from neighbour max
def neighbour_value_m(state):
    n_values = [n.value for n in state.neighbours]
    if len(n_values) > 0:
        v_max = np.argmax(n_values)
        return state.neighbours[v_max].value
    return 0


# initialize model
model = mrp.MRP(mrp.RNode([[1, 2, 3], [], []]))
# set all rewards to -1
model.reward_matrix.fill(-1)
# set termination states
model.nodes[7].reward = 1
model.nodes[18].reward = 1
model.nodes[59].reward = 1
model.nodes[7].neighbours = []
model.nodes[18].neighbours = []
model.nodes[59].neighbours = []


for i in range(1000):
    print('----------------------------------------------------------------------------------------------------')
    value_iteration(model.nodes)
    model.print_nodes(10)   # print state space to console
    input()  # comment out to do all iterations at once



#model.print_nodes(10)
