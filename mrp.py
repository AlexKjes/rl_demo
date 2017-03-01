import mp as env
import numpy as np


# reward node
class RNode(env.Node):
    def __init__(self, state, reward=-1, value=0, sid=0):
        super(RNode, self).__init__(state, sid=sid)
        self.value = value
        self.reward = reward
        self.print_order.append(self.reward_string)
        self.print_order.append(self.value_string)

    def value_string(self):
        return 'v(s)=' + str(np.round(self.value, 2))

    def reward_string(self):
        return 'r=' + str(self.reward)


# Markov reward process
class MRP(env.Graph):
    def __init__(self, start_node):
        super(MRP, self).__init__(start_node)
        self.reward_matrix = np.zeros((1, len(self.nodes)))


"""
mrp = MRP(RNode([[1, 2, 3], [], []]))

mrp.reward_matrix[0][7] = 1
mrp.reward_matrix[0][18] = 1
mrp.reward_matrix[0][59] = 1

mrp.print_nodes(10)
"""