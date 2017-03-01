import numpy as np
import copy as cpy


class Environment:
    def __init__(self, n_stacks, n_boxes):
        self.nStacks = n_stacks
        self.nBoxes = n_boxes
        self.stacks = [[]] * n_stacks
        self.prob_matrix = np.zeros((n_stacks, n_stacks))
        self.states = {}
        self._generate_probability_matrix_and_state_space()

    def _generate_probability_matrix_and_state_space(self):
        for i in range(self.nStacks):
            for j in range(self.nBoxes):
                self.stacks[i] = np.arange(0, self.nBoxes)
                box = self.stacks[i].pop()
                for k in range(self.nStacks):
                   pass


class Node:
    def __init__(self, state, sid=0):
        self.id = sid
        self.state = state
        self.neighbours = []
        self.print_order = [self.state_to_string, self.id_string]

    def find_neighbours(self):
        ret = []
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                tmp = cpy.deepcopy(self.state)
                if len(tmp[i]) != 0:
                    b = tmp[i].pop()
                    tmp[j].append(b)
                    ret.append(tmp)
        self.neighbours = ret
        return ret

    def __str__(self):
        ret = ''
        for i in self.print_order:
            s = i()
            ret += s
            for _ in range(len(self.state)*4 - len(s)):
                ret += ' '
            ret += '\n'
        return ret

    def state_to_string(self):
        ret = ''
        max_size = np.sum(len(i) for i in self.state)
        for i in range(max_size):
            for platform in self.state:
                if len(platform) >= max_size - i:
                    ret += '[' + str(platform[max_size-1 - i]) + '] '
                else:
                    ret += '    '
            ret += '\n'
        for i in range(len(self.state)*2):
            if i < len(self.state):
                ret += '--- '
            if i == len(self.state):
                ret += '\n 1  '
            if i > len(self.state):
                ret += ' ' + str(i-len(self.state)+1) + '  '
        ret += '\n'
        return ret

    def id_string(self):
        return 'Id: ' + str(self.id)


class Graph:
    def __init__(self, start_node):
        self.nodes = [start_node]
        self.find_neighbours()
        self.transition_matrix = self._generate_transition_matrix()
        self.__set_neighbour_references()

    def find_neighbours(self):
        for node in self.nodes:
            self._neighbours_to_states(node.find_neighbours())

    def _neighbours_to_states(self, neighbours):
        for n in neighbours:
            b = False
            for node in self.nodes:
                if n == node.state:
                    b = True
                    break
            if not b:
                self.nodes.append(type(node)(n, sid=len(self.nodes)))

    def _generate_transition_matrix(self):
        ret = np.zeros((len(self.nodes), len(self.nodes)), dtype=int)
        for m in range(len(self.nodes)):
            for n in range(len(self.nodes)):
                b = False
                for a in self.nodes[m].neighbours:
                    if self.nodes[n].state == a:
                        b = True
                if b:
                    ret[m][n] = 1
                else:
                    ret[m][n] = 0
        return ret

    def __set_neighbour_references(self):
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            node.neighbours = []
            for j in range(len(self.transition_matrix[i])):
                if self.transition_matrix[i][j] == 1:
                    node.neighbours.append(self.nodes[j])

    def print_nodes(self, width):
        row = []
        for n in self.nodes:
            row.append(n)
            if width == len(row):
                self.__row_to_string(row)
                row = []
        if len(row) > 0:
            self.__row_to_string(row)

    @staticmethod
    def __row_to_string(row):
        lines = [str.split(str(n), '\n') for n in row]
        for line in range(len(lines[0])):
            ln = ''
            for column in range(len(lines)):
                ln += lines[column][line] + '   '
            print(ln)


#g = Graph(Node([[1, 2, 3], [], []]))
#g.print_nodes(10)