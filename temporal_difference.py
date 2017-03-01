import numpy as np
import environment
import mrp


class TDAgent:
    def __init__(self, epsilon=.01, alpha=.9, gamma=.9):
        self.state_data = {}    # contains states and action values respectively
        self.epsilon = epsilon  # percentage rate at which the agent takes random actions [0, 100]
        self.alpha = alpha  # q value rate of change factor (0, 1]
        self.gamma = gamma  # reward discount factor (0, 1]
        self.previous_data = {}  # contains previous state visited and previous action taken
        self.is_first_turn = True
        self.mean_episode_length = 0
        self.n_episodes = 0  # number of episodes taken
        self.play_counter = 0   # number of plays in current episode

    # where the magic happens
    def state_action_map(self, state, reward, n_actions, termination_flag=False):
        self._initialize_state(state, n_actions)
        self.play_counter += 1
        if self.is_first_turn:
            self.n_episodes += 1
            self.play_counter = 0
            self.is_first_turn = False
            action = self.get_action(state)
            self.previous_data = {'state': state, 'action': action}
            return action
        elif not termination_flag:
            self._calculate_q(state, reward)
            next_action = self.get_action(state)
            self.previous_data = {'state': state, 'action': next_action}
            return next_action
        else:
            self.update_mean_episode(self.play_counter)
            self._calculate_q(state, reward)
            self.is_first_turn = True

    def get_action(self, state):
        if self.epsilon/100 > np.random.rand():  # return epsilon random action
            if len(self.state_data[state]) != 0:   # if state has actions
                return np.random.randint(0, len(self.state_data[state]))
        elif len(self.state_data[state]) > 0:  # return best action
            return np.argmax(self.state_data[state])
        return 0

    # calculate mean actions taken per episode
    def update_mean_episode(self, episode_length):
        self.mean_episode_length += 1/self.n_episodes*(episode_length - self.mean_episode_length)

    # calculate q value of previously taken action based on current states q_max
    def _calculate_q(self, current_state, reward):

        if len(self.state_data[current_state]) != 0:
            s_q_max = np.max(self.state_data[current_state])
        else:
            s_q_max = 0
        prev_state = self.previous_data['state']
        action = self.previous_data['action']
        td_target = reward + self.gamma*s_q_max
        self.state_data[prev_state][action] += self.alpha*(td_target-self.state_data[prev_state][action])

    # sets q values of unvisited states to 0
    def _initialize_state(self, state, n_actions):
        if state not in self.state_data:
            self.state_data[state] = [0 for _ in range(n_actions)]


##################################################################################


def History(n_episodes=1000):
    model = mrp.MRP(mrp.RNode([[1, 2, 3], [], []]))
    # set all rewards to 0
    model.reward_matrix.fill(-1)
    # set rewards
    model.nodes[7].reward = 1
    model.nodes[18].reward = 1
    model.nodes[59].reward = 1
    model.nodes[7].neighbours = []
    model.nodes[18].neighbours = []
    model.nodes[59].neighbours = []

    agent = TDAgent()
    env = environment.Environment(model)
    history = []


    for i in range(n_episodes):
        while True:
            rnd_state = np.random.randint(0, len(model.nodes))
            if len(model.nodes[rnd_state].neighbours) != 0:
                break
        env.take_sample(rnd_state, agent.state_action_map)
        history.append((i, agent.play_counter))
        #print('Episode:' + str(agent.n_episodes) + ', Episode Length Mean: ' + str(agent.mean_episode_length))


    return (np.round(agent.mean_episode_length, 2), history)


if __name__ == '__main__':
    model = mrp.MRP(mrp.RNode([[1, 2, 3], [], []]))
    # set all rewards to 0
    model.reward_matrix.fill(-1)
    # set rewards
    model.nodes[7].reward = 1
    model.nodes[18].reward = 1
    model.nodes[59].reward = 1
    model.nodes[7].neighbours = []
    model.nodes[18].neighbours = []
    model.nodes[59].neighbours = []

    agent = TDAgent()
    env = environment.Environment(model)
    history = []

    for i in range(1000):
        # draw a random start state != termination state
        while True:
            rnd_state = np.random.randint(0, len(model.nodes))
            if len(model.nodes[rnd_state].neighbours) != 0:
                break
        # sample environment
        env.take_sample(rnd_state, agent.state_action_map)
        # append sample to history
        history.append((i, agent.play_counter))
        print('Episode:' + str(agent.n_episodes) + ', Episode Length Mean: ' + str(agent.mean_episode_length))
