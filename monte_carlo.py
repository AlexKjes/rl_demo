import numpy as np
import mrp
import environment
import copy


class EveryVisitExploringStart:
    def __init__(self, gamma=0.9):
        self.gamma = gamma  # reward discount factor (0, 1]
        self.state_data = {}  # contains states and action values respectively
        self.current_sample = []  # aggregates episode data
        self.is_first_turn = True
        self.epsilon = 10  # percentage rate at which the agent takes random actions [0, 100]
        self.n_episodes = 0  # number of episodes taken
        self.mean_episode_length = 0
        self.policy_state_data = copy.deepcopy(self.state_data)
        self.last_sample_size = 0   # length of previous episode

    # prepare agent for new episode
    def new_sample(self):
        self.is_first_turn = True
        self.n_episodes += 1
        self.update_mean_episode()
        self._calculate_sample()
        self.last_sample_size = len(self.current_sample)
        self.current_sample = []

    # where the magic happens
    def state_action_map(self, state, reward, n_actions, termination_flag=False):
        self._initialize_state(state, n_actions)
        if self.is_first_turn:
            self.is_first_turn = False
            rand_action = np.random.randint(0, n_actions)   # exploring start, every first action taken is random
            self.current_sample.append({'state': state, 'action': rand_action})
            return rand_action
        elif not termination_flag:
            self.current_sample[-1]['reward'] = reward
            next_action = self.get_action(state)
            self.current_sample.append({'state': state, 'action': next_action})
            return next_action
        else:
            self.current_sample[-1]['reward'] = reward
            self.new_sample()

    def get_action(self, state):
        if self.epsilon/100 > np.random.rand():  # return epsilon random action
            if len(self.state_data[state]) != 0:  # if state has actions
                return np.random.randint(0, len(self.state_data[state]))
        elif len(self.state_data[state]) > 0:
            return self._argmax_or_zero(state)  #take best action
        return 0

    # do policy update
    def update_policy(self):
        self.policy_state_data = copy.deepcopy(self.state_data)

    # add state to state data
    def _initialize_state(self, state, n_actions):
        if state not in self.state_data:
            self.state_data[state] = [[0, 0] for _ in range(n_actions)]

    def _argmax_or_zero(self, state):
        if state in self.policy_state_data:
            action = np.argmax([q[1] for q in self.policy_state_data[state]])
            return action
        return 0

    # updates off policy
    def _calculate_sample(self):
        visited = []
        #print(len(self.current_sample))
        if len(self.current_sample) < 100000:
            for i, t in enumerate(self.current_sample[1:-1]):
                state = t['state']
                action = t['action']
                if (state, action) not in visited:
                    visited.append((state, action))
                    self.state_data[state][action][0] += 1
                    s_return = np.sum([np.power(self.gamma, t) * r['reward'] for t, r in enumerate(self.current_sample[i:])])
                    self.state_data[state][action][1] += 1/self.state_data[state][action][0]*(s_return-self.state_data[state][action][1])

    # calculates average play per episode
    def update_mean_episode(self):
        self.mean_episode_length += 1/self.n_episodes*(len(self.current_sample) - self.mean_episode_length)


# runs the agent through the environment n number of times
# returns stats
def History(n_episodes=1000):
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

    agent = EveryVisitExploringStart()
    env = environment.Environment(model)
    history = []

    for i in range(n_episodes):
        while True:
            rnd_state = np.random.randint(0, len(model.nodes))
            if len(model.nodes[rnd_state].neighbours) != 0:
                break
        if agent.n_episodes % 1 == 0:
            agent.update_policy()
        env.take_sample(rnd_state, agent.state_action_map)
        history.append((i, agent.last_sample_size))

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

    agent = EveryVisitExploringStart()
    env = environment.Environment(model)
    history = []

    for i in range(1000):
        while True:
            rnd_state = np.random.randint(0, len(model.nodes))
            if len(model.nodes[rnd_state].neighbours) != 0:
                break
        if agent.n_episodes % 1 == 0:
            agent.update_policy()
        env.take_sample(rnd_state, agent.state_action_map)
        history.append((i, agent.last_sample_size))
        print('Episode:' + str(agent.n_episodes) + ', Episode Length Mean: ' + str(agent.mean_episode_length))


