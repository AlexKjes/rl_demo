

class Environment:
    def __init__(self, model):
        self.model = model
        self.status = '0'

    def take_sample(self, start_state, policy_function):
        current_state = self.model.nodes[start_state]
        self.status = 1
        while self.status == 1:
            if current_state.reward == 1:
                self.status = 0
                policy_function(current_state.id, current_state.reward,
                                len(current_state.neighbours), termination_flag=True)
                break
            action = policy_function(current_state.id, current_state.reward, len(current_state.neighbours))
            current_state = current_state.neighbours[action]

