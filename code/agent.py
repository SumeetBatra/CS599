import random
import numpy as np
from multi_armed_bandits import *
from typing import Dict

"""
 Base class of an autonomously acting and learning agent.
"""
class Agent:

    def __init__(self, params):
        self.nr_actions = params["nr_actions"]

    """
     Behavioral strategy of the agent. Maps states to actions.
    """
    def policy(self, state):
        pass

    """
     Learning method of the agent. Integrates experience into
     the agent's current knowledge.
    """
    def update(self, state, action, reward, next_state, terminated, truncated):
        pass
        

"""
 Randomly acting agent.
"""
class RandomAgent(Agent):

    def __init__(self, params):
        super(RandomAgent, self).__init__(params)
        
    def policy(self, state):
        return random.choice(range(self.nr_actions))

"""
 Autonomous agent base for learning Q-values.
"""
class TemporalDifferenceLearningAgent(Agent):
    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = {}
        self.alpha = params["alpha"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon = 1.0
        
    def Q(self, state):
        state = np.array2string(state)
        if state not in self.Q_values:
            self.Q_values[state] = np.zeros(self.nr_actions)
        return self.Q_values[state]

    def policy(self, state):
        Q_values = self.Q(state)
        return epsilon_greedy(Q_values, None, epsilon=self.epsilon)
    
    def decay_exploration(self):
        self.epsilon = max(self.epsilon-self.epsilon_decay, self.epsilon_decay)

"""
 Autonomous agent using on-policy SARSA.
"""
class SARSALearner(TemporalDifferenceLearningAgent):
        
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            next_action = self.policy(next_state)
            Q_next = self.Q(next_state)[next_action]
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error
        
"""
 Autonomous agent using off-policy Q-Learning.
"""
class QLearner(TemporalDifferenceLearningAgent):
        
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error


class TransitionBuffer:
    def __init__(self, episode_length: int, params: Dict):
        self.idx = 0
        self.state_shape = params['state_shape']
        self.action_dim = params['nr_actions']
        self.episode_length = episode_length
        self.state_buffer = np.zeros((episode_length,) + self.state_shape)
        self.action_buffer = np.zeros((episode_length, self.action_dim))
        self.next_state_buffer = np.zeros((episode_length,) + self.state_shape)
        self.reward_buffer = np.zeros((episode_length, 1))
        self.terminated_buffer = np.zeros((self.episode_length, 1))
        self.truncated_buffer = np.zeros((self.episode_length, 1))
        self.values_buffer = np.zeros((self.episode_length, 1))

    def add_transition(self, state, action, reward, next_state, terminated, trunc, value):
        self.state_buffer[self.idx] = state
        self.action_buffer[self.idx] = action
        self.next_state_buffer[self.idx] = next_state
        self.reward_buffer[self.idx] = reward
        self.terminated_buffer[self.idx] = terminated
        self.truncated_buffer[self.idx] = trunc
        self.values_buffer[self.idx] = value
        self.idx += 1

    def __len__(self):
        return self.idx

    def reset(self):
        self.state_buffer = np.zeros((self.episode_length,) + self.state_shape)
        self.action_buffer = np.zeros((self.episode_length, 1))
        self.next_state_buffer = np.zeros((self.episode_length,) + self.state_shape)
        self.reward_buffer = np.zeros((self.episode_length, 1))
        self.terminated_buffer = np.zeros((self.episode_length, 1))
        self.truncated_buffer = np.zeros((self.episode_length, 1))
        self.values_buffer = np.zeros((self.episode_length, 1))
        self.idx = 0


class AdvancedQLearner(QLearner):
    def __init__(self, params):
        super(AdvancedQLearner, self).__init__(params)
        self.action_counts = {}
        self.explore_constant = params['explore_constant']
        self.ep_len = params['episode_length']
        self.state_shape = params['state_shape']
        self.lambda_ = params['lambda']
        self.use_td_lambda = params['use_td_lambda']
        # only used for n-step returns
        self._transition_buffer = TransitionBuffer(self.ep_len, params)
        self.exploration_strategy = params['exploration_strategy']

    def add_transition(self, state, action, reward, next_state, terminated, trunc):
        q_value = self.Q(state)[action]
        self._transition_buffer.add_transition(state, action, reward, next_state, terminated, trunc, q_value)

    def compute_td_targets(self, next_state):
        # compute td(lambda) return targets
        episode_length = len(self._transition_buffer)
        values = self._transition_buffer.values_buffer[:episode_length]
        next_value = max(self.Q(next_state))
        values = np.concatenate((values, np.array(next_value).reshape(1, -1)), axis=0)
        rewards = self._transition_buffer.reward_buffer[:episode_length]
        dones = self._transition_buffer.terminated_buffer[:episode_length]

        deltas = rewards + (1 - dones) * (self.gamma * values[1:])
        cumulative = 0
        returns = np.zeros_like(deltas)
        i = len(deltas) - 1
        while i >= 0:
            cumulative = deltas[i] + (self.lambda_ * self.gamma) * cumulative * (1.0 - dones[i])
            returns[i] = cumulative
            i -= 1

        return returns

    def add_experience(self, state, action, reward, next_state, terminated, truncated):
        q_val = self.Q(state)[action]

        self.update_visits(state, action)

        self._transition_buffer.add_transition(state, action, reward, next_state, terminated, truncated, q_val)

    def update_td_lambda(self, next_state):
        episode_length = len(self._transition_buffer)
        TD_targets = self.compute_td_targets(next_state)
        errors = TD_targets - self._transition_buffer.values_buffer[:episode_length]
        for i in range(episode_length):
            state, action = self._transition_buffer.state_buffer[i], self._transition_buffer.action_buffer[i]
            self.Q(state)[int(action)] += self.alpha * errors[i].flatten()

    def reset_buffer(self):
        self._transition_buffer.reset()

    def policy(self, state):
        Q_values = self.Q(state)
        if self.exploration_strategy == 'ucb':
            action_counts = self.visit_count(state)
            action = UCB1(Q_values, action_counts, exploration_constant=self.explore_constant)
        else:
            action = QLearner.policy(self, state)
        return action

    def visit_count(self, state):
        # For UCB1
        state = np.array2string(state)
        if state not in self.action_counts:
            self.action_counts[state] = np.zeros(self.nr_actions)
        return self.action_counts[state]

    def update_visits(self, state, action):
        # For UCB1
        state = np.array2string(state)
        if state not in self.action_counts:
            self.action_counts[state] = np.zeros(self.nr_actions)
        self.action_counts[state][action] += 1
