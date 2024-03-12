import numpy as np

from rooms import (ROOMS_ACTIONS,
                   MOVE_EAST,
                   MOVE_WEST,
                   MOVE_NORTH,
                   MOVE_SOUTH)
from copy import deepcopy
from typing import Tuple

MAX_REPEATS = 0


class Node:
    def __init__(self,
                 state: np.ndarray,
                 agent_position: Tuple[int, int],
                 action_space,
                 parent=None,
                 parent_action=None,
                 is_terminal: bool = False,
                 is_truncated: bool = False):
        self.state = state
        self.agent_position = agent_position
        self.action_space = action_space
        self.parent = parent
        self.parent_action = parent_action
        self._untried_actions = list(range(action_space.n))
        self.children = {}
        self.terminal = is_terminal
        self.truncated = is_truncated
        self.repeats = 0
        self.n_visits = 1
        self.q = 0
        self.avg_q = 0

    def __eq__(self, other):
        return np.allclose(self.state, other.state)

    def expand(self, env):
        # if self.is_terminal():
        #     return self, self.q, self.terminal, self.truncated
        if self.is_leaf():
            action = self._untried_actions.pop()
            next_state, reward, terminated, truncated, info = env.step(action)
            child_node = Node(state=next_state,
                              agent_position=env.agent_position,
                              action_space=self.action_space,
                              parent=self.agent_position + (self.repeats,),
                              parent_action=action,
                              is_terminal=terminated,
                              is_truncated=truncated)

            if child_node == self:
                child_node.repeats += 1
                if child_node.repeats > MAX_REPEATS:
                    return self, float(
                        self.agent_position == env.goal_position), False, self.agent_position == env.goal_position

            self.children[action] = env.agent_position + (child_node.repeats,)
        else:
            return self, float(self.agent_position == env.goal_position), False, self.agent_position == env.goal_position
        return child_node, reward, terminated, truncated

    def is_leaf(self):
        return len(self._untried_actions) > 0

    def is_terminal(self):
        return self.terminal or self.truncated

    def uct(self, all_states):
        parent_n_visits = all_states[self.parent].n_visits if self.parent is not None else 1
        return self.q / self.n_visits + np.sqrt(2) * np.sqrt(np.log(parent_n_visits) / self.n_visits)


class MCTS:
    def __init__(self, root: Node, action_space, discount: float = 0.99):
        self.root = root
        self.discount = discount
        self.current_node = root
        self.action_space = action_space
        self.all_states = {root.agent_position + (0,): root}

    def select(self, env):
        current = self.current_node
        depth = 0
        done = False
        while not current.is_leaf() and not done:
            best_action = self.get_action(current)
            # print(f'Pre-step: Node pos: {current.agent_position}, env pos: {env.agent_position}')
            _, _, terminated, truncated, _ = env.step(best_action)
            next_node = self.all_states[current.children[best_action]]
            next_node.parent = current.agent_position + (current.repeats,)
            current = next_node
            # print(f'Node pos: {current.agent_position}, env pos: {env.agent_position}')
            assert current.agent_position == env.agent_position
            depth += 1
            done = terminated or truncated
        # if depth == 100:
        #     print('triggered termination criteria b/c depth = 100')
        # print(f'{depth=}')
        return current, done

    def get_action(self, state: Node, explore: bool = True):
        maxval = -1e9
        action = None
        for k, v in state.children.items():
            value = self.all_states[v].uct(self.all_states) if explore else self.all_states[v].avg_q
            if value > maxval:
                maxval = value
                action = k
            if value == maxval:
                action = np.random.choice([action, k])
        return action

    def expand(self, node: Node, env):
        # node = self.all_states.setdefault(node.agent_position, node)
        child_node, reward, terminated, truncated = node.expand(env)
        self.add(child_node, node)
        return child_node, reward, terminated, truncated

    def backpropagate(self, state: Node, reward: float):
        returns = reward
        i = 0
        while state is not None:
            state.n_visits += 1
            state.q = returns
            state.avg_q = state.q / state.n_visits
            # state.q = state.q + (returns - state.q) / state.n_visits
            # state.q = (state.q + returns) / state.n_visits
            # state.q += returns / state.n_visits
            returns = self.discount * returns
            if state == self.root:
                break

            if state.parent is None:
                state = None
            else:
                if isinstance(state.parent, tuple):
                    state = self.all_states[state.parent]
                elif isinstance(state.parent, Node):
                    state = state.parent
            i += 1

    def policy(self, state: np.ndarray):
        best_action = self.get_action(self.current_node, explore=False)
        self.current_node = self.all_states[self.current_node.children[best_action]]

        return best_action

    def reset(self, root_pos: Tuple[int, int], state: np.ndarray):
        # self.root = self.all_states.setdefault(root_pos,
        #                                        Node(state=state,
        #                                             agent_position=root_pos,
        #                                             parent=None,
        #                                             action_space=self.action_space))
        if root_pos + (0,) not in self.all_states:
            print(f'{root_pos=} not found. Setting for first time')
            self.all_states[root_pos + (0,)] = Node(state=state,
                                                    agent_position=root_pos,
                                                    parent=None,
                                                    action_space=self.action_space)
        self.root = self.all_states[root_pos + (0,)]
        self.current_node = self.root

    def update(self, state, action, reward, next_state, terminated, truncated):
        pass

    def add(self, node: Node, parent: Node):
        if node.agent_position + (node.repeats,) not in self.all_states:
            print(f'{node.agent_position=} not found. Setting for first time')
            self.all_states[node.agent_position + (node.repeats,)] = node

        node = self.all_states[node.agent_position + (node.repeats,)]
        node.parent = parent.agent_position + (parent.repeats,) if node != parent else None
