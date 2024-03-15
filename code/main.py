import matplotlib.pyplot as plt

import rooms
import agent as a
import matplotlib.pyplot as plot
import sys
import numpy as np
import seaborn as sns
import pickle
import pandas as pd

from typing import List


def episode(env, agent, nr_episode=0, train:bool = True):
    state = env.reset()
    agent.reset_buffer()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # 3. Integrate new experience into agent
        # agent.update(state, action, reward, next_state, terminated, truncated)
        if train:
            agent.update_visits(state, action)
            agent.add_transition(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    if train:
        agent.update(state)
    print(nr_episode, ":", discounted_return)
    return discounted_return


def evaluate(agent_fp: str, rooms_instance: str, n_trials: int = 20, n_eval_episodes: int = 10):
    params = {}
    env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
    params["nr_actions"] = env.action_space.n
    params['state_shape'] = env.observation_space.shape
    params["gamma"] = 0.99
    params["lambda"] = 0.8
    params["epsilon_decay"] = 0.01
    params["alpha"] = 0.001
    params["env"] = env
    params['episode_length'] = env.time_limit
    params["explore_constant"] = np.sqrt(2)

    with open(agent_fp, 'rb') as f:
        agent = pickle.load(f)
    agent.explore_constant = 0
    agent.epsilon = 0
    all_returns = []
    for _ in range(n_trials):
        returns = []
        for _ in range(n_eval_episodes):
            env.seed()
            discounted_return = episode(env, agent, train=False)
            returns.append(discounted_return)
        all_returns.append(np.array(returns))

    all_returns = np.vstack(all_returns)
    df = pd.DataFrame(all_returns).melt(var_name='Episode', value_name='Return')
    return df


def plot_results(easy_data: pd.DataFrame, hard_data: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sns.lineplot(easy_data, x='Episode', y='Return', errorbar=('ci', 95), ax=axs[0])
    sns.lineplot(hard_data, x='Episode', y='Return', errorbar=('ci', 95), ax=axs[1])

    axs[0].set_title(f'Evaluation Easy 0')
    axs[1].set_title(f'Evaluation Hard 0')
    plot.show()


def main():
    params = {}
    rooms_instance = sys.argv[1]
    env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
    params["nr_actions"] = env.action_space.n
    params['state_shape'] = env.observation_space.shape
    params["gamma"] = 0.99
    params["lambda"] = 0.8
    params["epsilon_decay"] = 0.01
    params["alpha"] = 0.001
    params["env"] = env
    params['episode_length'] = env.time_limit
    params["explore_constant"] = np.sqrt(2)

    #agent = a.RandomAgent(params)
    #agent = a.SARSALearner(params)
    # agent = a.QLearner(params)
    agent = a.UCBQLearner(params)
    training_episodes = 1000
    test_episodes = 10
    train_returns = [episode(env, agent, i) for i in range(training_episodes)]
    agent.explore_constant = 0
    agent.epsilon = 0
    test_returns = [episode(env, agent, i, train=False) for i in range(test_episodes)]

    x = range(training_episodes + test_episodes)
    y = train_returns + test_returns

    plot.plot(x,y)
    plot.title("Progress")
    plot.xlabel("Episode")
    plot.ylabel("Discounted Return")
    plot.show()

    env.save_video()

    with open(f'agent_{rooms_instance}_{training_episodes}.pkl', 'wb') as f:
        pickle.dump(agent, f)


if __name__ == '__main__':
    easy_data = evaluate('agent_easy_0_200.pkl', 'easy_0')
    hard_data = evaluate('agent_hard_0_1000.pkl', 'hard_0')
    plot_results(easy_data, hard_data)
    # main()