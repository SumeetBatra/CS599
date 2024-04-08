import matplotlib.pyplot as plt
import argparse
import rooms
import agent as a
import matplotlib.pyplot as plot
import sys
import numpy as np
import seaborn as sns
import pickle
import pandas as pd

from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rooms_instance', type=str)
    parser.add_argument('--exploration_strategy', type=str, choices=['greedy', 'ucb'])
    parser.add_argument('--use_td_lambda', action='store_true')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--train_steps', type=int, default=800)
    args = parser.parse_args()
    return args


def episode(env, agent, nr_episode=0, train: bool = True):
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

        if train:
            if agent.use_td_lambda:
                # store transition to compute n-step return target at the end
                agent.add_transition(state, action, reward, next_state, terminated, truncated)
            else:
                # 1-step return, so we can update the q-function right away
                agent.update(state, action, reward, next_state, terminated, truncated)
            agent.update_visits(state, action)

        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1

    if train and agent.use_td_lambda:
        # compute n-step return and update Q values
        agent.update_td_lambda(state)
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

    axs[0].set_title(f'Evaluation Medium 0')
    axs[1].set_title(f'Evaluation Medium 1')
    plot.show()


def main():
    params = {}
    args = parse_args()
    rooms_instance = args.rooms_instance
    env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
    params["nr_actions"] = env.action_space.n
    params['state_shape'] = env.observation_space.shape
    params["gamma"] = 0.99
    params["lambda"] = 0.8
    params["epsilon_decay"] = 0.01
    params["alpha"] = args.lr
    params["env"] = env
    params['episode_length'] = env.time_limit
    params["explore_constant"] = np.sqrt(2)
    params["use_td_lambda"] = args.use_td_lambda
    params["exploration_strategy"] = args.exploration_strategy

    agent = a.AdvancedQLearner(params)

    training_episodes = args.train_steps
    test_episodes = 100
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

    with open(f'agent_{rooms_instance}_{args.exploration_strategy}_use_td_lambda_{args.use_td_lambda}_{training_episodes}.pkl', 'wb') as f:
        pickle.dump(agent, f)


if __name__ == '__main__':
    hard0_data = evaluate('agent_hard_0_ucb_use_td_lambda_True_800.pkl', 'hard_0')
    hard1_data = evaluate('agent_hard_1_ucb_use_td_lambda_True_800.pkl', 'hard_0')
    plot_results(hard0_data, hard1_data)
    # main()