import matplotlib
import matplotlib.pyplot as plt
import argparse
import rooms
import agent as a
import matplotlib.pyplot as plot
import numpy as np
import seaborn as sns
import pickle
import pandas as pd

from tqdm import tqdm

matplotlib.rcParams.update({'font.size': 20})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rooms_instance', type=str)
    parser.add_argument('--exploration_strategy', type=str, choices=['greedy', 'ucb'])
    parser.add_argument('--use_td_lambda', action='store_true')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--train_steps', type=int, default=500)
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
    if train:
        print(nr_episode, ":", discounted_return)
    return discounted_return


def evaluate(agent_fp: str, rooms_instance: str, n_trials: int = 20, n_eval_episodes: int = 10):
    env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")

    with open(agent_fp, 'rb') as f:
        agent = pickle.load(f)
    agent.explore_constant = 0
    agent.epsilon = 0
    all_returns = []
    for _ in range(n_trials):
        returns = []
        for _ in range(n_eval_episodes):
            discounted_return = episode(env, agent, train=False)
            returns.append(discounted_return)
        all_returns.append(np.array(returns))

    all_returns = np.vstack(all_returns)
    df = pd.DataFrame(all_returns).melt(var_name='Episode', value_name='Return')
    return df


def plot_all_results():
    '''
    Plot the results of all trained agents on all maps
    '''
    results = {
        'easy_0': 'agent_easy_0_ucb_use_td_lambda_False_100.pkl',
        'easy_1': 'agent_easy_1_ucb_use_td_lambda_False_100.pkl',
        'medium_0': 'agent_medium_0_ucb_use_td_lambda_False_300.pkl',
        'medium_1': 'agent_medium_1_ucb_use_td_lambda_False_300.pkl',
        'hard_0': 'agent_hard_0_ucb_use_td_lambda_False_500.pkl',
        'hard_1': 'agent_hard_1_ucb_use_td_lambda_False_500.pkl',
    }

    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    axs = axs.flatten()

    for i, (map, agent_fp) in tqdm(enumerate(results.items()), total=len(results)):
        df = evaluate(agent_fp, map)
        sns.lineplot(data=df, x='Episode', y='Return', errorbar=('ci', 95), ax=axs[i])
        axs[i].set_title(map.upper())

    fig.tight_layout()
    plt.show()


def ablation():
    agent_fps = {
        'greedy_1step': 'ablation/agent_hard_0_greedy_use_td_lambda_False_800.pkl',
        'ucb_1step': 'ablation/agent_hard_0_ucb_use_td_lambda_False_800.pkl',
        'greedy_nstep': 'ablation/agent_hard_0_greedy_use_td_lambda_True_800.pkl',
        'ucb_nstep': 'ablation/agent_hard_0_ucb_use_td_lambda_True_800.pkl',
    }

    names, all_data = [], []
    for name, fp in tqdm(agent_fps.items(), total=len(agent_fps)):
        eval_data = evaluate(agent_fp=fp, rooms_instance='hard_0', n_trials=20)
        all_data.append(eval_data)
        names.append(name)

    # plot the results in a 2x2
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs = axs.flatten()
    for i, (name, df) in tqdm(enumerate(zip(names, all_data))):
        sns.lineplot(df, x='Episode', y='Return', ax=axs[i])
        axs[i].set_title(name.upper())
    fig.tight_layout()
    plt.show()


def main():
    params = {}
    args = parse_args()
    rooms_instance = args.rooms_instance
    env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
    params["nr_actions"] = env.action_space.n
    params['state_shape'] = env.observation_space.shape
    params["gamma"] = 0.90
    # lambda if using td-lambda return targets
    params["lambda"] = 0.8
    params["epsilon_decay"] = 0.01
    params["alpha"] = args.lr
    params["env"] = env
    params['episode_length'] = env.time_limit
    # exploration constant for ucb1
    params["explore_constant"] = np.sqrt(2)
    params["use_td_lambda"] = args.use_td_lambda
    params["exploration_strategy"] = args.exploration_strategy

    # class defining our method. Allows for td(lambda) return targets if --use_td_lambda is passed as a cmd line argument
    agent = a.AdvancedQLearner(params)

    training_episodes = args.train_steps
    train_returns = [episode(env, agent, i) for i in range(training_episodes)]

    test_episodes = 100
    # disable exploration for evaluation
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
    # uncomment this to plot all results.
    # plot_all_results()

    # uncomment this to run a main experiment using the cmd line args from the argumentparser
    main()

    # to run the ablation results presented in intermediate report 2, please uncomment this line.
    # ablation()