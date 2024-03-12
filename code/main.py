import rooms
import agent as a
import matplotlib.pyplot as plot
import sys
import numpy as np

from mcts import MCTS, Node


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
        agent.update()
    print(nr_episode, ":", discounted_return)
    return discounted_return


def main():
    params = {}
    rooms_instance = sys.argv[1]
    env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{rooms_instance}.mp4")
    params["nr_actions"] = env.action_space.n
    params['state_shape'] = env.observation_space.shape
    params["gamma"] = 0.99
    params["lambda"] = 0.8
    params["epsilon_decay"] = 0.01
    params["alpha"] = 0.0001
    params["env"] = env
    params['episode_length'] = env.time_limit
    params["explore_constant"] = np.sqrt(2)

    #agent = a.RandomAgent(params)
    #agent = a.SARSALearner(params)
    # agent = a.QLearner(params)
    agent = a.UCBQLearner(params)
    training_episodes = 400
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


if __name__ == '__main__':
    main()