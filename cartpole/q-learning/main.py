import gym
import numpy as np

from utils import *


def train_model():
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset()

    NO_BINS = [1, 1, 6, 3]
    LIMITS = list(zip(env.observation_space.low, env.observation_space.high))
    LIMITS[1] = (-0.5, 0.5)
    LIMITS[3] = (-math.radians(50), math.radians(50))


    no_episodes = 180

    Q_table = np.zeros(NO_BINS + [2])

    for episode in range(no_episodes):
        obs, info = env.reset()
        current_state = discretize_state(obs, LIMITS, NO_BINS)
        print(current_state)

        done = False
        t = 0
        score = 0
        while not done:
            env.render()

            if np.random.random() > exploration_rate(episode):
                action = get_opt_action(Q_table, current_state)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            new_state = discretize_state(obs, LIMITS, NO_BINS)

            lr = get_l_rate(episode)
            Q_table = update_q_table(Q_table, new_state, current_state, action, reward, lr)

            t += 1
            score += reward
            current_state = new_state

            if done:
                print(f"Episode no {episode} - Score: {score}")

    np.save("Q_table.npy", Q_table)
    env.close()
    return


def use_trained_model():
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset()

    NO_BINS = [1, 1, 6, 3]
    LIMITS = list(zip(env.observation_space.low, env.observation_space.high))
    LIMITS[1] = (-0.5, 0.5)
    LIMITS[3] = (-math.radians(50), math.radians(50))

    Q_table = np.load("Q_table.npy")

    for i in range(10):
        obs, info = env.reset()
        state = discretize_state(obs, LIMITS, NO_BINS)

        done = False
        while not done:
            env.render()

            action = get_opt_action(Q_table, state)

            obs, reward, done, truncated, info = env.step(action)
            state = discretize_state(obs, LIMITS, NO_BINS)

    env.close()
    return


if __name__ == "__main__":
    # train_model()
    try:
        use_trained_model()
    except FileNotFoundError:
        train_model()
