import gym
import numpy as np


from utils import *

NO_BINS = [6, 12]
LIMITS = [
    [-4.19, -4.19],
    [-np.radians(50), -np.radians(50)]
]

def train_model():
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset()

    Q_table = np.zeros((NO_BINS[0], NO_BINS[1], 2))

    for episode in range(10000):
        obs, info = env.reset()
        current_state = discretize_state(obs[2:], LIMITS, NO_BINS)

        done = False
        t = 0
        score = 0
        while not done:
            env.render()

            if np.random.random() < exploration_rate(episode):
                action = get_opt_action(Q_table, current_state)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            new_state = discretize_state(obs[2:], LIMITS, NO_BINS)

            lr = get_l_rate(episode)
            Q_table = update_q_table(Q_table, new_state, current_state, action, reward, lr)

            t += 1
            score += reward
            current_state = new_state
            if done:
                print(f"done in {t} steps - episode no {episode} - score: {score}")
                break

    np.save("Q_table.npy", Q_table)
    env.close()
    return

def use_trained_model():
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset()

    Q_table = np.load("Q_table.npy")

    for i in range(10):
        obs, info = env.reset()
        state = discretize_state(obs[2:], LIMITS, NO_BINS)

        done = False
        while not done:
            env.render()

            action = get_opt_action(Q_table, state)

            obs, reward, done, truncated, info = env.step(action)
            state = discretize_state(obs[2:], LIMITS, NO_BINS)

    env.close()
    return


if __name__ == "__main__":
    try:
        use_trained_model()
    except FileNotFoundError:
        train_model()
