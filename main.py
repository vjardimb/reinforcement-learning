import gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

for episode in range(10):
    obs = env.reset()
    done = False
    t = 0
    score = 0
    while not done:
        env.render()

        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        t += 1
        score += reward
        if done:
            print(f"done in {t} steps - episode no {episode} - score: {score}")
            break

env.close()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
