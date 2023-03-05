import tensorflow as tf
import numpy as np


def play_one_step(env, obs, model, loss_fn):
	with tf.GradientTape() as tape:
		left_proba = model(obs[np.newaxis])
		action = (tf.random.uniform([1, 1]) > left_proba)
		y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
		loss = tf.reduce_mean(loss_fn(y_target, left_proba))
	grads = tape.gradient(loss, model.trainable_variables)
	obs, reward, done, truncated, info = env.step(int(action[0, 0].numpy()))
	return obs, reward, done, grads


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
	all_rewards = []
	all_grads = []
	for episode in range(n_episodes):
		current_rewards = []
		current_grads = []
		obs, info = env.reset()
		for step in range(n_max_steps):
			obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
			current_rewards.append(reward)
			current_grads.append(grads)
			if done:
				break
		all_rewards.append(current_rewards)
		all_grads.append(current_grads)
	return all_rewards, all_grads


def discount_rewards(rewards, discount_factor):
	discounted = np.array(rewards)
	for step in range(len(rewards) - 2, -1, -1):
		discounted[step] += discounted[step + 1] * discount_factor
	return discounted


def discount_and_normalize_rewards(all_rewards, discount_factor):
	all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
	flat_rewards = np.concatenate(all_discounted_rewards)
	reward_mean = flat_rewards.mean()
	reward_std = flat_rewards.std()
	return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


def show_results(all_rewards):
	all_rewards = np.array(list(map(sum, all_rewards)))
	print(np.mean(all_rewards), np.std(all_rewards), np.min(all_rewards), np.max(all_rewards))

