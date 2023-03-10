import tensorflow as tf
from tensorflow import keras
import gym
import numpy as np
from functions import *


def train_model(my_model=None):
	"""
	This function allows you to either train a model from scratch or extend the training of a previously trained model.
	:param my_model: name of the file containing your model (.h5 file)
	:return: None
	"""
	n_iterations = 400
	n_episodes_per_update = 10
	n_max_steps = 300
	discount_factor = 0.95

	optimizer = keras.optimizers.Adam(lr=0.01)
	loss_fn = keras.losses.binary_crossentropy

	env = gym.make("CartPole-v1")
	obs, info = env.reset()

	if my_model == None:
		n_inputs = 4  # == env.observation_space.shape[0]

		model = keras.models.Sequential([
			keras.layers.Dense(10, activation="relu", input_shape=[n_inputs]),
			keras.layers.Dense(5, activation="relu"),
			keras.layers.Dense(1, activation="sigmoid"),
		])
	else:
		root_path = r'C:\Users\Asus\Desktop\prog\reinforcement-learning\cartpole\policy-gradient'
		model = tf.keras.models.load_model(root_path + rf"\{my_model}")

	for iteration in range(n_iterations):
		all_rewards, all_grads = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn)
		print(f"Iteration: {iteration}: ")
		show_results(all_rewards)
		all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
		all_mean_grads = []
		for var_index in range(len(model.trainable_variables)):
			mean_grads = tf.reduce_mean(
				[final_reward * all_grads[episode_index][step][var_index]
				 for episode_index, final_rewards in enumerate(all_final_rewards)
				 for step, final_reward in enumerate(final_rewards)], axis=0)
			all_mean_grads.append(mean_grads)
		optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
		if iteration % 20 == 0:
			print("model_saved")
			model.save("cart_pole" + str(iteration) + ".h5")

	env.close()


def use_my_model(my_model):
	"""
	Show the performance of a trained model, rendering the cartpole env being controlled by the model
	:param my_model: name of the file model
	:return: None
	"""

	root_path = r'C:\Users\Asus\Desktop\prog\reinforcement-learning\cartpole\policy-gradient'
	model = tf.keras.models.load_model(root_path + rf"\{my_model}")

	env = gym.make("CartPole-v1", render_mode="human")

	totals = []
	for episode in range(500):
		episode_rewards = 0
		obs, info = env.reset()
		for step in range(200):
			left_prob = model.predict(obs[np.newaxis])
			action = (tf.random.uniform([1, 1]) > left_prob)
			obs, reward, done, truncated, info = env.step(int(action[0, 0].numpy()))
			# plot_environment(env)
			env.render()
			episode_rewards += reward
			if done:
				break
		totals.append(episode_rewards)

	print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))


if __name__ == "__main__":
	use_my_model("cart_pole240.h5")
