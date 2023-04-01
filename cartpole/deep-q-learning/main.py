import gym
from tensorflow import keras
import numpy as np
from collections import deque
import tensorflow as tf

# the goal here is to create a NN that imitates and replaces the need of a q table
# for the we will first build the environment using the open ai gym library
env = gym.make("CartPole-v1")
action_shape = 2
state_shape = 4

# next we create the model that will replace the need of a q table
# we feed a state and it outputs the q value for every possible action

model = keras.models.Sequential(
	[
		keras.layers.Dense(32, activation="relu", input_dim=state_shape),
		keras.layers.Dense(32, activation="relu"),
		keras.layers.Dense(action_shape)
	]
)


# then we need to be sure the agent explores all the possible states and not only those with
# the highest known reward. This function returns either a random acion or the best action for future known rewards
def policy(state, epsilon=0):
	if np.random.rand() > epsilon:
		action = env.action_space.sample()
	else:
		action = np.argmin(model.predict(state))

	return action


# now we need to create what is called a replay buffer, that stores all experiences in order to have a vast data set to
# train our model. Each experience will contain 5 values inside, the current state, the chosen action, the reward,
# the next state and whether that state is a terminal state (done == True).

# the data structure selected for that is a deque
replay_buffer = deque(maxlen=2000)


# A function is needed to sample some of the stored experiences to make our training batch. It will return five
# numpy arrays containing each experience elements
def sample_experiences(batch_size):
	sample_indexes = np.random.randint(len(replay_buffer), size=(batch_size,))
	samples = [replay_buffer[index] for index in sample_indexes]

	cur_states, actions, next_states, rewards, dones = [np.array([sample[index] for sample in samples]) for index in range(5)]

	return cur_states, actions, next_states, rewards, dones


# for each step being played, the agent must choose the action taking the epsilon-greedy policy, and then store the
# result in the replay buffer
def play_one_step(env, state, epsilon):
	action = policy(epsilon)
	new_state, reward, done, truncated, info = env.step(action)

	replay_buffer.append((state, action, new_state, reward, done))

	return new_state, reward, done, info


# training parameters
batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error
n_episodes = 6000
max_steps = 200
filling_period = 50
epsilon = 0.0001


# define function to train the NN to make it learn the patterns of our env and replicate the behavior of a Q table.
# that means that if we give our NN a state, it will return the Q value for every action possible.
# for that we use the bellman´s equation : Q(s,a) = r + eps*Q(s,a)
def train_dqn(batch_size):
	# the goal here is to apply the right gradients to make the current predictions for each current state and only for
	# the action chosen for that state closer to the ideal q value considering that the next state has already the
	# proper Q value associated to it.

	# the more we train, the better the q values get for every possible state and for every possible action

	# get samples from experiences buffer
	cur_states, actions, next_states, rewards, dones = sample_experiences(batch_size)

	# get next states Q values.
	# They're needed to compute the current states Q values using the discount factor and the current reward.
	all_next_Q_values = model.predict(next_states)  # Q value for every action
	next_Q_values = np.max(all_next_Q_values, axis=1)  # highest Q value: defines the chosen action

	# get goal Q values
	target_Q_values = (rewards + (1-done)*discount_factor*next_Q_values)

	# convert actions to one hot
	mask = tf.one_hot(actions, action_shape)

	with tf.GradientTape() as tape:
		# this part is used to compute the loss function keeping track of its gradient,
		# so we can apply them to the NN later.
		all_Q_values = model(cur_states)
		Q_values = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
		loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
		print("\n\nLoss: ", loss)

	# get the gradients for each trainable variable and apply them
	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))


# first we need to fill the buffer a little. It is important to fill the buffer and train at the same time,
# otherwise, our deque would be only composed of random experiences an not the experiences our environment is most likely
# to produce when trained. Anyway, we give it 50 episodes to fill it up, end then we start training
for episode in range(n_episodes):
	state, info = env.reset()
	rewards = 0
	for step in range(max_steps):
		new_state, reward, done, info = play_one_step(env, state, epsilon=epsilon)
		rewards += reward
		if done:
			break
	print("Episode: ", episode)
	print("Rewards:", rewards)

	if episode > 50:
		train_dqn(batch_size)

	if episode % 50 == 0:
		print("model_saved")
		model.save("cart_pole" + str(episode) + ".h5")

