import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from reinforce import Reinforce
from reinforce_with_baseline import ReinforceWithBaseline

def visualize_episode(env, model):
    """
    HELPER - do not edit.
    Takes in an enviornment and a model and visualizes the model's actions for one episode.
    We recomend calling this function every 20 training episodes. Please remove all calls of 
    this function before handing in.

    :param env: The cart pole enviornment object
    :param model: The model that will decide the actions to take
    """

    done = False
    state = env.reset()
    env.render()

    while not done:
        newState = np.reshape(state, [1, state.shape[0]])
        prob = model.call(newState)
        newProb = np.reshape(prob, prob.shape[1])
        action = np.random.choice(np.arange(newProb.shape[0]), p = newProb)

        state, _, done, _ = env.step(action)
        env.render()


def visualize_data(total_rewards):
    """
    HELPER - do not edit.
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep, which
    are calculated by summing the rewards for each future timestep, discounted
    by how far in the future it is.
    For example, in the simple case where the episode rewards are [1, 3, 5] 
    and discount_factor = .99 we would calculate:
    dr_1 = 1 + 0.99 * 3 + 0.99^2 * 5 = 8.8705
    dr_2 = 3 + 0.99 * 5 = 7.95
    dr_3 = 5
    and thus return [8.8705, 7.95 , 5].
    Refer to the slides for more details about how/why this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    # TODO: Compute discounted rewards
    reversed = rewards[::-1]   # [5, 3 ,1]
    discounted = []
    discounted.append(reversed[0]) # [5]
    for i in range(1, len(reversed)):
      add_to_list = reversed[i] + discounted[0] * discount_factor   # 3 + 5*0.99
      discounted.insert(0, add_to_list)  # [7.95, 5]
      # second round: 1 + 7.95*0.99
    return discounted 



def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False

    while not done:
        # TODO:
        # 1) use model to generate probability distribution over next actions

        # state from env is of shape (state_size,)
        # change shape to (1, state_size)


        # this is the shape of state: (4,)
        state = np.reshape(state, (1, state.shape[0])) # need to match the input dimension of the call function 
        # this is the shape of state after reshaping: (1, 4)

        # here, the episode_length is 1 when call the model to compute prob for 1 state
        prob = model.call(state)  # shape: (1, num_actions)
        # this is the shape of prob: (1, 2)
        prob = np.reshape(prob, (model.num_actions)) # shape: (num_actions,)
        # this is the shape of prob after reshaping: (2,)

        # 2) sample from this distribution to pick the next action
        action = np.random.choice(a = np.arange(model.num_actions), p = prob)  # a will be either 0 or 1 
        # a random sample is generated from np.arange(model.num_actions) with probabilities prob 


        states.append(state)
        actions.append(action)
        state, rwd, done, _ = env.step(action)
        rewards.append(rwd)

    return states, actions, rewards


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """
    # TODO:


    with tf.GradientTape() as tape:
      # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
      states, actions, rewards = generate_trajectory(env, model)
      # 2) Compute discounted rewards.
      discounted_rewards = discount(rewards)
      # 3) Compute the loss from the model and run backpropagation on the model.
      states = np.array(states)
      loss = model.loss(states, actions, discounted_rewards)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    total_rewards = np.sum(rewards)

    return total_rewards



def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
        exit()

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print(f'the state_size is: {state_size}')   # 4
    print(f'the num_actions is: {num_actions}') # 2

    # Initialize model
    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions)
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)

    # TODO:
    # 1) Train your model for 650 episodes, passing in the environment and the agent.
    # 1a) OPTIONAL: Visualize your model's performance every 20 episodes.
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards.
    # 3) After training, print the average of the last 50 rewards you've collected.

    episodes = 650
    total_rewards = []
    
    # model = Reinforce(state_size, num_actions)
    # model = ReinforceWithBaseline(state_size, num_actions)

    for i in range(episodes):
      episode_reward = train(env, model)
      total_rewards.append(episode_reward)
    average = np.mean(total_rewards[-50:])
    print(f'The average of the last 50 rewards is: {average}')

    # TODO: Visualize your rewards.
    visualize_data(total_rewards)



if __name__ == '__main__':
    main()
