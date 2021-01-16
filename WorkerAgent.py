import numpy as np
import tensorflow as tf
import threading
from ActorCriticNet import ActorCriticNet
from Trajectory import Trajectory
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

LOOKAHEAD = 5
GAMMA = 0.99
EPSILON = tf.keras.backend.epsilon() # Used to add 'fuzz' when computing the log of the policy.

import gym

class WorkerAgent(threading.Thread):
    def __init__(self, num_actions, num_states, worker_id, global_actor_critic, global_optimizer, res_queue):
        super(WorkerAgent, self).__init__()

        self.stop = False # flag to stop the thread
        self.num_actions = num_actions
        self.num_states = num_states
        # Recover the instance of the Global ActorCriticNet from the MasterAgent
        self.global_actor_critic = global_actor_critic
        # Create the local ActorCritic Network
        self.local_actor_critic = ActorCriticNet(self.num_actions, self.num_states).create_network()
        self.global_optimizer = global_optimizer
        self.res_queue = res_queue
        self.worker_id = worker_id
        self.critic_loss = tf.keras.losses.Huber() # Ex: Mean Squared Loss MSL

        # Copy the global network's weights in the local model with new weights.
        self.local_actor_critic.set_weights(self.global_actor_critic.get_weights())

    def terminate(self):
        self._running = False

    def is_eval(self):
        """
        Checks if the current agent is for evaluating the global network
        :return:
        """
        return self.worker_id ==-1

    def act(self, state): # DONE
            # Recover current Policy
            policy, _ = self.local_actor_critic(state)
            return np.random.choice(self.num_actions, p=np.squeeze(policy)) # Select an action based on the probability distribution provided by the policy.


    def train(self, trajectory, next_state):

            # Create network update from the trajectory
            # State batch

            states_batch = trajectory.state_history[0]

            for i in range(1, len(trajectory.state_history)):
                states_batch = tf.concat([states_batch, trajectory.state_history[i]], 0)
            #print("States_batch\n",states_batch)

            # Compute the One-Hot encoding vectors for each action
            actions_one_hot = tf.one_hot(trajectory.action_history, depth=self.num_actions)
            #print("Actions_one_hot\n", actions_one_hot)

            tvs = self.local_actor_critic.trainable_variables # Recover training weights
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]

            with tf.GradientTape(persistent=True) as tape:
                # Operations are recorded
                # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically) are automatically watched
                roll_back_reward = 0
                targets = []
                terminal = True

                # TODO check: last state
                # If the Agent successfully completed the task
                if len(next_state) > 0:
                    next_state = tf.convert_to_tensor(next_state)
                    next_state = tf.expand_dims(next_state, 0)
                    _, critic = self.local_actor_critic(next_state)
                    roll_back_reward = critic[0, 0]
                    terminal = False

                # Calculate the discounted reward at every time-step.
                for r in reversed(trajectory.reward_history):
                    roll_back_reward = r + GAMMA * roll_back_reward
                    targets.insert(0, roll_back_reward) # Insert reward in list at position 0

                # Compute the Update
                action_probs, critic = self.local_actor_critic(states_batch) # For each state in the batch recover the action_probs and critic values.
                # Use these to calculate the loss
                for i in range(len(targets)): # For every time-step in the trajectory
                    # Value loss: use the set loss
                    critic_loss = self.critic_loss(tf.expand_dims(targets[i], 0), tf.expand_dims(critic[i, 0], 0))
                    # Actor Loss
                    actor_loss = -tf.reduce_mean(actions_one_hot[i] * tf.math.log((EPSILON + action_probs[i])))
                    advantage = targets[i] - critic[i]
                    total_loss = (actor_loss * advantage) + critic_loss # Policy Loss + value loss
                # Computing the gradient of the loss
                grads = tape.gradient(total_loss, self.local_actor_critic.trainable_variables)
                accum_grads = [accum_vars[i].assign_add(gv) for i, gv in enumerate(grads)]
            del tape
            # Apply gradient to each trainable variable
            self.global_optimizer.apply_gradients(zip(accum_grads, self.global_actor_critic.trainable_variables))
            # Update local model with new weights
            self.local_actor_critic.set_weights(self.global_actor_critic.get_weights())

    def stop_(self):
        print("STOP")
        self.stop = True



    def run(self):


        env = gym.make("CartPole-v0") # Create the environment

        episode_done = True
        trajectory = Trajectory()
        next_state = []
        episode_num = 0
        episode_reward = 1 # TODO: check for alternative init rewards

        while True:

            if self.stop: # Check if we need to stop the current thread
                print("Worker:",self.worker_id, "STOP")
                break

            if episode_done: # Check if the episode is done
                state = env.reset() # Reset the environment and recover initial state
                episode_done = False
                template = "in worker {}, episode {} done after {} steps" # Template for print
                print(template.format(self.worker_id, episode_num, episode_reward))
                self.res_queue.put(episode_reward)
                episode_num += 1
                episode_reward = 1

            for i in range(LOOKAHEAD):
                #print("I=",i)
                if self.is_eval(): # Render curretn state of the environement
                    env.render()
                state = tf.convert_to_tensor(state)  # Convert state to tensor
                state = tf.expand_dims(state, 0)
                action = self.act(state) # Select action based on the current policy.
                trajectory.store(s=state, a=action)

                state, reward, episode_done, _ = env.step(action) # Recover next state, reward and if the episode is done.
                #print("|New state\n",state,"\n|reward: \n",reward," \n|episode done \n",episode_done)

                if episode_done: # If the agent fails the task
                    reward = -1
                # Store the reward
                trajectory.store(r=reward)
                episode_reward += reward
                next_state = state # update current state

                if episode_done:
                    next_state = []
                    break

            if episode_done and self.is_eval():
                if self.is_eval():
                    env.close()
                    template = "in worker {}, episode {} done after {} steps"  # Template for print
                    print(template.format(self.worker_id, episode_num, episode_reward))
                    break

            if not self.is_eval():
                self.train(trajectory, next_state)# Update network using the trajectory
            trajectory.clear()











