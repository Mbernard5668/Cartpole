
from ActorCriticNet import ActorCriticNet
from WorkerAgent import WorkerAgent
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue
tf.config.set_visible_devices([], 'GPU')

# import matplotlib.pyplot as plt

# Main objective: use image array.





class MasterAgent:
    def __init__(self):
        env = gym.make('CartPole-v0')
        self.num_actions = env.action_space.n # Action space
        self.num_states = env.observation_space.shape[0]
        # Global Network Initialization
        self.actor_critic_net = ActorCriticNet(self.num_actions, self.num_states).create_network()
        self.optimizer = tf.keras.optimizers.Adam()

    def train(self):
        # Parallel training
        res_queue = Queue() # Queue to store tasks
        # We pass the ActorCrticNet instance to the workers
        workers = [WorkerAgent(self.num_actions,self.num_states, i,self.actor_critic_net,self.optimizer,res_queue) for i in range(multiprocessing.cpu_count())]
        print("Number of workers: ",len(workers))
        for worker in workers:
            worker.start()

        average_reward = []
        cycle_ep_avg_reward = []
        EP_MAX = 4
        cycle_ep_count = 0
        while cycle_ep_count < EP_MAX: # TODO: include episode counter condition to stop loop
            if not res_queue.empty():
                average_reward.append(res_queue.get())
                if len(average_reward) == multiprocessing.cpu_count():
                    template = "Average reward: {} "
                    cycle_ep_avg_reward.append(sum(average_reward) / len(average_reward))
                    #print(template.format(sum(average_reward) / len(average_reward)))
                    print(template.format(cycle_ep_avg_reward[-1]))
                    average_reward.clear()
                    cycle_ep_count+=1

        [w.stop_() for w in workers]
        [w.join() for w in workers]
        # Plot results
        plt.xlabel('Cycles')
        plt.ylabel('Average Reward')
        plt.plot(cycle_ep_avg_reward, color='r')
        plt.show()

        eval_worker = WorkerAgent(self.num_actions,self.num_states,-1,self.actor_critic_net,self.optimizer,res_queue)
        eval_worker.start()


if __name__ == '__main__':
    

    a3c_agent = MasterAgent()
    a3c_agent.train()
    print("DONE")






"""
if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
"""