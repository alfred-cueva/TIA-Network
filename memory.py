import numpy as np
import random
import torch


class ReplayMemory:
    def __init__(self, memory_size, action_size, obs_size, latent_size, num_task):
        
        self.states = np.zeros(shape=(memory_size, obs_size))
        self.next_states = np.zeros(shape=(memory_size, obs_size))
        self.actions = np.zeros(memory_size)
        self.rewards = np.zeros(memory_size)
        self.terminals = np.zeros(memory_size)
        self.taus = []
        for i in range(num_task):
            self.taus.append([])
        self.zs = np.zeros(shape=(memory_size,latent_size))
        self.tasks = np.zeros(memory_size)
        self.count = 0
        self.current = 0
        self.count_task = []
        for i in range(num_task):
            self.count_task.append(0)
        self.memory_size = memory_size

    def add(self, state, reward, action, terminal, next_state, tau, z, task):
        self.states[self.current] = state
        self.rewards[self.current] = reward
        self.actions[self.current] = action
        self.terminals[self.current] = terminal
        self.next_states[self.current] = next_state
        if self.count_task[task] > self.memory_size:
            self.taus[task][self.count_task[task] % self.memory_size] = tau
        else:
            self.taus[task].append(tau) 
        
        self.zs[self.current] = z.detach().numpy()
        self.tasks[self.current] = task
        

        self.current = (self.current + 1) % self.memory_size
        self.count += 1
        self.count_task[task] += 1

    def sample(self, batch_size):
        state_batch = []
        reward_batch = []
        action_batch = []
        terminal_batch = []
        next_state_batch = []
        tau_batch = []
        z_batch = []
        task_batch = []


        for i in range(batch_size):
            data_index = random.randint(0, self.current-1 if self.count < self.memory_size else self.memory_size-1)
            state_batch.append(self.states[data_index])
            reward_batch.append(self.rewards[data_index])
            action_batch.append(self.actions[data_index])
            terminal_batch.append(self.terminals[data_index])
            next_state_batch.append(self.next_states[data_index])

            
            tau_batch.append(torch.stack(self.taus[int(self.tasks[data_index])][0:data_index+1]))
            
            z_batch.append(self.zs[data_index])
            task_batch.append(self.tasks[data_index])

        return state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, tau_batch, z_batch, task_batch
