import torch
import numpy as np
import gym
import os
import random
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import AgentConfig, EnvConfig
from memory import ReplayMemory
from network import MlpDQN, ModelPredictor, Encoder_rnn, Decoder_rnn, Seq2Seq
from ops import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class Agent(AgentConfig, EnvConfig):
    def __init__(self):

        #setting multiple environments
        self.env = []
        self.tau = []
        self.state = []
        self.z = []
        self.max_reward = []
        self.reward_episode = []
        self.epsilon = []
        self.alpha = []
        self.reward_term_epi = []
        self.desired_pos = []
        for i in range(len(self.env_list)):
            #For Atari
            # self.env.append(gym.make(self.env_list[i],full_action_space=True))
            self.env.append(gym.make(self.env_list[i]))
            self.tau.append([])
            # print(self.env[i].reset())
            obs, dp = self.env[i].reset()
            self.state.append(obs)
            self.desired_pos.append(dp['desired_pos'])
            self.z.append(torch.zeros(self.latent_size, device=device))
            self.max_reward.append(0)
            self.reward_episode.append(0)
            self.epsilon.append(1)
            self.alpha.append(1)
            self.reward_term_epi.append(0)
        
        self.action_size = self.env[i].action_space.n  
        self.obs_size = self.env[i].observation_space.shape[0]
        self.memory_buffer = ReplayMemory(memory_size=100000, action_size=self.action_size, obs_size=self.obs_size, latent_size=self.latent_size, num_task = len(self.env_list))               
        self.dqn_network = MlpDQN(action_size=self.action_size, input_size=self.obs_size+self.latent_size).to(device)
        self.dqn_network_target = MlpDQN(action_size=self.action_size, input_size=self.obs_size+self.latent_size).to(device)
        self.predictor_network = ModelPredictor(output_size=(2+self.obs_size), input_size=(self.action_size+self.obs_size+self.latent_size)).to(device) #reward and terminal


        input_dim = self.obs_size+self.action_size+2
        output_dim = self.obs_size+self.action_size+2
        hid_dim = self.latent_size

        self.enc = Encoder_rnn(input_dim = input_dim, hid_dim = hid_dim, n_layers=self.n_layers, dropout = self.enc_dropout)
        self.dec = Decoder_rnn(output_dim = output_dim, hid_dim = hid_dim, n_layers=self.n_layers, dropout = self.dec_dropout)
        self.seq2seq = Seq2Seq(self.enc, self.dec, device).to(device)

        ##modify weight decay TODO
        self.optimizer_dqn = optim.Adam(self.dqn_network.parameters(), lr=self.learning_rate_dqn)#, weight_decay = self.weight_decay_dqn)
        self.optimizer_pred = optim.Adam(self.predictor_network.parameters(), lr=self.learning_rate_pred)#, weight_decay = self.weight_decay_pred)
        self.optimizer_encoder = optim.Adam(self.seq2seq.parameters(), lr=self.learning_rate_encd)#, weight_decay = self.weight_decay_encd)
        
        #modify loss & criterion
        self.loss_dqn = 0
        self.loss_pred = 0
        self.loss_encd = 0
        self.criterion_dqn = nn.MSELoss()
        self.criterion_pred = nn.MSELoss()
        self.criterion_encd = nn.MSELoss()
        

    def train(self):
        step =0
        plotlist = []
        # A new episode
        while step < self.max_step:
            i = random.randrange(len(self.env_list))


            #Extracting latent representation from tau
            if len(self.tau[i])==0:
                z_hat = self.z[i]
            
            else:
                input_tau = torch.zeros(len(self.tau[i]), (self.obs_size+self.action_size+2),device=device)
                for j in range(len(self.tau[i])):
                    input_tau[j][:] = self.tau[i][j]
                z_hat = self.seq2seq.encoder(input_tau.unsqueeze(1)).squeeze(0).squeeze(0)

            self.z[i] = self.z[i]*(1-self.alpha[i])+self.alpha[i]*z_hat
            # self.alpha_decay(i)
            
            self.z[i]= z_hat
            # self.z[i] = torch.zeros_like(z_hat)

            current_state = self.state[i]
            current_state_torch = torch.tensor(current_state , device = device, dtype = torch.float)
            input_policy = torch.cat([current_state_torch,self.z[i]]).to(device)

            action = random.randrange(self.action_size) if np.random.rand() < self.epsilon[i] else \
                    torch.argmax(self.dqn_network(input_policy)).item() 
            # print("state and task", i, current_state)
            # print(action)
            next_state, reward, terminal, _, _ = self.env[i].step(action)
            
            if terminal:
                reward = -1
            
            if len(self.tau[i])!=0:
                self.memory_buffer.add(current_state, reward, action, terminal, next_state, self.tau[i][-1], self.z[i].cpu(), i)
            
            self.state[i] = next_state
            # print(self.reward_episode[i], terminal)
            self.reward_episode[i] = self.reward_episode[i]*self.gamma + reward 
            # print(i, self.reward_episode[i], self.gamma, reward)
            
            if self.reward_episode[i] > self.max_reward[i]:
                self.max_reward[i] = self.reward_episode[i]

            if len(self.tau[i]) == self.tau_max_length: 
                self.tau[i].pop(0)
            action_onehot = torch.zeros(self.action_size).to(device)
            action_onehot[action] = 1    
            next_state_torch = torch.tensor(next_state, device = device, dtype = torch.float)
            new_tau = torch.cat([next_state_torch, action_onehot]).to(device)
            new_tau = torch.cat([new_tau, torch.FloatTensor([reward, terminal]).to(device)])
            self.tau[i].append(new_tau)


            if step > self.learning_start_dqn:
                if step % self.learning_freq_dqn==0:
                    self.learn_dqn()
            if step > self.learning_start_encd:
                if step % self.learning_freq_encd==0:
                    self.learn_encd()
            if step > self.learning_start_pred:
                if step % self.learning_freq_pred==0:
                    self.learn_pred()
                
            if step % self.target_update_freq==0:
                self.dqn_network_target.load_state_dict(self.dqn_network.state_dict())

            plotlist.append(self.max_reward)            
            self.epsilon_decay(i)
            step += 1
            if terminal:
                print("current step: ", step, "(", i, " :terminated )")
                for jj in range(len(self.env_list)):
                    print("Task " + str(jj) + "Desired Pos " + str(round(self.desired_pos[jj],2)) + " max reward: ", self.max_reward[jj], "epi reward", self.reward_term_epi[jj], "Epsillon ", self.epsilon[jj], "z ", self.z[jj]) 
                print("==================================================================")
                self.state[i], _ = self.env[i].reset()
                self.reward_term_epi[i] = self.reward_episode[i]
                self.reward_episode[i] = 0
                # print(i,"terminated :", self.reward_episode[i])
                
                
            if step % 200==0:
                print("DQN_loss")
                print(self.loss_dqn)
                print("encd_loss")
                print(self.loss_encd)
                print("pred_loss")
                print(self.loss_pred)
                
            if step > self.weight_save_min and step % self.weight_save_freq == 0:
                torch.save(self.dqn_network.state_dict(), 'result4/dqn_step_'+str(step)+'.pth')
                torch.save(self.predictor_network.state_dict(), 'result4/pred_step_'+str(step)+'.pth')
                torch.save(self.seq2seq.state_dict(), 'result4/encode_step_'+str(step)+'.pth')
                



        # for env in self.env_list:
        #     self.env.close()

        #self.plot_graph(plotlist)

    def learn_dqn(self):
        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, tau_batch, z_batch, task_batch = self.memory_buffer.sample(self.batch_size_rl_dqn)
        
        y_batch = torch.FloatTensor().to(device)
        q_val_batch = torch.FloatTensor().to(device)
        for p in self.predictor_network.parameters():
            p.requires_grad = False
        for i in range(self.batch_size_rl_dqn):
            if terminal_batch[i]:
                y_batch = torch.cat([y_batch,torch.FloatTensor([reward_batch[i]]).to(device)])                                                                          
            elif (self.TD_step > 0):
                j=0
                state_next = next_state_batch[i]
                one = 1
                y = one*reward_batch[i]
                while (j<self.TD_step):
                    input_policy = torch.cat([torch.FloatTensor(state_next),torch.FloatTensor(z_batch[i])]).to(device)
                    action_idx = random.randrange(self.action_size) if np.random.rand() < self.epsilon[int(task_batch[i].item())] else \
                        torch.argmax(self.dqn_network_target(input_policy)).item() 
                    action_next = torch.zeros(self.action_size).to(device)
                    action_next[action_idx] = 1
                    state_next_torch = torch.tensor(state_next, device= device, dtype = torch.float)              
                    estimate = self.predictor_network.forward(torch.cat([\
                    state_next_torch,action_next,torch.FloatTensor(z_batch[i]).to(device)]))
                    state_next_next = estimate[0:self.obs_size]
                    reward_next = estimate[-2]
                    terminal_next = estimate[-1]
                    one *= self.gamma
                    state_next = state_next_next
                    y += one*reward_next
                    if terminal_next:
                        break
                    j +=1
                if not terminal_next:
                    input_policy = torch.cat([torch.FloatTensor(state_next),torch.FloatTensor(z_batch[i])]).to(device)
                    y += one*self.gamma*torch.max(self.dqn_network(input_policy).to(device))
                y_batch = torch.cat([y_batch, y.reshape(1)])
            else :
                input_policy = torch.cat([torch.FloatTensor(next_state_batch[i]),torch.FloatTensor(z_batch[i])]).to(device)
                y = reward_batch[i] + self.gamma*torch.max(self.dqn_network_target(input_policy).to(device))
                y_batch = torch.cat([y_batch, y.reshape(1)])
            
            input_policy = torch.cat([torch.FloatTensor(state_batch[i]),torch.FloatTensor(z_batch[i])]).to(device)
            q_val_batch = torch.cat([q_val_batch,self.dqn_network(input_policy).to(device)[int(action_batch[i].item())].reshape(1)])
            

        self.loss_dqn = self.criterion_dqn(q_val_batch, y_batch).mean()
        # print('action')
        # print(action_batch)
        # print('state')
        # print(state_batch)
        # print('qval')
        # print(q_val_batch)
        # print('target')
        # print(y_batch)

        self.optimizer_dqn.zero_grad()
        
        self.loss_dqn.backward()
        self.optimizer_dqn.step()
        

    def learn_pred(self):
        
        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, tau_batch, z_batch, task_batch = self.memory_buffer.sample(self.batch_size_rl_pred)
        for p in self.predictor_network.parameters():
            p.requires_grad = True
        pred_batch = torch.FloatTensor().to(device)
        real_batch = torch.FloatTensor().to(device)
        for i in range(self.batch_size_rl_pred):    
            action_next = torch.zeros(self.action_size).to(device)
            action_next[int(action_batch[i].item())] = 1
            pred_batch = torch.cat([pred_batch,self.predictor_network.forward(torch.cat([torch.FloatTensor(state_batch[i]).to(device), \
            action_next, torch.FloatTensor(z_batch[i]).to(device)]))],0) 
            real_batch = torch.cat([real_batch,torch.cat([torch.FloatTensor(next_state_batch[i]).to(device),torch.FloatTensor([reward_batch[i],terminal_batch[i]]).to(device)])])

        self.loss_pred = self.criterion_pred(pred_batch, real_batch).mean()

        self.optimizer_pred.zero_grad()
        
        self.loss_pred.backward()
        self.optimizer_pred.step()

    def epsilon_decay(self, i):
        self.epsilon[i] *= self.epsilon_decay_rate
        self.epsilon[i] = max(self.epsilon[i], self.epsilon_minimum)

    def alpha_decay(self, i):
        self.alpha[i] *= self.alpha_decay_rate
        self.alpha[i] = max(self.alpha[i], self.alpha_minimum)

    def learn_encd(self):

        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch, tau_batch, z_batch, task_batch = self.memory_buffer.sample(self.batch_size_seq)

        tau_batch_torch = nn.utils.rnn.pad_sequence(tau_batch)
        outputs, _ = self.seq2seq(tau_batch_torch, tau_batch_torch, teacher_forcing_ratio=self.teacher_forcing_ratio)
        ## input = [src len, batch size, input dim]
        self.loss_encd = self.criterion_encd(outputs, tau_batch_torch)

        self.optimizer_encoder.zero_grad()
        self.loss_encd.backward()
        self.optimizer_encoder.step() 


    def plot_graph(self, plotlist):
        return 0


        