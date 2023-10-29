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

class Deployment(AgentConfig, EnvConfig):
    def __init__(self):
        #setting multiple environments
        self.env = wrap_env(gym.make('CartPoleStay-v0', render_mode='rgb_array'))
        self.env = gym.make('CartPoleStay-v0')
        self.tau = []
        self.state, dp = self.env.reset()
        self.z = torch.zeros(self.latent_size, device=device)
        self.max_reward = 0
        self.reward_episode = 0
        self.epsilon = 1
        self.alpha = 1
        self.reward_term_epi = 0
        self.desired_pos = dp['desired_pos']
        
        self.action_size = self.env.action_space.n  
        self.obs_size = self.env.observation_space.shape[0]
        self.memory_buffer = ReplayMemory(memory_size=100000, action_size=self.action_size, obs_size=self.obs_size, latent_size=self.latent_size, num_task = len(self.env_list))               
        self.dqn_network = MlpDQN(action_size=self.action_size, input_size=self.obs_size+self.latent_size).to(device)
        self.predictor_network = ModelPredictor(output_size=(2+self.obs_size), input_size=(self.action_size+self.obs_size+self.latent_size)).to(device) #reward and terminal


        input_dim = self.obs_size+self.action_size+2
        output_dim = self.obs_size+self.action_size+2
        hid_dim = self.latent_size

        self.enc = Encoder_rnn(input_dim = input_dim, hid_dim = hid_dim, n_layers=self.n_layers, dropout = self.enc_dropout)
        self.dec = Decoder_rnn(output_dim = output_dim, hid_dim = hid_dim, n_layers=self.n_layers, dropout = self.dec_dropout)
        self.seq2seq = Seq2Seq(self.enc, self.dec, device).to(device)

        
        step = 175000
        self.dqn_network.load_state_dict(torch.load('result/dqn_step_'+str(step)+'.pth'))
        self.predictor_network.load_state_dict(torch.load('result/pred_step_'+str(step)+'.pth'))
        self.seq2seq.load_state_dict(torch.load('result/encode_step_'+str(step)+'.pth'))

    def alpha_decay(self):
        self.alpha *= self.alpha_decay_rate
        self.alpha = max(self.alpha, self.alpha_min_run)

    def run(self):
        for i in range(400000):
            
            if len(self.tau)==0:
                z_hat = self.z
            else:
                input_tau = torch.zeros(len(self.tau), (self.obs_size+self.action_size+2),device=device)
                for j in range(len(self.tau)):
                    input_tau[j][:] = self.tau[j]
                z_hat = self.seq2seq.encoder(input_tau.unsqueeze(1)).squeeze(0).squeeze(0)
            self.z = self.z*(1-self.alpha)+self.alpha*z_hat
            # self.alpha_decay()
            
            self.z= z_hat

            current_state = self.state
            current_state_torch = torch.tensor(current_state , device = device, dtype = torch.float)
            input_policy = torch.cat([current_state_torch,self.z]).to(device)

            action = torch.argmax(self.dqn_network(input_policy)).item() 
            next_state, reward, terminal, _, _ = self.env.step(action)

            self.state = next_state
            # print(self.reward_episode[i], terminal)
            self.reward_episode = self.reward_episode*self.gamma + reward 
            # print(i, self.reward_episode[i], self.gamma, reward)
            
            if self.reward_episode > self.max_reward:
                self.max_reward = self.reward_episode

            if len(self.tau) == self.tau_max_length: 
                self.tau.pop(0)
            action_onehot = torch.zeros(self.action_size).to(device)
            action_onehot[action] = 1    
            next_state_torch = torch.tensor(next_state, device = device, dtype = torch.float)
            new_tau = torch.cat([next_state_torch, action_onehot]).to(device)
            new_tau = torch.cat([new_tau, torch.FloatTensor([reward, terminal]).to(device)])
            self.tau.append(new_tau)
            
            # print("current pos", next_state[0])
            
            if terminal:
                print("Step "+ str(i)+ " Desired Pos " + str(round(self.desired_pos,2)) +\
                    " max reward: ", self.max_reward, " epi reward", self.reward_term_epi, " Z ", self.z)
                print("==================================================================")
                self.state, _ = self.env.reset()
                self.reward_term_epi = self.reward_episode
                self.reward_episode = 0
                
        
            
