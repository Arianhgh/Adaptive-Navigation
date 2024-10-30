import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import sys
import gymnasium as gym
import random
import numpy as np
import time
# import tensorflow as tf
from nn_builder.pytorch.NN import NN
# from tensorboardX import SummaryWriter
# from torch.optim import optimizer as optim
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from torch.utils.tensorboard import SummaryWriter
import traci

class DRQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HyperNetworkForWeights(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HyperNetworkForWeights, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.abs(self.fc1(x))
        return x

class HyperNetworkForBiases(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HyperNetworkForBiases, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x

class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hypernet_embed_dim):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hypernet_embed_dim = hypernet_embed_dim

        self.hyper_w1 = HyperNetworkForWeights(state_dim, hypernet_embed_dim * n_agents)
        self.hyper_w2 = HyperNetworkForWeights(state_dim, hypernet_embed_dim)
        self.hyper_b1 = HyperNetworkForBiases(state_dim, hypernet_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, self.n_agents)

        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.hypernet_embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.hypernet_embed_dim)
        w2 = self.hyper_w2(states).view(-1, self.hypernet_embed_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, -1, 1)

class QMIX:
    def __init__(self, n_agents, state_dim, obs_dim, n_actions, hidden_dim, hypernet_embed_dim, config):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.agent = DRQN(obs_dim, hidden_dim, n_actions)
        self.mixer = MixingNetwork(n_agents, state_dim, hypernet_embed_dim)
        self.target_agent = DRQN(obs_dim, hidden_dim, n_actions)
        self.target_mixer = MixingNetwork(n_agents, state_dim, hypernet_embed_dim)

        self.optimizer = optim.RMSprop(list(self.agent.parameters()) + list(self.mixer.parameters()), lr=0.0005)
        self.loss_fn = nn.MSELoss()
        self.env_episode_number = 0
        self.config = config
        self.environment = config.environment

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        print("num_episodes: ", num_episodes)
        while self.env_episode_number < num_episodes:
            # if self.env_episode_number==num_episodes-1:
            #     breakpoint()
            print("episode number: ", self.env_episode_number)
            self.run()
            self.env_episode_number += 1
            self.environment.log_data()
    
    def run(self):
        """Runs a step within a game including a learning step if required"""
        self.reset_game()
        k = 0
        while not self.done:
            if k % 100 == 0:
                print("traci simulation time: ", traci.simulation.getTime())
            k += 1
            routing_queries=self.environment.get_routing_queries()
            actions = self.pick_action(routing_queries)
            num_new_exp=self.conduct_action(actions)
            if num_new_exp==0:
                continue

            if self.config.training_mode:
                self.save_experience()
                self.learn()
    
    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.seed = self.config.seed
        self.states = self.environment.reset(self.env_episode_number,self.config)
        self.mem_states= None
        self.mem_next_states = None
        self.mem_actions = None
        self.mem_rewards = None
        self.mem_done =None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()
        self.logger.info("Reseting game -- New start state {}".format(self.states))


    def learn(self, batch, max_episode_len, gamma):
        states, actions, rewards, next_states, dones, avail_actions, agent_ids = batch

        # Compute Q values for the current state
        q_values, _ = self.agent(states)
        q_values = q_values.gather(2, actions)

        # Compute target Q values for the next state
        with torch.no_grad():
            next_q_values, _ = self.target_agent(next_states)
            next_q_values[avail_actions == 0] = float('-inf')
            next_q_values = next_q_values.max(dim=2)[0]
            next_q_tot = self.target_mixer(next_q_values, next_states)

        targets = rewards + gamma * (1 - dones) * next_q_tot
        q_tot = self.mixer(q_values, states)

        # Compute loss and update parameters
        loss = self.loss_fn(q_tot, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_targets(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def get_action(self, obs, hidden):
        q_values, hidden = self.agent(obs, hidden)
        action = q_values.argmax(dim=2)
        return action, hidden
