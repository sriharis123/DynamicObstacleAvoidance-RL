import torch
import torch.nn as nn
import numpy as np

import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from simulator import rollout
import sys

STATE_DIM = 23  # LIDAR dims + other state stuff
HIDDEN_DIM = 36
ACTION_DIM = 2  # 2 Actions: L/R and F/B

class ActorLSTM(nn.Module):
    # int, int, int
    def __init__(self, state_dim, hidden_dim, action_dim, alpha=0.0001):
        super(ActorLSTM, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.Softmax(action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), alpha)
        self.loss = nn.MSELoss()

    # state must be torch tensor of shape (time_steps, state_dim)
    # output shape is (time_steps, action_dim)
    def forward(self, state):
        h0 = torch.zeros(1, state.shape[0], self.hidden_dim)
        c0 = torch.zeros(1, state.shape[0], self.hidden_dim)
        ht, (hn, cn) = self.lstm(state, (h0, c0))     # ht will be size (time_steps, hidden_dims)
        return self.activation(self.linear(hn))

class CriticLSTM(nn.Module):
        # int, int, int
    def __init__(self, state_dim, hidden_dim, action_dim, alpha=0.0001):
        super(CriticLSTM, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.Softmax(action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), alpha)
        self.loss = nn.MSELoss()

    # state must be torch tensor of shape (time_steps, state_dim)
    # output shape is (time_steps, action_dim)
    def forward(self, state):
        h0 = torch.zeros(1, state.shape[0], self.hidden_dim)
        c0 = torch.zeros(1, state.shape[0], self.hidden_dim)
        ht, (hn, cn) = self.lstm(state, (h0, c0))     # ht will be size (time_steps, hidden_dims)
        return self.activation(self.linear(hn))

class AdvantageFunction(nn.Module):
    # int, int, int
    def __init__(self, state_dim, action_dim, alpha=0.0001):
        super(AdvantageFunction, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.Softmax(action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), alpha)
        self.loss = nn.CrossEntropyLoss()

    # state must be torch tensor of shape (time_steps, state_dim)
    # output shape is (time_steps, action_dim)
    def forward(self, state):
        h0 = torch.zeros(1, state.shape[0], self.hidden_dim)
        c0 = torch.zeros(1, state.shape[0], self.hidden_dim)
        ht, (hn, cn) = self.lstm(state, (h0, c0))     # ht will be size (time_steps, hidden_dims)
        return self.activation(self.linear(hn))

class A2C():
    
    def __init__(self):
        self.actor = ActorLSTM(STATE_DIM, HIDDEN_DIM, 2)
        self.critic = CriticLSTM(STATE_DIM, HIDDEN_DIM, 1)
        self.adv = AdvantageFunction(STATE_DIM, HIDDEN_DIM, 2)

    def train(self, epochs=2000, discount=0.95, alpha=0.0011):

        tau = None
        obs = None
        area = None
        car = None

        reward_per_100 = 0

        for e in range(epochs):

            #start = time.time_ns()
            
            if e % 100 == 0:
                print("Epoch:", e)

            env = 'dynamic_obstacle'
            # x = e % 2000
            # if x >= 500 and x < 1000:
            #     env = 'static_90'
            # elif x >= 1000 and x < 1500:
            #     env = 'static_180_goal_right'
            # elif x >= 1500 and x < 2000:
            #     env = 'static_90_goal_down'

            # perform rollout
            tau, obs, area, car, inputs, outputs = rollout(self.lstm, env=env, rolloutLimit=250, isTest=False)

            avg_loss = 0
            reward = 0
            
            for t, value in enumerate(tau):

                reward += tau[t][3]

                # get current network actions y_hat
                current_a = self.lstm(inputs[t])[0]
                chosen_a = outputs[t].unsqueeze(0) # class label

                # init optimizer
                self.lstm.optimizer.zero_grad()

                # calculate utility
                utility = 0
                for j in range(len(tau) - t):
                    utility = utility + discount**j * tau[t + j][3] # expected, discounted reward of future states

                # calculate loss given actual actions take y
                loss = utility * self.lstm.loss(current_a, chosen_a) * alpha / len(tau)
                avg_loss += loss
                loss.backward()

                # perform SGD (ascent since negative LL)
                self.lstm.optimizer.step()

            reward_per_100 += reward

            if e % 100 == 0:
                print("Loss:", avg_loss/len(tau))
                print("Avg Reward:", reward_per_100 / 100.0)
                reward_per_100 = 0
            if e != 0 and e % 1000 == 0:
                torch.save(self.lstm.state_dict(), 'model_' + str(e) + '_save.pt')

        torch.save(self.lstm.state_dict(), 'model_final.pt')

if __name__=="__main__":
    np.random.seed()
    torch.random.seed()
    pg = PG()
    pg.train(epochs=8000)

