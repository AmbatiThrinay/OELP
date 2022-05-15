# Author : Ambati Thrinay Kumar Reddy
#  Deep Q Neural network Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, n_inputs, fc1_dims, fc2_dims, n_actions, seed):
        '''
        Deep Neural Network model

        params
        ======
            n_inputs (int): Dimensions of each state
            fc1_dims (int): Number of nodes in first hidden layer
            fc2_dims (int): Number of nodes in second hidden layer
            n_actions (int): Number of actions
            seed (int) : Random seed for random number generator
        '''
        super(DeepQNetwork, self).__init__() # intialize the nn.Module class

        # generator the same intial weights for the network
        self.seed = torch.manual_seed(seed)

        # Network layers
        self.fc1=nn.Linear(n_inputs, fc1_dims)
        self.fc2=nn.Linear(fc1_dims, fc2_dims)
        self.fc3=nn.Linear(fc2_dims, n_actions)

    def forward(self, state):
        '''
        Build the neural network that maps states(x) to action probabilities
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save(self, filepath):
        print(f'..saving checkpoint .. at {filepath}')
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        print(f'... loading checkpoint ... from {filepath}')
        self.load_state_dict(torch.load(filepath))

class DQN():
    def __init__(
        self, n_inputs, n_actions, lr, gamma, batch_size,
        seed, epsilon, target_update_rate = 100, eps_end=0.01, eps_dec=5e-4, mem_size=100_000):
        '''
        Deep Q Netwerk agent

        Params
        =======
            n_inputs (int): Dimensions of each state
            n_actions (int): Number of actions
            lr (float): Learning rate for neural network
            gamma (float): discount factor
            batch_size (int): Minibatch size for training neural network
            seed (int): Seed for random generator for intializing network weight
            epsilon (float): for epsilon greedy policy
            target_update_rate (int): Update rate of target network
            eps_end (float): Final annealed exploration rate
            eps_dec (float): Decay rate of the epsilon
            mem_size (int): Size of the replay buffer

        '''
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = list(range(n_actions))
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0 # memory counter - no of experiences considered
        self.learn_step_cntr = 0 # for updating the target network for every few steps
        self.replace_rate = target_update_rate

        # train on GPU if available
        self.device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

        self.Q_network_local = DeepQNetwork(n_inputs, 128, 128, n_actions, seed).to(self.device)
        self.Q_network_target = DeepQNetwork(n_inputs, 128, 128, n_actions, seed).to(self.device)

        self.optimizer = torch.optim.Adam(self.Q_network_local.parameters(), lr=lr) # Adam optimizer
        self.loss = nn.HuberLoss() # Huber loss

        # Replay memory
        self.state_memory = np.zeros((self.mem_size, n_inputs), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, n_inputs), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)

    def store_transition(self, state, action, reward, next_state, done):
        '''
        Add new experience in the buffer
        '''
        index = self.mem_cntr % self.mem_size # index of memory to be overwritten
        self.mem_cntr += 1

        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

    def choose_action(self, state):

        # epsilon greedy policy
        if np.random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            actions = self.Q_network_local.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
        
    def learn(self):

        # learning starts only after the memory has batch size of experience
        if self.mem_cntr < self.batch_size: return

        # Update the target network for every 'repace_rate' steps
        if self.learn_step_cntr % self.replace_rate == 0:
            self.Q_network_target.load_state_dict(self.Q_network_local.state_dict())
        
        # randomly select only the batch size experiences(not zeroes)
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem,self.batch_size,replace=False)

        batch_index = np.arange(self.batch_size,dtype=np.int32) 
        state_batch = torch.tensor(self.state_memory[batch]).to(self.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.device)   
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.device)
        action_batch = self.action_memory[batch]

        # Q predict from local network and Q target from target network for TD error
        q_pred = self.Q_network_local.forward(state_batch)[batch_index,action_batch]
        q_next =  self.Q_network_local.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next,dim=1)[0]

        # training the local neural network
        self.optimizer.zero_grad()
        loss = self.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.optimizer.step()
        self.learn_step_cntr += 1

        # annealing the exploratory rate(epsilon)
        if self.epsilon > self.eps_min: self.epsilon -= self.eps_dec
        else: self.epsilon = self.eps_min
    
    def save(self, filepath):
        self.Q_network_local.save(filepath+'_local.pth')
        self.Q_network_target.save(filepath+'_target.pth')

    def load(self, filepath):
        self.Q_network_local.load(filepath+'_local.pth')
        self.Q_network_target.load(filepath+'_target.pth')
