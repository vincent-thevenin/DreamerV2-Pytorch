import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ReplayQueue():
    def __init__(self, capacity):
        self.capacity = capacity

        self.memory = []
        self.num_steps = 0
        self.smallest_episode_length = 0

    def push(self, episode):
        self.memory.append(episode)
        num_steps = (len(episode) - 1) // 4
        self.num_steps += num_steps

        if self.smallest_episode_length > num_steps:
            self.smallest_episode_length = num_steps

        find_new_smallest = self.smallest_episode_length == 0
        while self.num_steps > self.capacity:
            num_steps = (len(self.memory[0]) - 1) // 4
            self.num_steps -= num_steps
            del self.memory[0]
            if num_steps == self.smallest_episode_length:
                find_new_smallest = True
        
        if find_new_smallest:
            for e in self.memory:
                self.smallest_episode_length = min(self.smallest_episode_length, (len(e) - 1) // 4)

    def load(self, path):
        with open(path, 'rb') as f:
            self.memory = pickle.load(f)
        for i, e in enumerate(self.memory):
            num_steps = (len(e) - 1) // 4
            self.num_steps += num_steps
            if i == 0:
                self.smallest_episode_length = num_steps
            elif self.smallest_episode_length > num_steps:
                self.smallest_episode_length = num_steps

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def clear(self):
        self.memory = []
        self.num_steps = 0
        self.smallest_episode_length = 0


    def __len__(self):
        #retrieve number of steps
        return self.num_steps

class ModelDataset(Dataset):
    def __init__(self, replay, seq_len, gamma):
        self.gamma = gamma
        self.replay = replay #replay is updated outside
        self.seq_len = seq_len

    def __len__(self):
        return len(self.replay.memory)

    def __getitem__(self, idx):
        episode = self.replay.memory[idx]

        # idx_sample = np.random.randint(0, (len(episode)-1)//4) #sample random part of episode
        # idx_sample = max(0, min(idx_sample, (len(episode)-1)//4-seq_len)) #clip to not exceed limit
        seq_len = min(self.seq_len, self.replay.smallest_episode_length)
        if (len(episode)-1)//4 == seq_len:
            idx_sample = 0
        else:
            idx_sample = np.random.randint(0, (len(episode)-1)//4 - seq_len)

        states = episode[idx_sample*4 : (idx_sample+1)*4+seq_len*4 : 4]
        actions= episode[idx_sample*4+1 : idx_sample*4+seq_len*4+1 : 4]
        rewards= episode[idx_sample*4+2 : idx_sample*4+seq_len*4+2 : 4]
        gammas = episode[idx_sample*4+3 : idx_sample*4+seq_len*4+3 : 4]

        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.Tensor(rewards)
        gammas = (-torch.Tensor(gammas) + 1)*self.gamma #gamma if not done else 0 but with tensors

        return states, actions, rewards.unsqueeze(1), gammas.unsqueeze(1)