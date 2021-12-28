import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ModelDataset(Dataset):
    def __init__(self, history, seq_len, gamma):
        self.h = history #history is passed as list and updated outside
        self.seq_len = seq_len
        self.gamma = gamma

    def __len__(self):
        return len(self.h)

    def __getitem__(self, idx):
        idx = idx % (len(self.h) - 1) #do not exceed history length and do not take episode being gathered
        episode = self.h[idx]

        seq_len = self.seq_len
        for e in self.h[:-1]:
            seq_len = min(seq_len, (len(e) - 1) // 4)


        idx_sample = np.random.randint(0, (len(episode)-1)//4) #sample random part of episode
        idx_sample = max(0, min(idx_sample, (len(episode)-1)//4-seq_len)) #clip to not exceed limit

        states = episode[idx_sample*4 : (idx_sample+1)*4+seq_len*4 : 4]
        actions= episode[idx_sample*4+1 : idx_sample*4+seq_len*4+1 : 4]
        rewards= episode[idx_sample*4+2 : idx_sample*4+seq_len*4+2 : 4]
        gammas = episode[idx_sample*4+3 : idx_sample*4+seq_len*4+3 : 4]

        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.Tensor(rewards)
        gammas = (-torch.Tensor(gammas) + 1)*self.gamma #gamma if not done else 0 but with tensors

        return states, actions, rewards.unsqueeze(1), gammas.unsqueeze(1)