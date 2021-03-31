from concurrent.futures import ThreadPoolExecutor
import gym
import matplotlib.pyplot as plt
from math import tanh
import numpy as np
import os
import PIL
import threading
from time import sleep, time
from tqdm import tqdm
import torch
from torch.optim import Adam
import torchvision

from dataset import ModelDataset
from model import WorldModel, Actor, Critic, LossModel, ActorLoss, CriticLoss
# plt.ion()


### HYPERPARAMETERS ###
save_path = "save.chkpt"
env_name = #TODO
env = gym.make(env_name, frameskip = 4)
num_actions = env.action_space.n

batch = 64
L = 50 #seq len world training
history_size = 64
lr_world = 2e-4

H = 15 #imagination length
gamma = 0.995 #discount factor
lamb = 0.95 #lambda-target
lr_actor = 4e-5
lr_critic = 1e-4
target_interval = 100 #update interval for target critic

gradient_clipping = 100
adam_eps = 1e-5
decay = 1e-6


### MODELS ###
world = WorldModel(gamma, num_actions).cuda()
actor = Actor(num_actions).cuda()
critic = Critic().cuda()
target = Critic().cuda()

criterionModel = LossModel()
criterionActor = ActorLoss()
criterionCritic = CriticLoss()

optim_model = Adam(world.parameters(), lr=lr_world, eps=adam_eps, weight_decay=decay)
optim_actor = Adam(actor.parameters(), lr=lr_actor, eps=adam_eps, weight_decay=decay)
optim_critic = Adam(critic.parameters(), lr=lr_critic, eps=adam_eps, weight_decay=decay)
optim_target = Adam(target.parameters())

if os.path.isfile(save_path):
    w = torch.load(save_path)
    try:
        world.load_state_dict(w["world"])
        optim_model.load_state_dict(w["optim_model"])
        actor.load_state_dict(w["actor"])
        optim_actor.load_state_dict(w["optim_actor"])
        critic.load_state_dict(w["critic"])
        optim_critic.load_state_dict(w["optim_critic"])
        criterionActor = ActorLoss(*w["criterionActor"])
    except:
        print("error loading model")
        world = WorldModel(gamma, num_actions).cuda()
        actor = Actor(num_actions).cuda()
        critic = Critic().cuda()
        target = Critic().cuda()
        criterionModel = LossModel()
        criterionActor = ActorLoss()
        criterionCritic = CriticLoss()
        optim_model = Adam(world.parameters(), lr=lr_world, eps=adam_eps, weight_decay=decay)
        optim_actor = Adam(actor.parameters(), lr=lr_actor, eps=adam_eps, weight_decay=decay)
        optim_critic = Adam(critic.parameters(), lr=lr_critic, eps=adam_eps, weight_decay=decay)
        optim_target = Adam(target.parameters())
    del w
with torch.no_grad():
    target.load_state_dict(critic.state_dict())


### MISC ###
resize = torchvision.transforms.Resize(
    (64,64),
    interpolation=PIL.Image.BICUBIC
)
grayscale = torchvision.transforms.Grayscale()
def transform_obs(obs):
    obs = resize(
        torch.from_numpy(obs.transpose(2,0,1))
    ) #3, 64, 64
    return (obs.float() - 255/2).unsqueeze(0)
history = []
tensor_range = torch.arange(0, num_actions).unsqueeze(0)
random_action_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
    torch.ones((1,num_actions))
)

def gather_episode():
    with torch.no_grad():
        while True:
            obs = env.reset()
            obs = transform_obs(obs)
            episode = [obs]
            obs = obs.cuda()
            z_sample, h = world(None, obs, None, inference=True)
            done = False
            while not done:
                # env.render()
                a = actor(z_sample)
                a = torch.distributions.one_hot_categorical.OneHotCategorical(
                    logits = a
                ).sample()
                obs, rew, done, _ = env.step(int((a.cpu()*tensor_range).sum().round())) # take a random action (int)
                obs = transform_obs(obs)
                obs = obs.cuda()
                episode.extend([a.cpu(), tanh(rew), done, obs.cpu()])
                if not done:
                    z_sample, h = world(a, obs, z_sample.reshape(-1, 1024), h, inference=True)
                # plt.imshow(obs[0].cpu().numpy().transpose(1,2,0)/2+0.5)
                # plt.show()
            history.append(episode)
            for _ in range(len(history) - history_size):
                history.pop(0)

#start gathering episode thread
t = threading.Thread(target=gather_episode)
t.start()

print("Dataset init")
while len(history) < 1:
    pass
print("done")

def act_straight_through(z_hat_sample):
    """
    IN:
        z_hat_sample
    Out:
        a_sample: with straight through gradient
        a_logits
    """
    #TODO
    return a_sample, a_logits

### DATASET ###
ds = ModelDataset(history, seq_len=L, gamma=gamma, history_size=history_size)
loader = torch.utils.data.DataLoader(
    ds,
    batch_size=batch,
    shuffle=True,
    num_workers=0,
    drop_last=False,
    pin_memory=True
)

iternum = 0
start = time()
while True:
    pbar = tqdm(loader)
    for s, a, r, g in pbar:
        s = s.cuda()
        a = a.cuda()
        r = r.cuda()
        g = g.cuda()
        z_list = []
        h_list = []

        ### Train world model ###
        #TODO

        ### Train actor critic ###
        #store every value to compute V since we sum backwards
        r_hat_sample_list = []
        gamma_hat_sample_list = []
        a_sample_list = []
        a_logits_list = []

        z_hat_sample = torch.cat(z_list, dim=0).detach() #convert all z to z0, squash time dim
        z_hat_sample_list = [z_hat_sample]

        h = torch.cat(h_list, dim=0).detach() #get corresponding h0
        
        # store values
        #TODO

        # calculate paper recursion by looping backward
        ## start with last timestep (t=H in paper)
        #TODO
        ## do next stuff backward
        #TODO

        #update actor
        loss_actor.backward()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clipping)
        optim_actor.step()
        optim_actor.zero_grad()
        optim_model.zero_grad()

        #update critic
        torch.nn.utils.clip_grad_norm_(critic.parameters(), gradient_clipping)
        optim_critic.step()
        optim_critic.zero_grad()
        optim_target.zero_grad()

        #update target network with critic weights
        iternum += 1
        if not iternum % target_interval:
            with torch.no_grad():
                target.load_state_dict(critic.state_dict())
        
        # display
        pbar.set_postfix(
            l_world = loss_model.item(),
            l_actor = loss_actor.item(),
            l_critic = loss_critic.item(),
            len_h = len(history),
            iternum=iternum,
            last_rew=sum(history[-1][2::4]),
        )
        print(a_logits_list[0][0].detach())
        print(list(z_hat_sample_list[-1][0,1].detach().cpu().numpy().round()).index(1))

    #save once in a while
    if time() - start > 1*60:
        start = time()
        print("Saving...")
        torch.save(
            {
                "world":world.state_dict(),
                "actor":actor.state_dict(),
                "critic":critic.state_dict(),
                "optim_model":optim_model.state_dict(),
                "optim_actor":optim_actor.state_dict(),
                "optim_critic":optim_critic.state_dict(),
                "criterionActor": (criterionActor.ns, criterionActor.nd, criterionActor.ne),
            },
            save_path
        )
        print("...done")
        plt.imsave("img.png", np.clip((x_hat[0].detach().cpu().numpy().transpose(1,2,0))/255+0.5, 0, 1))

    # plt.figure(1)
    # # plt.clf()
    # plt.imshow(x_hat[0].detach().cpu().numpy().transpose(1,2,0), cmap='gray')
    # # plt.pause(0.001)
    # plt.show()

env.close()
exit()
