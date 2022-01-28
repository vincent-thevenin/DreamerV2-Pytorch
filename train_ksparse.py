from concurrent.futures import ThreadPoolExecutor
import copy
import gym
import gym.wrappers
from gym import wrappers
import gym.envs.atari
import matplotlib.pyplot as plt
from math import tanh
import numpy as np
import os
import pickle
import PIL
from pyvirtualdisplay import Display
import threading
from time import sleep, time
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torchvision

from dataset import ModelDataset, ReplayQueue
from model_accurate import WorldModel, Actor, Critic, LossModel, ActorLoss, CriticLoss

class Timer():
    def __init__(self, name, verbose=False):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time()

    def __exit__(self, *args):
        self.end = time()
        if self.verbose:
            print(f'Time {self.name}: {self.end - self.start}s')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

K_list = [1, 4, 8, 16, 32]
experiment_id = 0
experiment_type = ""  # "", "Noise", "SampleAverage"
noise_std = 0.01

for K in K_list:
    ### HYPERPARAMETERS ###
    save_path =        f"save_ksparse_K{K}_{experiment_type}{experiment_id}.chkpt"
    reward_path =      f"reward_listDREAMER_ksparse_K{K}_{experiment_type}{experiment_id}/.pkl"
    tensorboard_path = f"runs/K{K}_{experiment_type}{experiment_id}"
    env_name =         f"CartPole-v0"
    prefill_path =     f"prefill_{env_name}.pkl"
    action_repeat = 4
    noop = 30
    screen_size = 64
    life_done = False
    grayscale = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=True)

    writer = SummaryWriter(tensorboard_path)
    env = gym.make(env_name)
    is_atari = isinstance(env.unwrapped, gym.envs.atari.environment.AtariEnv)
    if is_atari:
        # setup environment for atari
        env = gym.make(
                id=env_name,
                mode=None,
                difficulty=None,
                obs_type='grayscale' if grayscale else 'rgb',
                frameskip=1,
                repeat_action_probability=0.25,
                full_action_space=False,
                render_mode="rgb_array",
        )
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None
        # # Tell wrapper that the inner env has no action repeat.
        env.env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
        env = gym.wrappers.AtariPreprocessing(
            env, noop, action_repeat, screen_size, life_done, grayscale
        )
    else:
        # create fake display
        disp = Display().start()

    num_actions = env.action_space.n

    batch = 50
    L = 50 #seq len world training
    prefill_steps = 50_000
    replay_capacity_steps = max(2e4, prefill_steps) #1e5
    max_steps = 200_000_000//action_repeat if is_atari else 2_000_000
    lr_world = 2e-4
    train_every = 16
    K = K #ksparse
    H = 15 #imagination length
    gamma = 0.999 #discount factor
    lamb = 0.95 #lambda-target
    lr_actor = 4e-5
    lr_critic = 1e-4
    target_interval = 100 #update interval for target critic

    gradient_clipping = 100
    adam_eps = 1e-5
    decay = 1e-6


    ### MODELS ###
    world = WorldModel(
        gamma,
        num_actions,
        has_noise="Noise" in experiment_type,
	    noise_std=noise_std,
        is_sample_average="SampleAverage" in experiment_type,
    ).to(device)
    actor = Actor(num_actions).to(device)
    critic = Critic().to(device)
    target = Critic().to(device)

    print(f"World: {count_parameters(world):,} parameters")
    print(f"Actor: {count_parameters(actor):,} parameters")
    print(f"Critic: {count_parameters(critic):,} parameters")

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
            world = WorldModel(gamma, num_actions).to(device)
            actor = Actor(num_actions).to(device)
            critic = Critic().to(device)
            target = Critic().to(device)
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
    if not os.path.isdir(reward_path.split('/')[0]):
        os.mkdir(reward_path.split('/')[0])

    resize = torchvision.transforms.Resize(
        (screen_size, screen_size),
        interpolation=PIL.Image.BICUBIC
    )
    grayscale = torchvision.transforms.Grayscale()
    def transform_obs(obs):
        if len(obs.shape) == 2:  # already grayscale
            obs =torch.from_numpy(obs).unsqueeze(0)
        else:
            obs = grayscale(
                    torch.from_numpy(obs.transpose(2,0,1))
                ) #1, H, W
        obs = resize(obs)
        return (obs.float() / 255 - 0.5).unsqueeze(0)  # 1, C, H, W

    episode = []
    replay = ReplayQueue(capacity=replay_capacity_steps)
    if os.path.isfile(prefill_path):
        replay.load(prefill_path)
        
    tensor_range = torch.arange(0, num_actions).unsqueeze(0)
    step_counter = [replay.num_steps]

    def gather_step(done, z_sample, h, k, train_every=1):
        with torch.no_grad():
            for _ in range(train_every):
                if not done: #while not done:
                    a = actor(z_sample)
                    a = torch.distributions.one_hot_categorical.OneHotCategorical(
                        logits = a
                    ).sample()
                    # while max(0, (step_counter[0] - prefill_steps)) // train_every > train_step_list[0]:
                    #     sleep(0.1)
                    obs, rew, done, _ = env.step(int((a.cpu()*tensor_range).sum().round())) # take a random action (int)
                    rew_list[-1].append(rew)

                    if not is_atari:
                        obs = env.render(mode="rgb_array")
                    obs = transform_obs(obs.copy())

                    step_counter[0] += 1
                    episode.extend([a.cpu(), tanh(rew), done, obs])
                    if not done:
                        z_sample, h = world(a, obs.to(device), z_sample.reshape(-1, 1024), h, inference=True, k=k)

                    # plt.imshow(obs[0].cpu().numpy().transpose(1,2,0)/2+0.5)
                    # plt.show()
                else:
                    rew_list.append([])

                    obs = env.reset()
                    if not is_atari:
                        obs = env.render(mode="rgb_array")
                    obs = transform_obs(obs.copy())

                    replay.push(copy.deepcopy(episode)) if len(episode) > 0 else None
                    episode.clear()
                    episode.append(obs)
                    step_counter[0] += 1
                    
                    z_sample, h = world(None, obs.to(device), None, inference=True, k=k)
                    done = False
            
            return done, z_sample, h

    print("Dataset init")
    pbar = tqdm(total=replay.num_steps)
    done = True
    z_sample_env, h_env = None, None
    rew_list = []
    while replay.num_steps < prefill_steps:
        done, z_sample_env, h_env = gather_step(done, z_sample_env, h_env, k=1)
        pbar.set_postfix(
            total=replay.num_steps,
        )
    print("done")

    if not os.path.isfile(prefill_path):
        replay.save(prefill_path)

    # from copy import deepcopy
    # pbar = tqdm()
    # hs = get_history_steps(history)
    # hshort = deepcopy(history)
    # while  hs < replay_capacity_steps:
    #     history += deepcopy(hshort)
    #     hs = get_history_steps(history)
    #     pbar.set_postfix(hs = hs)


    def act_straight_through(z_hat_sample):
        a_logits = actor(z_hat_sample)
        a_sample = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=a_logits
        ).sample()
        a_probs = torch.softmax(a_logits, dim=-1)
        a_sample = a_sample + a_probs - a_probs.detach()

        return a_sample, a_logits

    def dict_sum(d1, d2):
        """sum dict entries"""
        for k in d1:
            d1[k] += d2[k]

    ### DATASET ###
    ds = ModelDataset(replay, seq_len=L, gamma=gamma)
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
    while step_counter[0] < max_steps:
        pbar = tqdm(loader)
        for s, a, r, g in pbar:
            done, z_sample_env, h_env = gather_step(
                done,
                z_sample_env,
                h_env,
                K,
                train_every=train_every
            )
            
            s = s.to(device)
            a = a.to(device)
            r = r.to(device)
            g = g.to(device)
            K = max(round(K * (1 - step_counter[0] / max_steps)), 1)

            ### Train world model ###
            with Timer("Total"):
                with autocast(enabled=True):
                    with Timer("Forward"):
                        z_logit, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, _ = world(
                            a=a,
                            x=s,
                            z=None,
                            k= K, # - int(step_counter[0] // (max_steps/K)),
                            h=None
                        )
                    with Timer("loss model"):
                        loss_model, loss_dict_model = criterionModel(
                            s[:,0],
                            r[:,0], #false r_0 does not exist, 0: t=1 but expect 0:t=0
                            g[:,0], #same but ok since never end of episode
                            x_hat[:, 0],
                            0, #rhat
                            gamma_hat[:, 0],
                            z_logit[:, 0],
                            z_hat_logits[:, 0],
                        )
                        _loss_model, _loss_dict_model = criterionModel(
                            s[:, 1:a.shape[1] + 1].reshape(-1, *s.shape[2:]),
                            r[:, :a.shape[1]].reshape(-1, *r.shape[2:]), #r time array starts at 1; 0: t=1
                            g[:, :a.shape[1]].reshape(-1, *g.shape[2:]), #g time array starts at 1; 0: t=1
                            x_hat[:, 1:a.shape[1] + 1].reshape(-1, *x_hat.shape[2:]),
                            r_hat[:, 1:a.shape[1] + 1].reshape(-1, *r_hat.shape[2:]),
                            gamma_hat[:, 1:a.shape[1] + 1].reshape(-1, *gamma_hat.shape[2:]),
                            z_logit[:, 1:a.shape[1] + 1].reshape(-1, *z_logit.shape[2:]),
                            z_hat_logits[:, 1:a.shape[1] + 1].reshape(-1, *z_hat_logits.shape[2:]),
                        )
                        loss_model += _loss_model
                        dict_sum(loss_dict_model, _loss_dict_model)
                            

                        # print([list(z_list[_][0,1].detach().cpu().numpy().round()).index(1) for _ in range(len(z_list))])
                        for k in loss_dict_model:
                            loss_dict_model[k] /= a.shape[1]
                        writer.add_scalars('loss_model', loss_dict_model, iternum)
                        loss_model /= a.shape[1]

                with Timer("backward model"):
                    scaler.scale(loss_model).backward()
                with Timer("optim model"):
                    scaler.unscale_(optim_model)
                    torch.nn.utils.clip_grad_norm_(world.parameters(), gradient_clipping)
                    scaler.step(optim_model)
                    scaler.update()
                    # optim_model.zero_grad()

            ### Train actor critic ###
            with Timer("Actor Critic", True):
                if step_counter[0] > 0:
                    with autocast(enabled=True):
                        #store every value to compute V since we sum backwards
                        r_hat_sample_list = []
                        gamma_hat_sample_list = []
                        a_sample_list = []
                        a_logits_list = []

                        z_hat_sample = z_sample.reshape(-1, *z_sample.shape[2:]).detach() #convert all z to z0, squash time dim
                        z_hat_sample_list = [z_hat_sample]

                        h = h.reshape(-1, *h.shape[2:]).detach() #get corresponding h0
                        
                        # store values
                        for _ in range(H):
                            a_sample, a_logits = act_straight_through(z_hat_sample)

                            *_, h, (z_hat_sample, r_hat_sample, gamma_hat_sample) = world(
                                a_sample,
                                x = None,
                                z = z_hat_sample.reshape(-1, 1024),
                                k = K,
                                h = h,
                                dream=True
                            )
                            r_hat_sample_list.append(r_hat_sample)
                            gamma_hat_sample_list.append(gamma_hat_sample)
                            z_hat_sample_list.append(z_hat_sample)
                            a_sample_list.append(a_sample)
                            a_logits_list.append(a_logits)

                        # calculate paper recursion by looping backward
                        # calculate all targets
                        targets = target(torch.cat(z_hat_sample_list[1:], dim=0))
                        targets = targets.chunk(len(z_hat_sample_list) - 1, dim=0)

                        ves = critic(torch.cat(z_hat_sample_list[:-1], dim=0).detach())
                        ves = ves.chunk(len(z_hat_sample_list) - 1, dim=0)


                        # V = r_hat_sample_list[-1] + gamma_hat_sample_list[-1] * target(z_hat_sample_list[-1]) #V_H-1
                        # ve = critic(z_hat_sample_list[-2].detach())
                        # loss_critic = criterionCritic(V.detach(), ve)
                        # loss_actor = criterionActor(
                        #     a_sample_list[-1],
                        #     torch.distributions.one_hot_categorical.OneHotCategorical(
                        #         logits=a_logits_list[-1]
                        #     ),
                        #     V,
                        #     ve.detach()
                        # )
                        # for t in range(H-2, -1, -1):
                        #     V = r_hat_sample_list[t] + gamma_hat_sample_list[t] * ((1-lamb)*target(z_hat_sample_list[t+1]) + lamb*V)
                        #     ve = critic(z_hat_sample_list[t].detach())
                        #     loss_critic += criterionCritic(V.detach(), ve)
                        #     loss_actor += criterionActor(
                        #         a_sample_list[t],
                        #         torch.distributions.one_hot_categorical.OneHotCategorical(
                        #             logits=a_logits_list[t]
                        #         ),
                        #         V,
                        #         ve.detach()
                        #     )

                        V = r_hat_sample_list[-1] + gamma_hat_sample_list[-1] * targets[-1] #V_H-1
                        loss_critic, loss_dict_critic = criterionCritic(V.detach(), ves[-1])
                        loss_actor, loss_dict_actor = criterionActor(
                            a_sample_list[-1],
                            torch.distributions.one_hot_categorical.OneHotCategorical(
                                logits=a_logits_list[-1]
                            ),
                            V,
                            ves[-1].detach()
                        )
                        c = zip(reversed(a_sample_list[:-1]), reversed(a_logits_list[:-1]), reversed(r_hat_sample_list[:-1]), reversed(gamma_hat_sample_list[:-1]), reversed(targets[:-1]), reversed(ves[:-1]))
                        for a_sample, a_logits, r_hat_sample, gamma_hat_sample, target_v, ve in c:
                            V = r_hat_sample + gamma_hat_sample * ((1-lamb)*target_v + lamb*V)
                            _loss_critic, _loss_dict_critic = criterionCritic(V.detach(), ve)
                            _loss_actor, _loss_dict_actor = criterionActor(
                                a_sample,
                                torch.distributions.one_hot_categorical.OneHotCategorical(
                                    logits=a_logits
                                ),
                                V,
                                ve.detach()
                            )
                            loss_actor += _loss_actor
                            loss_critic += _loss_critic
                            dict_sum(loss_dict_actor, _loss_dict_actor)
                            dict_sum(loss_dict_critic, _loss_dict_critic)


                        loss_actor /= H
                        loss_critic /= H

                        for k in loss_dict_actor:
                            loss_dict_actor[k] /= H
                        writer.add_scalars('loss_actor', loss_dict_actor, iternum)
                        for k in loss_dict_critic:
                            loss_dict_critic[k] /= H
                        writer.add_scalars('loss_critic', loss_dict_critic, iternum)

                    #update actor
                    scaler.scale(loss_actor).backward()
                    # loss_actor.backward()
                    scaler.unscale_(optim_actor)
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), gradient_clipping)
                    scaler.step(optim_actor)
                    # optim_actor.step()
                    optim_actor.zero_grad()
                    optim_model.zero_grad()

                    #update critic
                    scaler.scale(loss_critic).backward()
                    # loss_critic.backward()
                    scaler.unscale_(optim_critic)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), gradient_clipping)
                    scaler.step(optim_critic)
                    # optim_critic.step()
                    optim_critic.zero_grad()
                    optim_target.zero_grad()
                    scaler.update()

                    #clear all
                    r_hat_sample_list.clear()
                    gamma_hat_sample_list.clear()
                    z_hat_sample_list.clear()
                    a_sample_list.clear()
                    a_logits_list.clear()

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
                        len_h_e = len(replay.memory),
                        iternum=iternum,
                        last_rew=sum(replay.memory[-1][2::4]),
                        steps=step_counter[0],
                    )
                    # print(a_logits_list[0][0].detach())
            # print([list(z_hat_sample_list[_][0,1].detach().cpu().numpy().round()).index(1) for _ in range(len(z_hat_sample_list))])
            # print([(z_hat_sample_list[_][0] == z_hat_sample_list[_+1][0]).float().mean().item() for _ in range(len(z_hat_sample_list)-1)])

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
            with open(reward_path.split('.')[0] + str(len(os.listdir(reward_path.split('/')[0]))) + '.' + reward_path.split('.')[1], "wb") as f:
                pickle.dump(rew_list[:-1], f)
            rew_list = rew_list[-1:]
            print("...done")
            plt.imsave(
                "img.png",
                np.clip(
                    x_hat[0, 0].detach().expand(3, -1, -1).cpu().numpy().transpose(1,2,0) / 2 + 0.25,
                    0,
                    1
                )
            )

            plt.imsave(
                "img_fake.png",
                np.clip(
                    world.compute_x_hat(torch.cat((h, z_hat_sample.reshape(-1, 1024)), dim=1))[0].detach().expand(3, -1, -1).cpu().numpy().transpose(1,2,0) / 2 + 0.25,
                    0,
                    1
                )
            )

        # plt.figure(1)
        # # plt.clf()
        # plt.imshow(x_hat[0].detach().cpu().numpy().transpose(1,2,0), cmap='gray')
        # # plt.pause(0.001)
        # plt.show()

    writer.close()
    env.close()

exit()

