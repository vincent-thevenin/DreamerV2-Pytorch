import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
import pickle
from brokenaxes import brokenaxes
import os


### Parameters ###
N = 2**14
K_values = [1, 4, 8, 16, 32]
is_plot_var = True

### Load data ###
plt.style.use('fivethirtyeight')
figsize = (
    11 * 0.78,
    8  * 0.78
)
fig = plt.figure(figsize=figsize, facecolor="white")

def get_K(path):
    # path in format: 'reward_listDREAMER_ksparse_K{K}_{experiment_name}'
    return path.split('_')[-2].split('K')[-1]

def remove_digit_from_string(string):
    return ''.join([i for i in string if not i.isdigit()])

def get_K_and_category(path):
    # path in format: 'reward_listDREAMER_ksparse_K{K}_{experiment_name}'
    category = remove_digit_from_string(path.split('_')[-1])
    return path.split('_')[-2].split('K')[-1] + ''.join(['_' if len(category) > 1 else '', category])

def get_experiment_name(path):
    # path in format: 'reward_listDREAMER_ksparse_K{K}_{experiment_name}'
    return path.split('_')[-2][1:] + '_' + path.split('_')[-1]

reward_paths = [path for path in os.listdir() if "reward_listDREAMER_ksparse_K" in path and int(get_K(path)) in K_values]
dre = []
for path in reward_paths:
    n = len(os.listdir(path))
    dre.append([])
    for i in range(n):
        with open(os.path.join(path, str(i) + '.pkl'), 'rb') as f:
            dre[-1].extend(pickle.load(f))

get_name = get_K_and_category if is_plot_var else get_experiment_name
r_dict = {}

for path, dre_ in zip(reward_paths, dre):
    if get_name(path) not in r_dict:
        r_dict[get_name(path)] = np.array(dre_)[None, :]
    else:
        min_l = min(r_dict[get_name(path)].shape[1], len(dre_))
        r_dict[get_name(path)] = np.concatenate(
            (r_dict[get_name(path)][:, :min_l], np.array(dre_)[None, :min_l]),
            axis=0
        )

r_mean = {}
for k, v in r_dict.items():
    r_mean[k] = np.mean(v, axis=0)

r_var = {}
for k, v in r_dict.items():
    r_var[k] = np.var(v, axis=0)


max_l = max([len(r) for r in dre])
for k, v in r_mean.items():
    r_mean[k] = uniform_filter1d(v, size=N)
for k, v in r_var.items():
    r_var[k] = uniform_filter1d(v, size=N)



bax = brokenaxes()#xlims=((0, 10000), (10000 , max_l)), hspace=.05)

for k, v in r_mean.items():
    bax.plot(v, label="K"+str(k))
    plt.fill_between(np.arange(len(v)), r_mean[k] - np.sqrt(r_var[k]), r_mean[k] + np.sqrt(r_var[k]), alpha=0.2)

bax.legend([r.split('_')[-2]+'-'+r.split('_')[-1] for r in reward_paths])
bax.legend(loc=2)

bax.set_xlabel("Episode", weight = 'bold')
bax.set_ylabel("Total reward per episode", weight = 'bold')

plt.show()
plt.clf()

print()
