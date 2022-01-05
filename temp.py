import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
import pickle
from brokenaxes import brokenaxes
import os


N = 2**14
plt.style.use('fivethirtyeight')
figsize = (
    11 * 0.78,
    8  * 0.78
)
fig = plt.figure(figsize=figsize, facecolor="white")
# ax = fig.add_subplot(1, 1, 1)




# with open("average_step_sizeACP.pkl", "rb") as f:
#     acp = pickle.load(f)

# with open("reward_listDREAMER.pkl", "rb") as f:
#     dre = pickle.load(f)

reward_paths = [path for path in os.listdir() if "reward_listDREAMER_ksparse_K" in path]
dre = []
for path in reward_paths:
    n = len(os.listdir(path))
    dre.append([])
    for i in range(n):
        with open(os.path.join(path, str(i) + '.pkl'), 'rb') as f:
            dre[-1].extend(pickle.load(f))

# with open("reward_listDREAMER_ksparse.pkl", "rb") as f:
#     drek = pickle.load(f)

# with open("reward_listDREAMER_ksparse2.pkl", "rb") as f:
#     drek2 = pickle.load(f)

def get_K(path):
    # return path.split('_')[-2] + '_' + path.split('_')[-1]
    return path.split('_')[-2].split('K')[-1]

r_dict = {}
for path, dre_ in zip(reward_paths, dre):
    if get_K(path) not in r_dict:
        r_dict[get_K(path)] = np.array(dre_)[None, :]
    else:
        min_l = min(r_dict[get_K(path)].shape[1], len(dre_))
        r_dict[get_K(path)] = np.concatenate(
            (r_dict[get_K(path)][:, :min_l], np.array(dre_)[None, :min_l]),
            axis=0
        )

r_mean = {}
for k, v in r_dict.items():
    r_mean[k] = np.mean(v, axis=0)

r_var = {}
for k, v in r_dict.items():
    r_var[k] = np.var(v, axis=0)



min_l = min([len(r) for r in dre])
# acp = uniform_filter1d(     acp[-1][:min_l], size=N)
# dre = uniform_filter1d(     dre[    :min_l], size=N)
# drek2 = uniform_filter1d(   drek2[  :min_l], size=N)
# drek = uniform_filter1d(    drek[:min_l], size=N)
#
# dre_filtered = []
# for r in dre:
#     dre_filtered.append(uniform_filter1d(r[:min_l], size=N))
for k, v in r_mean.items():
    r_mean[k] = uniform_filter1d(v[:min_l], size=N)
for k, v in r_var.items():
    r_var[k] = uniform_filter1d(v[:min_l], size=N)



bax = brokenaxes(xlims=((0, 10000), (10000 , min_l)), hspace=.05)

# bax.plot(acp, label="Actor-critic with pathwise method")
# bax.plot(dre, label="DreamerV2")
# bax.plot(drek, label="DreamerV2 k-sparse")
for k, v in r_mean.items():
    bax.plot(v, label="K"+str(k))
    plt.fill_between(np.arange(len(v)), r_mean[k] - np.sqrt(r_var[k]), r_mean[k] + np.sqrt(r_var[k]), alpha=0.2)

# bax.legend(["Actor-critic with pathwise method", "DreamerV2", "DreamerV2 k-sparse", "DreamerV2 k-sparse2"])
bax.legend([r.split('_')[-2]+'-'+r.split('_')[-1] for r in reward_paths])
bax.legend(loc=2)

bax.set_xlabel("Episode", weight = 'bold')
bax.set_ylabel("Total reward per episode", weight = 'bold')

plt.show()
plt.clf()

print()
