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




with open("average_step_sizeACP.pkl", "rb") as f:
    acp = pickle.load(f)

with open("reward_listDREAMER.pkl", "rb") as f:
    dre = pickle.load(f)

reward_path = "reward_listDREAMER_ksparse_K16/"
dre = []
n = len(os.listdir(reward_path))
for i in range(n):
    with open(reward_path + str(i) + '.pkl', 'rb') as f:
        dre.extend(pickle.load(f))

with open("reward_listDREAMER_ksparse.pkl", "rb") as f:
    drek = pickle.load(f)

with open("reward_listDREAMER_ksparse2.pkl", "rb") as f:
    drek2 = pickle.load(f)


min_l = len(dre) #min([len(acp[-1]), len(dre)])#, len(drek2)])
acp = uniform_filter1d(     acp[-1][:min_l], size=N)
dre = uniform_filter1d(     dre[    :min_l], size=N)
drek2 = uniform_filter1d(   drek2[  :min_l], size=N)
drek = uniform_filter1d(    drek[:min_l], size=N)

bax = brokenaxes(xlims=((0, 10000), (10000 , min_l)), hspace=.05)

# bax.plot(acp, label="Actor-critic with pathwise method")
bax.plot(dre, label="DreamerV2")
# bax.plot(drek, label="DreamerV2 k-sparse")
# bax.legend(["Actor-critic with pathwise method", "DreamerV2", "DreamerV2 k-sparse", "DreamerV2 k-sparse2"])
bax.legend(loc=2)

bax.set_xlabel("Episode", weight = 'bold')
bax.set_ylabel("Total reward per episode", weight = 'bold')

plt.show()
plt.clf()

print()
