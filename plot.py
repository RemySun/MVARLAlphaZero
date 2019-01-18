import tensorflow as tf
import keras
import model
import align4.game as game
from mcts import MCTS
import numpy as np
import evaluate
import re
import matplotlib.pyplot as plt

n_games=300
n_MCTS_search=15


save_dir = 'toe_deep_narrow/'

with open(save_dir+'model_change','r') as f:
    change_ind = [int(re.findall(r'\d+',l)[0]) for l in f]

change_ind = [0]+change_ind+[29]
cutoff = len(change_ind)
baselines = ['random','gammon','notrain','net','greedy']

exps_wr = []
exps_dr = []
for baseline in baselines:
    with open(save_dir+'exp_'+baseline,'r') as f:
        exp_results = [l.split() for l in f]
        if baseline == 'notrain':
            exp_wr=[100*float(l[3])*(n_games/10-float(l[5]))/(n_games/10) for l in exp_results]
            exp_dr=[100*(float(l[3])*(n_games/10-float(l[5]))+float(l[5]))/n_games*10 for l in exp_results]

        else:
            exp_wr=[100*float(l[3])*(n_games-float(l[5]))/(n_games) for l in exp_results]
            exp_dr=[100*(float(l[3])*(n_games-float(l[5]))+float(l[5]))/n_games for l in exp_results]

        exps_wr.append(exp_wr[:1]+exp_wr+exp_wr[-1:])

        exps_dr.append(exp_dr[:1]+exp_dr+exp_dr[-1:])

colors = ['blue','orange','green','grey','black']
for i in range(len(baselines)):
    plt.plot(change_ind[:cutoff],exps_wr[i][:cutoff],label='win vs '+baselines[i],color=colors[i],marker='+')
    plt.plot(change_ind[:cutoff],exps_dr[i][:cutoff],linestyle='--',color=colors[i])

plt.plot(change_ind[:cutoff],np.ones(len(change_ind))*50,color='red')
legend=plt.legend(fancybox=True)
legend.get_frame().set_alpha(0.5)
plt.xlabel('Iteration')
plt.ylabel('Rate (%)')
plt.savefig('report/narrow_winrates.png')
plt.show()

