import tensorflow as tf
import keras
import model
import align4.game as game
from mcts import MCTS
import numpy as np
import evaluate
import re
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

n_games=300
n_MCTS_search=200

network=model.ToeNetwork()
save_dir = 'toe_shallow_10/'

with open(save_dir+'model_change','r') as f:
    change_ind = [re.findall(r'\d+',l)[0] for l in f]

for k in range(3):
    responses=[]
    for i in change_ind:
        print('Going back to iteration\n',i)
        network.load_weights(save_dir+'best_cnn_'+str(i)+".h5")
        ckpt='best_cnn_'+str(i)+".h5"

        responses.append(evaluate.stupidityCheck(network,n_MCTS_search=n_MCTS_search,config=k))

    plot_ind = change_ind[:]
    if change_ind[-1]!=29:
        plot_ind+=[29]
        responses.append(responses[-1])
    responses=np.array(responses).T

    y=[i for i in range(9)]
    x=[int(i) for i in plot_ind]
    X,Y=np.meshgrid(x,y)
    fig = plt.figure(k)
    ax = plt.axes(projection='3d')
    colors =plt.cm.viridis( (Y-Y.min())/float((Y-Y.min()).max()) )
    ax.plot_surface(X, Y, responses,facecolors=colors,alpha=0.5)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Move')
    ax.set_zlabel('MCTS visits N')
    plt.tight_layout()
    plt.savefig('report/minimax_'+str(k)+'.png')
    plt.show()
