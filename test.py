import tensorflow as tf
import keras
import model
import align4.game as game
from mcts import MCTS
import numpy as np
import evaluate
import re

n_games=300
n_MCTS_search=15

network=model.ToeDeepNetwork()
save_dir = 'toe_deep_narrow/'

with open(save_dir+'model_change','r') as f:
    change_ind = [re.findall(r'\d+',l)[0] for l in f]

for i in change_ind:
    print('Going back to iteration\n',i)
    network.load_weights(save_dir+'best_cnn_'+str(i)+".h5")
    ckpt='best_cnn_'+str(i)+".h5"
    print('Face... The raaaaannnnndddddooooooooooooommmmmmmmmmm')
    evaluate.sanityCheck(network,n_games = n_games,n_MCTS_search=n_MCTS_search,koth_ckpt=ckpt,exp=True,save_dir=save_dir)

    print('Meet the gammon')
    evaluate.sanityCheck(network,n_games = n_games,n_MCTS_search=n_MCTS_search,opponent='gammon',koth_ckpt=ckpt,exp=True,save_dir=save_dir)

    print('Yes, you sucked that much')
    evaluate.sanityCheck(network,n_games = n_games//10,n_MCTS_search=n_MCTS_search,opponent='notrain',koth_ckpt=ckpt,exp=True,save_dir=save_dir)

    print('Do not be greedy')
    evaluate.sanityCheck(network,n_games = n_games,n_MCTS_search=n_MCTS_search,opponent='greedy',koth_ckpt=ckpt,exp=True,save_dir=save_dir)

    print('Be grateful you have MCTS')
    evaluate.sanityCheck(network,n_games = n_games,n_MCTS_search=n_MCTS_search,opponent='net',koth_ckpt=ckpt,exp=True,save_dir=save_dir)
