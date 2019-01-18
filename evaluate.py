import tensorflow as tf
import keras
import model
import tictactoe.game as game
from mcts import MCTS
import numpy as np
import keras.backend as K
import tdgammon
from player import RandomPlayer,TDPlayer,AlphaZeroPlayer,HumanPlayer,AlphaGreedyPlayer,AlphaNetPlayer
from keras.callbacks import EarlyStopping

def duel(player_koth,player_chall,n_games=30,draw=False,loud=True):

    challenger_wins = 0
    koth_wins=0

    for i in range(n_games):
        player_koth.reset()
        player_chall.reset()
        fight_result= fight(player_koth,player_chall,first_player=(-1)**i)

        if  fight_result < 0:
            challenger_wins+=1
            if loud:
                print('One for the challenger!')

        elif fight_result > 0:
            if loud:
                print("Somethingsomethingaboutoldginger")
            koth_wins+=1

        else:
            if loud:
                print("And... Draw. Talk about anti climactic")
    if koth_wins + challenger_wins  == 0:
        wr = 1
    else:
        wr = challenger_wins/(koth_wins+challenger_wins)
    if draw:
        return wr,n_games-(koth_wins+challenger_wins)
    return wr

def fight(player1,player2,first_player=1):
    s=game.startState()

    while True:
        player = game.getCurrentPlayer(s)
        if (first_player*player) >0:
            a=player1.getMove(s)
        else:
            a=player2.getMove(s)

        s = game.nextState(s,a)
        if game.isEnded(s):
            return first_player*game.getWinner(s)

def sanityCheck(network,n_MCTS_search=25,n_games=30,opponent='random',exp=False,koth_ckpt='koth_net.h5',save_dir='toe_dense/'):
    if opponent == 'random':
        bench_player = RandomPlayer()
    elif opponent == 'gammon':
        bench_player = TDPlayer()
    elif opponent == 'notrain':
        bench_player = AlphaZeroPlayer(network,save_dir+'best_cnn_0.h5',n_MCTS_search=n_MCTS_search,tau=0.5)
    elif opponent == 'greedy':
        bench_player = AlphaGreedyPlayer(network,save_dir+koth_ckpt)
    elif opponent == 'net':
        bench_player = AlphaNetPlayer(network,save_dir+koth_ckpt)
    else:
        raise

    alpha_player = AlphaZeroPlayer(network,save_dir+koth_ckpt,n_MCTS_search=n_MCTS_search,tau=0.5)

    wr, draws = duel(bench_player,alpha_player,n_games=n_games,loud=False,draw=True)

    if exp:
        save_dir += "exp_"
    f=open(save_dir+opponent,'a')
    f.write('Network\'s winrate was '+str(wr)+" with "+str(draws)+" drawn \n")
    print('Network\'s winrate was '+str(wr)+" with "+str(draws)+" drawn \n")
    f.close()
    return

def stupidityCheck(net,n_MCTS_search=25,config=0):

    mcts = MCTS()
    s=game.startState()
    if config == 0:
        s=game.nextState(s,3)
        s=game.nextState(s,0)
        s=game.nextState(s,1)
    elif config == 1:
        s=game.nextState(s,3)
        s=game.nextState(s,0)
        s=game.nextState(s,1)
        s=game.nextState(s,6)
    else:
        s=game.nextState(s,6)
        s=game.nextState(s,0)
        s=game.nextState(s,7)
    print(s)
    for _ in range(n_MCTS_search):
        mcts.search(s,net)
    Ns=[mcts.N_s_a[str(s),a] if a in game.validMoves(s) else 0 for a in range(9)]
    print(Ns)
    return Ns
