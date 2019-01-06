import tensorflow as tf
import keras
import model
import align4.game as game
from mcts import MCTS
import numpy as np

def policyIteration(n_iters=30,n_episodes=20):
    network = model.PredictorNetwork()
    network.save_weights('koth_net.h5')

    for iteration in range(n_iters):
        examples = []
        print('Training cycle',iteration)
        for episode in range(n_episodes):
            print('playing episode',episode)
            examples += playMatch(network)
        print("Training network")

        trainNetwork(network,examples)
        print('Finding out who the better network is')
        network.save_weights('challenger_net.h5')
        frac_win = duel(network,'koth_net.h5','challenger_net.h5')
        if frac_win > 0.5:
            print('The young will replace the old. Or something to that effect')
            network.load_weights('challenger_net.h5')
            network.save_weights('koth_net.h5')

    return network

def trainNetwork(network, examples):
    input_data = np.array([ex[0] for ex in examples])
    ground_truths = [np.array([ex[1] for ex in examples]),np.array([ex[2] for ex in examples])]
 
    network.fit(input_data,ground_truths,batch_size=10,epochs=60,shuffle=True)


def playMatch(network,n_MCTS_search=50):
    examples = []

    s=game.startState()


    while True:
        mcts = MCTS()
        for _ in range(n_MCTS_search):
            mcts.search(s,network)
        examples.append([s,mcts.computePi(s,network),None])
        a = np.random.choice(game.ACTIONS,p=mcts.computePi(s,network))

        s = game.nextState(s,a)
        if game.isEnded(s):
            for example in examples:
                example[-1] = example[0][-1]*game.getWinner(s)
            return examples

def duel(network,koth_ckpt,challenger_ckpt,n_MCTS_search=20,n_games=30):
    challenger_wins = 0
    koth_wins=0
    for i in range(n_games):
        fight_result= fight(network,koth_ckpt,challenger_ckpt,first_player=(-1)**i,n_MCTS_search=20)
        if  fight_result< 0:
            challenger_wins+=1
            print('One for the challenger!')
        elif fight_result > 0:
            print("Somethingsomethingaboutoldginger")
            koth_wins+=1
        else:
            print("And... Draw. Talk about anti climactic")
    return challenger_wins/(koth_wins+challenger_wins)

def fight(network,koth_ckpt,challenger_ckpt,first_player=1,n_MCTS_search=100):
    s=game.startState()

    while True:
        mcts_koth = MCTS()
        mcts_challenger = MCTS()
        player = s[-1]
        if (first_player*player) >0:
            mcts = mcts_koth
            network.load_weights(koth_ckpt)
        else:
            mcts = mcts_challenger
            network.load_weights(challenger_ckpt)
        for _ in range(n_MCTS_search):
            mcts.search(s,network)
        a = np.random.choice(game.ACTIONS,p=mcts.computePi(s,network))
        s = game.nextState(s,a)
        if game.isEnded(s):
            return first_player*game.getWinner(s)

def faceTheMachine(network,koth_ckpt,n_MCTS_search=100):
    s=game.startState()

    while True:
        mcts_koth = MCTS()
        mcts_challenger = MCTS()
        player = s[-1]
        if (player) >0:
            mcts = mcts_koth
            network.load_weights(koth_ckpt)
            for _ in range(n_MCTS_search):
                mcts.search(s,network)
            a = np.random.choice(game.ACTIONS,p=mcts.computePi(s,network))
        else:
            print(game.printFriendly(s))
            a=int(input())
        s = game.nextState(s,a)
        if game.isEnded(s):
            return game.getWinner(s)

# net=model.PredictorNetwork()
# faceTheMachine(net,'koth_net.h5')
network=policyIteration()
