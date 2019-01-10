import tensorflow as tf
import keras
import model
import align4.game as game
from mcts import MCTS
import numpy as np

def policyIteration(n_iters=30,n_episodes=50):
    network = model.ConvNetwork()
    network.save_weights('koth_net.h5')
    history = []
    for iteration in range(n_iters):
        examples = []
        print('Training cycle',iteration)
        for episode in range(n_episodes):
            print('playing episode',episode)
            examples += playMatch(network)
        print("Training network")

        history.append(trainNetwork(network,examples))
        stupidityCheck(network)
        print('Finding out who the better network is')
        network.save_weights('challenger_net.h5')
        frac_win = duel(network,'koth_net.h5','challenger_net.h5')
        if frac_win > 0.5:
            print('The young will replace the old. Or something to that effect')
            network.load_weights('challenger_net.h5')
            network.save_weights('koth_net.h5')
        else:
            print('Come back in a million iterations')
            network.load_weights('koth_net.h5')
        network.save_weights('best_cnn_'+str(iteration)+'.h5')
    return network

def trainNetwork(network, examples):
    input_data = np.array([ex[0] for ex in examples])
    ground_truths = [np.array([ex[1] for ex in examples]),np.array([ex[2] for ex in examples])]
 
    history = network.fit(input_data,ground_truths,batch_size=10,epochs=60,shuffle=True)
    return history

def playMatch(network,n_MCTS_search=25):
    examples = []

    s=game.startState()

    while True:
        mcts = MCTS()
        for _ in range(n_MCTS_search):
            mcts.search(s,network)
        policy = mcts.computePi(s,network)
        examples.append([s,policy,None])
        a = np.random.choice(game.ACTIONS,p=policy)

        s = game.nextState(s,a)
        if game.isEnded(s):
            for example in examples:
                example[-1] = game.getCurrentPlayer(example[0])*game.getWinner(s)
            return examples

def duel(network,koth_ckpt,challenger_ckpt,n_MCTS_search=25,n_games=30):
    challenger_wins = 0
    koth_wins=0
    for i in range(n_games):
        fight_result= fight(network,koth_ckpt,challenger_ckpt,first_player=(-1)**i,n_MCTS_search=n_MCTS_search)
        if  fight_result< 0:
            challenger_wins+=1
            print('One for the challenger!')
        elif fight_result > 0:
            print("Somethingsomethingaboutoldginger")
            koth_wins+=1
        else:
            print("And... Draw. Talk about anti climactic")
    return challenger_wins/(koth_wins+challenger_wins)

def fight(network,koth_ckpt,challenger_ckpt,first_player=1,n_MCTS_search=25):
    s=game.startState()

    while True:
        player = game.getCurrentPlayer(s)
        mcts=MCTS()
        if (first_player*player) >0:
            network.load_weights(koth_ckpt)
        else:
            network.load_weights(challenger_ckpt)
        for _ in range(n_MCTS_search):
            mcts.search(s,network)
        policy=mcts.computePi(s,network)
        a = np.random.choice(game.ACTIONS,p=policy)
        s = game.nextState(s,a)
        print(policy)
        print(s)
        if game.isEnded(s):
            return first_player*game.getWinner(s)

def stupidityCheck(net):

    mcts = MCTS(c_puct=0.1)

    s=game.startState()
    s=game.nextState(s,3)
    s=game.nextState(s,3)
    s=game.nextState(s,4)

    for _ in range(50):
        mcts.search(s,net)
    print([mcts.N[str(s),a] for a in range(7)])
    return



def faceTheMachine(network,koth_ckpt,n_MCTS_search=25):
    s=game.startState()

    while True:
        mcts_koth = MCTS()
        mcts_challenger = MCTS()
        player = game.getCurrentPlayer(s)
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

# net=model.ConvNetwork()
# faceTheMachine(net,'koth_net.h5')
network=policyIteration()
