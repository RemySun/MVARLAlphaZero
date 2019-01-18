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
import evaluate


def policyIteration(n_iters=50,n_episodes=100,load=None,n_MCTS_search=15,max_ex=10000,save_dir="toe_deep_narrow/"):
    '''
    Runs the policy iteration training of a network specified in the first line of the function code

    Input:
    ------

    n_iters: number of training iterations

    n_episodes: number of episodes played per iterations

    load: If defined, the checkpoint path to load

    n_MCTS_search: Number of mcts searches before playing a move

    max_ex: maximum number of training examples

    save_dir: Where to save models (directory path)
    '''
    network = model.ToeDeepNetwork()

    if load:
        network.save_weights(load)

    network.save_weights(save_dir + 'koth_net.h5')

    history = []
    examples = []

    for iteration in range(n_iters):

        network.save_weights(save_dir + 'best_cnn_'+str(iteration)+'.h5')

        print('Face... The raaaaannnnndom')
        evaluate.sanityCheck(network,n_games = 100,n_MCTS_search=n_MCTS_search,save_dir=save_dir)

        print('Meet the gammon')
        evaluate.sanityCheck(network,n_games = 100,n_MCTS_search=n_MCTS_search,opponent='gammon',save_dir=save_dir)

        print('Yes, you sucked that much')
        evaluate.sanityCheck(network,n_games = 20,n_MCTS_search=n_MCTS_search,opponent='notrain',save_dir=save_dir)

        network.load_weights(save_dir + 'koth_net.h5')

        iterExamples = []
        print('Training cycle',iteration)
        for episode in range(n_episodes):
            print('playing episode',episode,end='\r')
            iterExamples += playMatch(network,n_MCTS_search=n_MCTS_search,tau=2)

        examples += iterExamples
        if len(examples) > max_ex:
            examples = examples[-max_ex:]

        print("Training network")

        history.append(trainNetwork(network,examples,save_dir=save_dir))

        evaluate.stupidityCheck(network,n_MCTS_search)

        print('Finding out who the better network is')

        network.save_weights(save_dir + 'challenger_net.h5')

        koth=AlphaZeroPlayer(network,save_dir+'koth_net.h5',n_MCTS_search=n_MCTS_search,tau=0.5)
        chall=AlphaZeroPlayer(network,save_dir+'challenger_net.h5',n_MCTS_search=n_MCTS_search,tau=0.5)
        frac_win = evaluate.duel(koth,chall)

        if frac_win > 0.6:

            print('The young will replace the old. Or something to that effect')

            network.load_weights(save_dir + 'challenger_net.h5')
            network.save_weights(save_dir + 'koth_net.h5')

            f=open(save_dir+'model_change','a')
            f.write('Changed model at the end of training iteration'+str(iteration)+'\n')

            f.close()
        else:

            print('Come back in a million iterations')
            network.load_weights(save_dir + 'koth_net.h5')

        if iteration % 5 == 0:
            K.set_value(network.optimizer.lr, 0.001*(0.9**(iteration//5)))

    return network, history



def trainNetwork(network, examples):
    '''
    Train a network from examples

    Input:
    ------

    network: Keras model to train

    examples: list of training examples
    '''
    input_data = np.array([ex[0] for ex in examples])
    ground_truths = [np.array([ex[1] for ex in examples]),np.array([ex[2] for ex in examples])]

    early_stopping_monitor = EarlyStopping(patience=3)

    history = network.fit(input_data,ground_truths,batch_size=32,epochs=300,shuffle=True,callbacks=[early_stopping_monitor],validation_split=0.1)

    return history

def playMatch(network,n_MCTS_search=25,tau=2):
    '''
    Runs a self play match.

    Input:
    ------

    networks: Keras model used

    n_MCTS_search: Number of mcts searches before playing a move

    tau: Temperature of the MCTS

    Returns:
    --------

    examples: list examples seen in the game.
    '''
    examples = []
    s=game.startState()
    mcts = MCTS(tau=tau)
    while True:
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

# net=model.ToeDeepNetwork()
# alpha_player = AlphaZeroPlayer(net,save_dir+'koth_net.h5',n_MCTS_search=15,tau=0.5)
# alpha_player.reset()
# human = HumanPlayer()
# fight(alpha_player,human)
# faceTheMachine(net,'koth_net.h5')
# network=policyIteration()

