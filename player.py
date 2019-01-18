import tictactoe.game as game
from mcts import MCTS
import tdgammon
import numpy as np
import tensorflow as tf

class RandomPlayer(object):
    def __init__(self):
          self.hist = []

    def reset(self):
        return


    def getMove(self,s):
        return np.random.choice(game.validMoves(s))

class HumanPlayer(object):
    def __init__(self,name='Bob'):
          self.name = name

    def reset(self):
        return

    def getMove(self,s):
        player = game.getCurrentPlayer(s)
        print('Hey',self.name,' it is your turn !\nBoard state is \n',s," you are playing ",player)
        s=int(input())
        return s

sess=tf.Session()
gammon=tdgammon.Model(sess,'TDsaves/','TDsaves/','TDsaves/')

class TDPlayer(object):
    def __init__(self,gammon=gammon):


        self.net=gammon
        self.net.restore()
        #tdgammon.test(self.net)

    def reset(self):
        return

    def getMove(self,s):
        best_a = -1
        best_v = -np.inf
        for a in game.validMoves(s):
            s_a = game.nextState(s,a)
            v= self.net.sess.run(self.net.V, feed_dict={self.net.x: [np.reshape(s_a,(9,))]})
            if -v > best_v:
                best_v = -v
                best_a = a
        return best_a

class AlphaZeroPlayer(object):
    def __init__(self,network,checkpoint,n_MCTS_search=25,tau=1):
        self.network = network
        self.checkpoint = checkpoint
        self.n_MCTS_search = n_MCTS_search
        self.tau = tau

    def reset(self):
        self.mcts = MCTS(tau=self.tau)

    def getMove(self,s):
        self.network.load_weights(self.checkpoint)
        for _ in range(self.n_MCTS_search):
            self.mcts.search(s,self.network)
        policy = self.mcts.computePi(s,self.network)
        a = np.random.choice(game.ACTIONS,p=policy)
        return a

class AlphaGreedyPlayer(object):
    def __init__(self,network,checkpoint,n_MCTS_search=25,tau=1):
        self.network = network
        self.checkpoint = checkpoint

    def reset(self):
        return

    def getMove(self,s):
        self.network.load_weights(self.checkpoint)
        best_a = -1
        best_v = -np.inf
        for a in game.validMoves(s):
            s_a = game.nextState(s,a)
            [p],[v] = self.network.predict(np.array([s_a]))
            if -v > best_v:
                best_v = -v
                best_a = a
        return best_a

class AlphaNetPlayer(object):
    def __init__(self,network,checkpoint):
        self.network = network
        self.checkpoint = checkpoint

    def reset(self):
        return

    def getMove(self,s):
        self.network.load_weights(self.checkpoint)
        [p],[v] = self.network.predict(np.array([s]))
        for i in range(len(p)):
            if i not in game.validMoves(s):
                p[i]=0
        p=p/np.sum(p)
        return np.random.choice(game.ACTIONS,p=p)
