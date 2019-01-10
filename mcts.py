import numpy as np
import align4.game as game

class MCTS():
    """
    Class implementing
    """
    def __init__(self,c_puct=0.1,n_sim=100,tau=1):


        self.c_puct = c_puct
        self.n_sim = n_sim
        self.tau = tau

        self.Q_s_a = {}
        self.N_s_a = {}
        self.W_s_a = {}

        self.P_s = {} # For simplicity this is the only value stored as lists for each state

        self.end_s = {}

    def search(self,s,network,T_max = 5000):

        # If we observe a terminal game state
        if game.isEnded(s):
            #return 1
            return -game.stateReward(s)
        
        if str(s) not in self.P_s:

            ([self.P_s[str(s)]],[v]) = network.predict(np.array([s]))

            for a in game.validMoves(s):
                self.Q_s_a[(str(s),a)] = 0
                self.N_s_a[(str(s),a)] = 0
                self.W_s_a[(str(s),a)] = 0
            return -v
        
        sum_N_s_b = np.sum([self.N_s_a[(str(s),b)] for b in game.validMoves(s)])

        best_U = -np.inf
        best_a = -1
        for a in game.validMoves(s):
            U = self.Q_s_a[(str(s),a)] + self.c_puct * self.P_s[str(s)][a] * np.sqrt(sum_N_s_b)/(1+self.N_s_a[(str(s),a)])
            if U > best_U:
                best_U = U
                best_a = a
        

        new_s = game.nextState(s,best_a)
        
        v = self.search(new_s, network)
        
        self.W_s_a[(str(s),best_a)]+=v
        self.N_s_a[(str(s),best_a)]+=1
        self.Q_s_a[(str(s),best_a)]+=self.W_s_a[(str(s),best_a)]/self.N_s_a[(str(s),best_a)]
        
        return -v

    def computePi(self,s,network,n_sim=100):
        # for _ in range(n_sim):
        #     self.search(s,network)

        hits = [self.N_s_a[(str(s),a)]**self.tau if (str(s),a) in self.N_s_a else 0 for a in (game.ACTIONS)]

        pi = [p/np.sum(hits) for p in hits]

        return pi

# import model

# mcts = MCTS(c_puct=0.1)
# net=model.ConvNetwork()
# net.load_weights('koth_net.h5')
# s=game.startState()
# s=game.nextState(s,3)
# s=game.nextState(s,3)
# s=game.nextState(s,4)


# v=[mcts.search(s,net) for _ in range(200)]
# n=[mcts.N_s_a[str(s),a] for a in range(7)]
