
from __future__ import division

import time
import random
import numpy as np
import tensorflow as tf
import os
import align4.game as game

# helper to initialize a weight and bias variable
def weight_bias(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name='bias')
    return W, b

# helper to create a dense, fully-connected layer
def dense_layer(x, shape, activation, name):
    with tf.variable_scope(name):
        W, b = weight_bias(shape)
        return activation(tf.matmul(x, W) + b, name='activation')

class Model(object):
    def __init__(self, sess, model_path, summary_path, checkpoint_path, restore=False):
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path

        # setup our session
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # lambda decay
        lamda = tf.maximum(0.7, tf.train.exponential_decay(0.9, self.global_step, \
            30000, 0.96, staircase=True), name='lambda')

        # learning rate decay
        alpha = tf.maximum(0.01, tf.train.exponential_decay(0.1, self.global_step, \
            40000, 0.96, staircase=True), name='alpha')

        # describe network size
        layer_size_input = 42
        layer_size_hidden = 30
        layer_size_output = 1

        # placeholders for input and target output
        self.x = tf.placeholder('float', [1, layer_size_input], name='x')
        self.V_next = tf.placeholder('float', [1, layer_size_output], name='V_next')

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = dense_layer(self.x, [layer_size_input, layer_size_hidden], tf.tanh, name='layer1')
        self.V = dense_layer(prev_y, [layer_size_hidden, layer_size_output], tf.tanh, name='layer2')

        # delta = V_next - V
        delta_op = tf.reduce_sum(self.V_next - self.V, name='delta')

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')

        # check if the model predicts the correct state
        accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)), dtype='float'), name='accuracy')

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)

            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            delta_sum = tf.Variable(tf.constant(0.0), name='delta_sum', trainable=False)
            accuracy_sum = tf.Variable(tf.constant(0.0), name='accuracy_sum', trainable=False)

            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            delta_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            accuracy_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)

            loss_sum_op = loss_sum.assign_add(loss_op)
            delta_sum_op = delta_sum.assign_add(delta_op)
            accuracy_sum_op = accuracy_sum.assign_add(accuracy_op)

            loss_avg_op = loss_sum / tf.maximum(game_step, 1.0)
            delta_avg_op = delta_sum / tf.maximum(game_step, 1.0)
            accuracy_avg_op = accuracy_sum / tf.maximum(game_step, 1.0)

            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            delta_avg_ema_op = delta_avg_ema.apply([delta_avg_op])
            accuracy_avg_ema_op = accuracy_avg_ema.apply([accuracy_avg_op])

            # reset per-game monitoring variables
            game_step_reset_op = game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars)

        # for each variable, define operations to update the var with delta,
        # taking into account the gradient as part of the eligibility trace
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((lamda * trace) + grad)

                # grad with trace = alpha * delta * e
                grad_trace = alpha * delta_op * trace_op


                grad_apply = var.assign_add(grad_trace)
                apply_gradients.append(grad_apply)

        # as part of training we want to update our step and other monitoring variables
        with tf.control_dependencies([
                global_step_op,
                game_step_op,
                loss_sum_op,
                delta_sum_op,
                accuracy_sum_op,
                loss_avg_ema_op,
                delta_avg_ema_op,
                accuracy_avg_ema_op
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*apply_gradients, name='train')


        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=1)

        # run variable initializers
        self.sess.run(tf.initialize_all_variables())

        # after training a model, we can restore checkpoints here
        if restore:
            self.restore()

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def get_output(self, x):
        return self.sess.run(self.V, feed_dict={ self.x: x })

# def test(network, episodes=100, draw=False):
#     td_wins = 0
#     rand_win = 0
#     for i in range(n_games):
#         fight_result= fight(network,koth_ckpt,challenger_ckpt,first_player=(-1)**i,n_MCTS_search=n_MCTS_search)
#         if  fight_result< 0:
#             challenger_wins+=1
#             print('One for the challenger!')
#         elif fight_result > 0:
#             print("Somethingsomethingaboutoldginger")
#             koth_wins+=1
#         else:
#             print("And... Draw. Talk about anti climactic")
#     return challenger_wins/(koth_wins+challenger_wins)

def fight(network,koth_ckpt,challenger_ckpt,first_player=1,n_MCTS_search=25):
    s=game.startState()

#     while True:
#         player = game.getCurrentPlayer(s)
#         mcts=MCTS()
#         if (first_player*player) >0:
#             network.load_weights(koth_ckpt)
#         else:
#             network.load_weights(challenger_ckpt)
#         for _ in range(n_MCTS_search):
#             mcts.search(s,network)
#         policy=mcts.computePi(s,network)
#         a = np.random.choice(game.ACTIONS,p=policy)
#         s = game.nextState(s,a)
#         print(policy)
#         print(s)
#         if game.isEnded(s):
#             return first_player*game.getWinner(s)

#     for episode in range(episodes):
#         game = Game.new()

#         winner = game.play(players, draw=draw)
#         winners[winner] += 1

#         winners_total = sum(winners)
#         print("[Episode %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode, \
#                                                                               players[0].name, players[0].player, \
#                                                                               players[1].name, players[1].player, \
#                                                                               winners[0], winners[1], winners_total, \
#                                                                               (winners[0] / winners_total) * 100.0))

def test(network):

    episodes = 100
    n_wins=0
    n_loss=0
    for episode in range(episodes):
        fplayer = (-1)**random.randint(0, 1)
        player = fplayer
        s = game.startState()

        game_step = 0
        while not game.isEnded(s):
            if player == 1: 
                best_a = -1
                best_v = -np.inf
                for a in game.validMoves(s):
                    s_a = game.nextState(s,a)
                    v=network.sess.run(network.V, feed_dict={network.x: [np.reshape(s_a,(42,))]})
                    #print(v)
                    if -v > best_v:
                        best_v = -v
                        best_a = a
                    #time.sleep(1)
            else:
                best_a = np.random.choice(game.validMoves(s))
            s_next=game.nextState(s,best_a)
            s = s_next
            player*=-1
            game_step += 1

        winner = game.getWinner(s)
        if winner == fplayer:
            n_wins += 1
        else:
            n_loss += 1
        _, global_step, _ = network.sess.run([
            network.train_op,
            network.global_step,
            network.reset_op
        ], feed_dict={network.x: [np.reshape(s,(42,))], network.V_next: np.array([[winner]], dtype='float') })
        

        print("Random Game %d/%d (Winner: %s) in %d turns" % (episode, episodes, fplayer*winner, game_step))
    print(n_wins/(n_wins+n_loss))


def train(network):

    validation_interval = 1000
    episodes = 5000

    for episode in range(episodes):
        if episode != 0 and episode % validation_interval == 0:
            test(network)

        player_num = (-1)**random.randint(0, 1)

        s = game.startState()

        game_step = 0
        while not game.isEnded(s):
            best_a = -1
            best_v = -np.inf
            for a in game.validMoves(s):
                s_a = game.nextState(s,a)
                v=-network.sess.run(network.V, feed_dict={network.x: [np.reshape(s_a,(42,))]})
                #print(v)
                if v > best_v:
                    best_v = v
                    best_a = a
                #time.sleep(1)
            s_next=game.nextState(s,best_a)
            V_next=-network.sess.run(network.V, feed_dict={network.x: [np.reshape(s_next,(42,))]})
            network.sess.run(network.train_op, feed_dict={network.x: [np.reshape(s,(42,))], network.V_next: V_next })

            s = s_next
            game_step += 1

        winner = game.stateReward(s)

        _, global_step, _ = network.sess.run([
            network.train_op,
            network.global_step,
            network.reset_op
        ], feed_dict={network.x: [np.reshape(s,(42,))], network.V_next: np.array([[winner]], dtype='float') })

        print("Game %d/%d (Winner: %s) in %d turns" % (episode, episodes, game.getWinner(s), game_step))
        network.saver.save(network.sess, network.checkpoint_path + 'checkpoint', global_step=global_step)

sess=tf.Session()

net=Model(sess,'TDsaves/','TDsaves/','TDsaves/')
train(net)
