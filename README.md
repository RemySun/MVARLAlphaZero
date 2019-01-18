# MVA Reinforcement learning project on Alpha Go Zero

This is a project centered around experimenting with DeepMind's AlphaGo Zero

Code for TDgammon training is in tdgammon.py using the model code found at https://github.com/fomorians/td-gammon

## Prerequisites

The following dependencies are recommended:
* Python 3
* Numpy
* Tensorflow
* Keras

Note that no code to create repository is built into the functions, the code supposes you created them yourself.

## Training a model

To train a model, creating a script with the following code should be enough

'''python
import train

train.policyIteration(save_dir='<Whatever you want>/')
'''

All functions in train.py and evaluate.py are documented, the first line of the function policyIteration in train.py is instantiates the keras model used. A zoo of possible models are available in model.py. Game logic for tic tac toe and align 4 is available in the corresponding directories. New game logic can be written by following the same API (we are really just providing function to do operations on game states) but switching is a bit clunky (you have to replace a lot of things yourself).
Evaluation functions are available in evaluate.py. For instance if you want to play yourself against the network, try something like this:

'''python
import evaluate
import model

net=model.<You pick...>()
evaluate.sanityCheck(net,opponent='human',save_dir='<Somewhere with checkpoints>/',koth_ckpt='<a checkpoint that exists>')
'''

Player abstractions are available in player.py
