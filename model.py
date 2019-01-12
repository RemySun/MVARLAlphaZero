import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Reshape
import keras

def PredictorNetwork(input_shape=(43,),learning_rate=0.01):
    input_tensor = keras.Input(shape=input_shape)
    layer0 = keras.layers.Dense(64,activation=tf.nn.relu)(input_tensor)
    layer1 = keras.layers.Dense(32,activation=tf.nn.relu)(layer0)
    layer2 = keras.layers.Dense(16,activation=tf.nn.relu)(layer1)
    p_moves = keras.layers.Dense(7,activation=tf.nn.softmax)(layer2)
    v_state = keras.layers.Dense(1,activation=tf.nn.tanh)(layer2)
    model = keras.Model(input_tensor,[p_moves,v_state])
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=keras.optimizers.SGD(learning_rate))
    return model

def ConvNetwork(input_shape=(6,7),learning_rate=0.01):
    input_tensor = keras.Input(shape=input_shape)
    reshaped_in = keras.layers.Reshape((6,7,1))(input_tensor)
    layer0 = keras.layers.Conv2D(32,(4,4),activation=tf.nn.relu)(reshaped_in)
    flat = keras.layers.Flatten()(layer0)
    layer1 = keras.layers.Dense(64,activation=tf.nn.relu)(flat)
    layer1_dr = keras.layers.Dropout(0.5)(layer1)
    layer2 = keras.layers.Dense(32,activation=tf.nn.relu)(layer1_dr)
    layer2_dr = keras.layers.Dropout(0.5)(layer2)
    layer3 = keras.layers.Dense(16,activation=tf.nn.relu)(layer2_dr)
    layer3_dr = keras.layers.Dropout(0.5)(layer3)
    p_moves = keras.layers.Dense(7,activation=tf.nn.softmax)(layer3_dr)
    v_state = keras.layers.Dense(1,activation=tf.nn.tanh)(layer3_dr)
    model = keras.Model(input_tensor,[p_moves,v_state])
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=keras.optimizers.SGD(learning_rate))
    return model
