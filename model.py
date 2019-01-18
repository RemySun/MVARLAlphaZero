import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Reshape
from keras.regularizers import l2
import keras

def ToeNetwork(input_shape=(3,3),learning_rate=0.001):
    input_tensor = keras.Input(shape=input_shape)
    reshaped_in = keras.layers.Reshape((9,))(input_tensor)
    layer1 = keras.layers.Dense(64,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(reshaped_in)
    layer1_bn = keras.layers.BatchNormalization()(layer1)
    layer1_dr = keras.layers.Dropout(0.5)(layer1_bn)
    p_moves = keras.layers.Dense(9,activation=tf.nn.softmax, kernel_regularizer=l2(0.01))(layer1_dr)
    v_state = keras.layers.Dense(1,activation=tf.nn.tanh, kernel_regularizer=l2(0.01))(layer1_dr)
    model = keras.Model(input_tensor,[p_moves,v_state])
    model.compile(loss=['binary_crossentropy','mean_squared_error'], optimizer=keras.optimizers.SGD(learning_rate,momentum=0.9))
    return model

def ToeMonsterNetwork(input_shape=(3,3),learning_rate=0.001):
    input_tensor = keras.Input(shape=input_shape)
    reshaped_in = keras.layers.Reshape((9,))(input_tensor)
    layer1 = keras.layers.Dense(512,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(reshaped_in)
    layer1_bn = keras.layers.BatchNormalization()(layer1)
    layer1_dr = keras.layers.Dropout(0.5)(layer1_bn)
    layer2 = keras.layers.Dense(512,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(layer1_dr)
    layer2_bn = keras.layers.BatchNormalization()(layer2)
    layer2_dr = keras.layers.Dropout(0.5)(layer2_bn)
    layer3 = keras.layers.Dense(256,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(layer2_dr)
    layer3_bn = keras.layers.BatchNormalization()(layer3)
    layer3_dr = keras.layers.Dropout(0.5)(layer3_bn)
    layer4 = keras.layers.Dense(128,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(layer3_dr)
    layer4_bn = keras.layers.BatchNormalization()(layer4)
    layer4_dr = keras.layers.Dropout(0.5)(layer4_bn)
    layer5 = keras.layers.Dense(64,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(layer4_dr)
    layer5_bn = keras.layers.BatchNormalization()(layer5)
    layer5_dr = keras.layers.Dropout(0.5)(layer5_bn)
    layer6 = keras.layers.Dense(32,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(layer5_dr)
    layer6_bn = keras.layers.BatchNormalization()(layer6)
    layer6_dr = keras.layers.Dropout(0.5)(layer6_bn)
    p_moves = keras.layers.Dense(9,activation=tf.nn.softmax, kernel_regularizer=l2(0.01))(layer6_dr)
    v_state = keras.layers.Dense(1,activation=tf.nn.tanh, kernel_regularizer=l2(0.01))(layer6_dr)
    model = keras.Model(input_tensor,[p_moves,v_state])
    model.compile(loss=['binary_crossentropy','mean_squared_error'], optimizer=keras.optimizers.SGD(learning_rate,momentum=0.9))
    return model


def ToeDeepNetwork(input_shape=(3,3),learning_rate=0.001):
    input_tensor = keras.Input(shape=input_shape)
    reshaped_in = keras.layers.Reshape((9,))(input_tensor)
    layer1 = keras.layers.Dense(64,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(reshaped_in)
    layer1_bn = keras.layers.BatchNormalization()(layer1)
    layer1_dr = keras.layers.Dropout(0.5)(layer1_bn)
    layer2 = keras.layers.Dense(64,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(layer1_dr)
    layer2_bn = keras.layers.BatchNormalization()(layer2)
    layer2_dr = keras.layers.Dropout(0.5)(layer2_bn)
    layer3 = keras.layers.Dense(32,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(layer2_dr)
    layer3_bn = keras.layers.BatchNormalization()(layer3)
    layer3_dr = keras.layers.Dropout(0.5)(layer3_bn)
    layer4 = keras.layers.Dense(16,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(layer3_dr)
    layer4_bn = keras.layers.BatchNormalization()(layer4)
    layer4_dr = keras.layers.Dropout(0.5)(layer4_bn)
    p_moves = keras.layers.Dense(9,activation=tf.nn.softmax, kernel_regularizer=l2(0.01))(layer4_dr)
    v_state = keras.layers.Dense(1,activation=tf.nn.tanh, kernel_regularizer=l2(0.01))(layer4_dr)
    model = keras.Model(input_tensor,[p_moves,v_state])
    model.compile(loss=['binary_crossentropy','mean_squared_error'], optimizer=keras.optimizers.SGD(learning_rate,momentum=0.9))
    return model


def ToeConvNetwork(input_shape=(3,3),learning_rate=0.01):
    input_tensor = keras.Input(shape=input_shape)
    reshaped_in = keras.layers.Reshape((3,3,1))(input_tensor)
    layer0 = keras.layers.Conv2D(32,(2,2),activation=tf.nn.relu, kernel_regularizer=l2(0.01))(reshaped_in)
    flat = keras.layers.Flatten()(layer0)
    layer = keras.layers.Dense(64,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(flat)
    layer_dr = keras.layers.Dropout(0.5)(layer)
    layer1 = keras.layers.Dense(32,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(layer_dr)
    layer1_dr = keras.layers.Dropout(0.5)(layer1)
    layer2 = keras.layers.Dense(16,activation=tf.nn.relu, kernel_regularizer=l2(0.01))(layer1_dr)
    layer2_dr = keras.layers.Dropout(0.5)(layer2)
    p_moves = keras.layers.Dense(9,activation=tf.nn.softmax, kernel_regularizer=l2(0.01))(layer2_dr)
    v_state = keras.layers.Dense(1,activation=tf.nn.tanh, kernel_regularizer=l2(0.01))(layer2_dr)
    model = keras.Model(input_tensor,[p_moves,v_state])
    model.compile(loss=['binary_crossentropy','mean_squared_error'], optimizer=keras.optimizers.SGD(learning_rate,momentum=0.9))
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
    model.compile(loss=['binary_crossentropy','mean_squared_error'], optimizer=keras.optimizers.SGD(learning_rate))
    return model
