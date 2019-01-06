import tensorflow as tf
import keras

def PredictorNetwork(input_shape=(43,),learning_rate=0.01):
    input_tensor = keras.Input(shape=input_shape)
    layer0 = keras.layers.Dense(64,activation=tf.nn.relu)(input_tensor)
    layer1 = keras.layers.Dense(32,activation=tf.nn.relu)(layer0)
    layer2 = keras.layers.Dense(16,activation=tf.nn.relu)(layer1)
    p_moves = keras.layers.Dense(8,activation=tf.nn.softmax)(layer2)
    v_state = keras.layers.Dense(1)(layer2)
    model = keras.Model(input_tensor,[p_moves,v_state])
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=keras.optimizers.SGD(learning_rate))
    return model
