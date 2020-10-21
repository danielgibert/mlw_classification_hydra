import tensorflow as tf

class APIsNN(tf.keras.Model):
    def __init__(self, parameters):
        super(APIsNN, self).__init__()
        self.parameters = parameters

    def build(self, input_shapes):
        self.input_dropout = tf.keras.layers.Dropout(self.parameters["input_dropout_rate"],
                                                     input_shape=(None, self.parameters["features"]))

        self.h1 = tf.keras.layers.Dense(self.parameters['hidden'],
                                        activation="relu",
                                        input_shape=(None, self.parameters["features"]))


        self.output_dropout = tf.keras.layers.Dropout(self.parameters["hidden_dropout_rate"],
                                                      input_shape=(None, self.parameters["hidden"]))

        self.out = tf.keras.layers.Dense(self.parameters['output'],
                                         activation="softmax",
                                         input_shape=(None, self.parameters["hidden"]))

    def call(self, input_tensor, training=False):
        input_dropout = self.input_dropout(input_tensor, training=training)
        hidden1 = self.h1(input_dropout)
        output_dropout = self.output_dropout(hidden1, training=training)
        out = self.out(output_dropout)
        return out
