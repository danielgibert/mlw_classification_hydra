import tensorflow as tf


class ShallowCNN(tf.keras.Model):
    def __init__(self, parameters):
        super(ShallowCNN, self).__init__()
        self.parameters = parameters

    def build(self, input_shapes):
        self.emb = tf.keras.layers.Embedding(self.parameters['V'], self.parameters['E'], input_shape=(None, self.parameters['seq_length']))

        self.conv_3 = tf.keras.layers.Conv2D(self.parameters['conv']['num_filters'],
                                             (self.parameters['conv']['size'][0], self.parameters['E']),
                                             activation="relu",
                                             input_shape=(None,
                                                          self.parameters['seq_length'],
                                                          self.parameters['E']))
        self.global_max_pooling_3 = tf.keras.layers.GlobalMaxPooling2D()

        self.conv_5 = tf.keras.layers.Conv2D(self.parameters['conv']['num_filters'],
                                             (self.parameters['conv']['size'][1], self.parameters['E']),
                                             activation="relu",
                                             input_shape=(None,
                                                          self.parameters['seq_length'],
                                                          self.parameters['E']))
        self.global_max_pooling_5 = tf.keras.layers.GlobalMaxPooling2D()


        self.conv_7 = tf.keras.layers.Conv2D(self.parameters['conv']['num_filters'],
                                             (self.parameters['conv']['size'][2], self.parameters['E']),
                                             activation="relu",
                                             input_shape=(None,
                                                          self.parameters['seq_length'],
                                                          self.parameters['E']))
        self.global_max_pooling_7 = tf.keras.layers.GlobalMaxPooling2D()

        self.dense_dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(self.parameters['output'],
                                           activation="softmax")

    def call(self, input_tensor, training=False):
        emb = self.emb(input_tensor)
        emb_expanded = tf.keras.backend.expand_dims(emb, axis=-1)


        conv_3 = self.conv_3(emb_expanded)
        pool_3 = self.global_max_pooling_3(conv_3)

        conv_5 = self.conv_5(emb_expanded)
        pool_5 = self.global_max_pooling_5(conv_5)

        conv_7 = self.conv_7(emb_expanded)
        pool_7 = self.global_max_pooling_7(conv_7)

        features = tf.keras.layers.concatenate([pool_3, pool_5, pool_7])
        features_dropout = self.dense_dropout(features, training=training)
        output = self.dense(features_dropout)

        return output