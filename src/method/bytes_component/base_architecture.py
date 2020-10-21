import tensorflow as tf


class DeepConv(tf.keras.Model):
    def __init__(self, parameters):
        super(DeepConv, self).__init__()
        self.parameters = parameters

    def build(self, input_shapes):
        self.emb = tf.keras.layers.Embedding(self.parameters['V'], self.parameters['E'],
                                             input_shape=(None, self.parameters['max_bytes_values']))

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.parameters['num_filters'][0],
                                                  kernel_size=[self.parameters['kernel_sizes'][0],
                                                               self.parameters['E']],
                                                  strides=(self.parameters['strides'][0],1),
                                                  data_format='channels_last',
                                                  use_bias=True,
                                                  activation="relu")

        self.conv_2 = tf.keras.layers.Conv2D(filters=self.parameters['num_filters'][1],
                                                  kernel_size=[self.parameters['kernel_sizes'][1],
                                                               1],
                                                  strides=(self.parameters['strides'][1],1),
                                                  data_format='channels_last',
                                                  use_bias=True,
                                                  activation="relu")

        self.max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(self.parameters['max_pool_size'], 1))

        self.conv_3 = tf.keras.layers.Conv2D(filters=self.parameters['num_filters'][2],
                                                 kernel_size=[self.parameters['kernel_sizes'][2],
                                                              1],
                                                 strides=(self.parameters['strides'][2], 1),
                                                 data_format='channels_last',
                                                 use_bias=True,
                                                 activation="relu")

        self.conv_4 = tf.keras.layers.Conv2D(filters=self.parameters['num_filters'][3],
                                                 kernel_size=[self.parameters['kernel_sizes'][3],
                                                              1],
                                                 strides=(self.parameters['strides'][3], 1),
                                                 data_format='channels_last',
                                                 use_bias=True,
                                                 activation="relu")

        self.global_avg_pool = tf.keras.layers.GlobalAvgPool2D()

        self.drop_1 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.dense_1 =  tf.keras.layers.Dense(self.parameters['hidden'][0],
                                                           activation="selu")

        self.drop_2 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.dense_2 = tf.keras.layers.Dense(self.parameters['hidden'][1],
                                                       activation="selu")

        self.drop_3 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])

        self.dense_3 = tf.keras.layers.Dense(self.parameters['hidden'][2],
                                                       activation="selu")

        self.drop_4 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.out = tf.keras.layers.Dense(self.parameters['output'],
                                           activation="softmax")


    def call(self, input_tensor, training=False):
        emb = self.emb(input_tensor)
        emb_expanded = tf.keras.backend.expand_dims(emb, axis=-1)

        conv_1 = self.conv_1(emb_expanded)
        conv_2 = self.conv_2(conv_1)

        max_pool_1 = self.max_pool_1(conv_2)

        conv_3 = self.conv_3(max_pool_1)
        conv_4 = self.conv_4(conv_3)

        features = self.global_avg_pool(conv_4)

        drop_1 = self.drop_1(features, training=training)
        dense_1 = self.dense_1(drop_1)

        drop_2 = self.drop_2(dense_1, training=training)
        dense_2 = self.dense_2(drop_2)

        drop_3 = self.drop_3(dense_2, training=training)
        dense_3 = self.dense_3(drop_3)

        drop_4 = self.drop_4(dense_3, training=training)
        output = self.out(drop_4)

        return output