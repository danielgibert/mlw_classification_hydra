import tensorflow as tf

class HYDRA(tf.keras.Model):
    def __init__(self, parameters):
        super(HYDRA, self).__init__()
        self.parameters = parameters

    def build(self, input_shapes):
        # Bytes component
        ######################################### Bytes component ######################################################
        self.bytes_emb = tf.keras.layers.Embedding(self.parameters['bytes']['V'], self.parameters['bytes']['E'],
                                             input_shape=(None, self.parameters['max_bytes']))

        self.bytes_conv_1 = tf.keras.layers.Conv2D(filters=self.parameters['bytes']['num_filters'][0],
                                             kernel_size=[self.parameters['bytes']['kernel_sizes'][0],
                                                          self.parameters['bytes']['E']],
                                             strides=(self.parameters['strides'][0], 1),
                                             data_format='channels_last',
                                             use_bias=True,
                                             activation="relu")

        self.bytes_conv_2 = tf.keras.layers.Conv2D(filters=self.parameters['bytes']['num_filters'][1],
                                             kernel_size=[self.parameters['bytes']['kernel_sizes'][1],
                                                          1],
                                             strides=(self.parameters['bytes']['strides'][1], 1),
                                             data_format='channels_last',
                                             use_bias=True,
                                             activation="relu")

        self.bytes_max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(self.parameters['bytes']['max_pool_size'], 1))

        self.bytes_conv_3 = tf.keras.layers.Conv2D(filters=self.parameters['bytes']['num_filters'][2],
                                             kernel_size=[self.parameters['bytes']['kernel_sizes'][2],
                                                          1],
                                             strides=(self.parameters['bytes']['strides'][2], 1),
                                             data_format='channels_last',
                                             use_bias=True,
                                             activation="relu")

        self.bytes_conv_4 = tf.keras.layers.Conv2D(filters=self.parameters['bytes']['num_filters'][3],
                                             kernel_size=[self.parameters['bytes']['kernel_sizes'][3],
                                                          1],
                                             strides=(self.parameters['bytes']['strides'][3], 1),
                                             data_format='channels_last',
                                             use_bias=True,
                                             activation="relu")

        self.bytes_global_avg_pool = tf.keras.layers.GlobalAvgPool2D()

        self.bytes_drop_1 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.bytes_dense_1 = tf.keras.layers.Dense(self.parameters['bytes']['hidden'][0],
                                             activation="selu")

        self.bytes_drop_2 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.bytes_dense_2 = tf.keras.layers.Dense(self.parameters['bytes']['hidden'][1],
                                             activation="selu")

        self.bytes_drop_3 = tf.keras.layers.Dropout(self.parameters["dropout_rate"])

        self.bytes_dense_3 = tf.keras.layers.Dense(self.parameters['bytes']['hidden'][2],
                                             activation="selu")

        ####################################### Opcodes component ######################################################
        self.opcodes_emb = tf.keras.layers.Embedding(self.parameters['opcodes']['V'],
                                                   self.parameters['opcodes']['E'],
                                                   input_shape=(None,
                                                                self.parameters['max_opcodes']))

        self.opcodes_conv_3 = tf.keras.layers.Conv2D(self.parameters['opcodes']['conv']['num_filters'],
                                                   (self.parameters['opcodes']['conv']['size'][0],
                                                    self.parameters['opcodes']['E']),
                                                   activation="relu",
                                                   input_shape=(None,
                                                                self.parameters['opcodes']['seq_length'],
                                                                self.parameters['opcodes']['E']))
        self.opcodes_global_max_pooling_3 = tf.keras.layers.GlobalMaxPooling2D()

        self.opcodes_conv_5 = tf.keras.layers.Conv2D(self.parameters['opcodes']['conv']['num_filters'],
                                                   (self.parameters['opcodes']['conv']['size'][1], self.parameters['E']),
                                                   activation="relu",
                                                   input_shape=(None,
                                                                self.parameters['opcodes']['seq_length'],
                                                                self.parameters['opcodes']['E']))
        self.opcodes_global_max_pooling_5 = tf.keras.layers.GlobalMaxPooling2D()

        self.opcodes_conv_7 = tf.keras.layers.Conv2D(self.parameters['opcodes']['conv']['num_filters'],
                                                   (self.parameters['opcodes']['conv']['size'][2], self.parameters['E']),
                                                   activation="relu",
                                                   input_shape=(None,
                                                                self.parameters['opcodes']['seq_length'],
                                                                self.parameters['opcodes']['E']))
        self.opcodes_global_max_pooling_7 = tf.keras.layers.GlobalMaxPooling2D()

        ################################################# APIs Component ###############################################
        self.apis_input_dropout = tf.keras.layers.Dropout(self.parameters["input_dropout_rate"],
                                                     input_shape=(None, self.parameters["api_features"]))

        self.apis_hidden_1 = tf.keras.layers.Dense(self.parameters['bytes']['hidden'],
                                        activation="relu",
                                        input_shape=(None, self.parameters["api_features"]))

        self.bytes_apis_dense_dropout = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.bytes_apis_dense = tf.keras.layers.Dense(self.parameters['hidden'][0], activation="selu")

        self.dense_dropout = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.dense = tf.keras.layers.Dense(self.parameters['hidden'][1], activation="selu")

        self.output_dropout = tf.keras.layers.Dropout(self.parameters["dropout_rate"])
        self.out = tf.keras.layers.Dense(self.parameters['output'],
                                           activation="softmax")

    def call(self, opcodes_tensor, bytes_tensor, apis_tensor, training=False):
        # Bytes subcomponent
        bytes_emb = self.bytes_emb(bytes_tensor)
        bytes_emb_expanded = tf.keras.backend.expand_dims(bytes_emb, axis=-1)

        bytes_conv_1 = self.bytes_conv_1(bytes_emb_expanded)
        bytes_conv_2 = self.bytes_conv_2(bytes_conv_1)

        bytes_max_pool_1 = self.bytes_max_pool_1(bytes_conv_2)

        bytes_conv_3 = self.bytes_conv_3(bytes_max_pool_1)
        bytes_conv_4 = self.bytes_conv_4(bytes_conv_3)

        bytes_features = self.bytes_global_avg_pool(bytes_conv_4)

        bytes_drop_1 = self.bytes_drop_1(bytes_features, training=training)
        bytes_dense_1 = self.bytes_dense_1(bytes_drop_1)

        bytes_drop_2 = self.bytes_drop_2(bytes_dense_1, training=training)
        bytes_dense_2 = self.bytes_dense_2(bytes_drop_2)

        bytes_drop_3 = self.bytes_drop_3(bytes_dense_2, training=training)
        bytes_dense_3 = self.bytes_dense_3(bytes_drop_3)

        # Opcodes subcomponent
        opcodes_emb = self.opcodes_emb(opcodes_tensor)
        opcodes_emb_expanded = tf.keras.backend.expand_dims(opcodes_emb, axis=-1)

        opcodes_conv_3 = self.opcodes_conv_3(opcodes_emb_expanded)
        opcodes_pool_3 = self.opcodes_global_max_pooling_3(opcodes_conv_3)

        opcodes_conv_5 = self.opcodes_conv_5(opcodes_emb_expanded)
        opcodes_pool_5 = self.opcodes_global_max_pooling_5(opcodes_conv_5)

        opcodes_conv_7 = self.opcodes_conv_7(opcodes_emb_expanded)
        opcodes_pool_7 = self.opcodes_global_max_pooling_7(opcodes_conv_7)

        #APIs subcomponent
        apis_input_dropout = self.apis_input_dropout(apis_tensor, training=training)
        apis_hidden1 = self.apis_hidden_1(apis_input_dropout)


        # Features fusion
        features_api_bytes = tf.keras.layers.concatenate([bytes_dense_3, apis_hidden1])
        features_api_bytes_dropout = self.bytes_apis_dense_dropout(features_api_bytes, training=training)
        dense_api_bytes = self.bytes_apis_dense(features_api_bytes_dropout)

        features = tf.keras.layers.concatenate([opcodes_pool_3, opcodes_pool_5, opcodes_pool_7, dense_api_bytes])
        features_dropout = self.dense_dropout(features, training=training)
        dense_opcodes_apis_bytes = self.dense(features_dropout)

        features_dropout = self.dense_dropout(dense_opcodes_apis_bytes, training=training)
        output = self.out(features_dropout)

        return output

    def load_opcodes_subnetwork_pretrained_weights(self, model):
        """
        Loads the pretrained weights of the opcodes subnetwork into the bimodal architecture
        :param model: filepath to the opcodes' model
        :return:
        """
        print("ToImplement")

    def load_bytes_subnetwork_pretrained_weights(self, model):
        """
        Loads the pretrained weights of the bytes subnetwork into the bimodal architecture
        :param model: filepath to the bytes' model
        :return:
        """
        print("ToImplement")

    def load_apis_subnetwork_pretrained_weights(self, model):
        """
        Loads the pretrained weights of the apis subnetwork into the bimodal architecture
        :param model: filepath to the apis' model
        :return:
        """
        print("ToImplement")