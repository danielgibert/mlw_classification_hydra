import json
import tensorflow as tf
import numpy as np
import math

def initialize_TFRecords(tfrecords_filepath, num_tfrecords=10, filename="training"):
    training_writers = []
    for i in range(num_tfrecords):
        training_writers.append(tf.io.TFRecordWriter(tfrecords_filepath + "{}{}.tfrecords".format(filename,i)))
    return training_writers

def create_lookup_table(vocabulary_mapping, num_oov_buckets):
    keys = [k for k in vocabulary_mapping.keys()]
    values = [tf.constant(vocabulary_mapping[k], dtype=tf.int64) for k in keys]

    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=keys,
            values=values
        ),
        num_oov_buckets
    )
    return table

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_vocabulary(vocabulary_filepath):
    """
    It reads and stores in a dictionary-like structure the data from the file passed as argument

    Parameters
    ----------
    vocabulary_filepath: str
        JSON-like file

    Return
    ------
    vocabulary_dict: dict
    """
    with open(vocabulary_filepath, "r") as vocab_file:
        vocabulary_dict = json.load(vocab_file)
    return vocabulary_dict

def serialize_mnemonics_example_IDs(mnemonic_IDs, label):
    """
    Creates a tf.Example message ready to be written to a file
    :param mnemonics: str -> "[4,67,109,...,402, 402]"
    :param label: int [0,8]
    :return:
    """
    feature = {
        'opcodes': _bytes_feature(np.array(mnemonic_IDs).tostring()),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_mnemonics_example(mnemonics, label):
    """
    Creates a tf.Example message ready to be written to a file
    :param mnemonics: str -> "push,pop,...,NONE"
    :param label: int [0,8]
    :return:
    """
    feature = {
        'opcodes': _bytes_feature(mnemonics.encode('UTF-8')),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_bytes_example(bytes, label):
    """
    Creates a tf.Example message ready to be written to a file
    :param bytes: str -> "00,FF,...,??,NONE"
    :param label: int [0,8]
    :return:
    """
    feature = {
        'bytes': _bytes_feature(bytes.encode('UTF-8')),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_apis_example(feature_vector, label):
    feature = {
        'APIs': _bytes_feature(feature_vector.tostring()),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_hydra_example(opcodes, bytes, apis_values, label):
    feature = {
        'opcodes': _bytes_feature(opcodes.encode('UTF-8')),
        'bytes': _bytes_feature(bytes.encode('UTF-8')),
        'APIs': _bytes_feature(apis_values.tostring()),
        'label': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def load_parameters(parameters_path):
    """
    It loads the network parameters

    Parameters
    ----------
    parameters_path: str
        File containing the parameters of the network
    """
    with open(parameters_path, "r") as param_file:
        params = json.load(param_file)
    return params
