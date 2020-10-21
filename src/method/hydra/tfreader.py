import tensorflow as tf
import tensorflow_text as text


def _parse_tfrecord_function(example, opcodes_lookup_table, bytes_lookup_table):
    example_fmt = {
        'opcodes': tf.io.FixedLenFeature([], tf.string),
        'bytes': tf.io.FixedLenFeature([], tf.string),
        'APIs': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
        }
    parsed = tf.io.parse_single_example(example, example_fmt)

    tokenizer = text.WhitespaceTokenizer()

    opcodes_tokens = tokenizer.tokenize(parsed['opcodes'])
    opcodes_IDs = opcodes_lookup_table.lookup(opcodes_tokens)

    bytes_tokens = tokenizer.tokenize(parsed['bytes'])
    bytes_IDs = bytes_lookup_table.lookup(bytes_tokens)

    feature_vector = tf.io.decode_raw(parsed['APIs'], tf.float32)
    return opcodes_IDs, bytes_IDs, feature_vector, parsed['label']


def make_dataset(filepath,
                 opcodes_lookup_table,
                 bytes_lookup_table,
                 SHUFFLE_BUFFER_SIZE=1024,
                 BATCH_SIZE=32,
                 EPOCHS=5):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.map(lambda x: _parse_tfrecord_function(x, opcodes_lookup_table, bytes_lookup_table))
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset
