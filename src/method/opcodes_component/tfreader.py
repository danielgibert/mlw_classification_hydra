import tensorflow as tf
import tensorflow_text as text


def _parse_tfrecord_function(example, lookup_table):
    example_fmt = {
            'opcodes': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
    parsed = tf.io.parse_single_example(example, example_fmt)
    tokenizer = text.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(parsed['opcodes'])
    IDs = lookup_table.lookup(tokens)
    return IDs, parsed['label']


def make_dataset(filepath, lookup_table, SHUFFLE_BUFFER_SIZE=1024, BATCH_SIZE=32, EPOCHS=5):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.map(lambda x: _parse_tfrecord_function(x, lookup_table))
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset

