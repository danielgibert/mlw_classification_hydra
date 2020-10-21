import argparse
import tensorflow as tf
import os
project_path = os.path.dirname(os.path.realpath("../../../"))
import sys
import csv
sys.path.append(project_path)
from metaphor.metaphor_engine import MetaPHOR
from src.method.utils import load_vocabulary, serialize_mnemonics_example


def dataset_to_tfrecords(pe_filepath,
                         tfrecords_filepath,
                         labels_filepath,
                         vocabulary_mapping_filepath,
                         max_mnemonics):

    vocabulary_mapping = load_vocabulary(vocabulary_mapping_filepath)
    tfwriter = tf.io.TFRecordWriter(tfrecords_filepath)

    i = 0

    # Training TFRecord
    with open(labels_filepath, "r") as labels_file:
        reader = csv.DictReader(labels_file, fieldnames=["Id",
                                                           "Class"])
        reader.__next__()
        for row in reader:
            print("{};{}".format(i, row['Id']))
            metaPHOR = MetaPHOR(pe_filepath + row['Id'] + ".asm")
            opcodes = metaPHOR.get_opcodes_data_as_list(vocabulary_mapping)

            if len(opcodes) < max_mnemonics:
                while len(opcodes) < max_mnemonics:
                    opcodes.append("PAD")
            else:
                opcodes = opcodes[:max_mnemonics]
            raw_mnemonics = " ".join(opcodes)

            example = serialize_mnemonics_example(raw_mnemonics, int(row['Class'])-1)
            tfwriter.write(example)
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mnemonics-based TFWriter Script')
    parser.add_argument("pe_filepath",
                        type=str,
                        help="Filepath describing the location of the pe files in asm format")
    parser.add_argument("tfrecords_filepath",
                        type=str,
                        help="Where the TFRecord files will be stores")
    parser.add_argument("labels_filepath",
                        type=str,
                        help="CSV filepath containing the ID and class of each PE file in pe_filepath")
    parser.add_argument("vocabulary_mapping_filepath",
                        type=str,
                        help="Filepath describing the vocabulary mapping between mnemonics and IDs")
    parser.add_argument("--max_mnemonics",
                        type=int,
                        help="Maximum number of mnemonics per file",
                        default=50000)
    args = parser.parse_args()
    dataset_to_tfrecords(args.pe_filepath,
                         args.tfrecords_filepath,
                         args.labels_filepath,
                         args.vocabulary_mapping_filepath,
                         args.max_mnemonics)