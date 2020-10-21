import argparse
import tensorflow as tf
import os
project_path = os.path.dirname(os.path.realpath("../../../"))
import sys
import csv
sys.path.append(project_path)
from metaphor.metaphor_engine import MetaPHOR
from src.method.utils import load_vocabulary, serialize_hydra_example


def dataset_to_tfrecords(pe_filepath,
                         tfrecords_filepath,
                         labels_filepath,
                         opcodes_vocabulary_mapping_filepath,
                         bytes_vocabulary_mapping_filepath,
                         max_mnemonics=50000,
                         max_bytes=2000000):

    tfwriter = tf.io.TFRecordWriter(tfrecords_filepath)
    opcodes_vocabulary_mapping = load_vocabulary(opcodes_vocabulary_mapping_filepath)
    bytes_vocabulary_mapping = load_vocabulary(bytes_vocabulary_mapping_filepath)

    i = 0
    with open(labels_filepath, "r") as labels_file:
        reader = csv.DictReader(labels_file, fieldnames=["Id",
                                                           "Class"])
        reader.__next__()
        for row in reader:
            print("{};{}".format(i, row['Id']))
            metaPHOR = MetaPHOR(pe_filepath + row['Id'] + ".asm")

            # Extract opcodes
            opcodes = metaPHOR.get_opcodes_data_as_list(opcodes_vocabulary_mapping)
            if len(opcodes) < max_mnemonics:
                while len(opcodes) < max_mnemonics:
                    opcodes.append("PAD")
            else:
                opcodes = opcodes[:max_mnemonics]
            raw_mnemonics = " ".join(opcodes)

            # Extract bytes
            bytes_sequence = metaPHOR.get_hexadecimal_data_as_list()
            for i in range(len(bytes_sequence)):
                if bytes_sequence[i] not in bytes_vocabulary_mapping.keys():
                    bytes_sequence[i] = "UNK"
            if len(bytes_sequence) < max_bytes:
                while len(bytes_sequence) < max_bytes:
                    bytes_sequence.append("PAD")
            else:
                bytes_sequence = bytes_sequence[:max_bytes]
            raw_bytes_sequence = " ".join(bytes_sequence)

            # Extract APIs
            feature_vector = metaPHOR.count_windows_api_calls()

            example = serialize_hydra_example(raw_mnemonics,
                                              raw_bytes_sequence,
                                              feature_vector,
                                              int(row['Class']) - 1)
            tfwriter.write(example)
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HYDRA TFWriter Script')
    parser.add_argument("pe_filepath",
                        type=str,
                        help="Filepath describing the location of the pe files in asm format")
    parser.add_argument("tfrecords_filepath",
                        type=str,
                        help="Where the TFRecord files will be stored")
    parser.add_argument("labels_filepath",
                        type=str,
                        help="CSV filepath containing the ID and class of each PE file in pe_filepath")
    parser.add_argument("opcodes_vocabulary_mapping_filepath",
                        type=str,
                        help="Filepath describing the vocabulary mapping between mnemonics and IDs")
    parser.add_argument("bytes_vocabulary_mapping_filepath",
                        type=str,
                        help="Filepath describing the vocabulary mapping between bytes and IDs")
    parser.add_argument("--max_opcodes",
                        type=int,
                        help="Maximum number of mnemonics per file",
                        default=50000)
    parser.add_argument("--max_bytes",
                        type=int,
                        help="Maximum number of bytes per file",
                        default=2000000)
    args = parser.parse_args()
    dataset_to_tfrecords(args.pe_filepath,
                        args.tfrecords_filepath,
                        args.labels_filepath,
                        args.opcodes_vocabulary_mapping_filepath,
                        args.bytes_vocabulary_mapping_filepath,
                        args.max_opcodes,
                        args.max_bytes)