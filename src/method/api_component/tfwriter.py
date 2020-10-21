import argparse
import tensorflow as tf
import os
project_path = os.path.dirname(os.path.realpath("../../../"))
import sys
import csv
sys.path.append(project_path)
from metaphor.metaphor_engine import MetaPHOR
from src.method.utils import serialize_apis_example


def dataset_to_tfrecords(pe_filepath,
                         tfrecords_filepath,
                         labels_filepath):

    tfwriter = tf.io.TFRecordWriter(tfrecords_filepath)

    i = 0
    with open(labels_filepath, "r") as labels_file:
        reader = csv.DictReader(labels_file, fieldnames=["Id",
                                                           "Class"])
        reader.__next__()
        for row in reader:
            print("{};{}".format(i, row['Id']))
            metaPHOR = MetaPHOR(pe_filepath + row['Id'] + ".asm")
            feature_vector = metaPHOR.count_windows_api_calls()

            example = serialize_apis_example(feature_vector, int(row['Class']) - 1)
            tfwriter.write(example)
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='API-based TFWriter Script')
    parser.add_argument("pe_filepath",
                        type=str,
                        help="Filepath describing the location of the pe files in asm format")
    parser.add_argument("tfrecords_filepath",
                        type=str,
                        help="Where the TFRecord files will be stored")
    parser.add_argument("labels_filepath",
                        type=str,
                        help="CSV filepath containing the ID and class of each PE file in pe_filepath")
    args = parser.parse_args()
    dataset_to_tfrecords(args.pe_filepath,
                        args.tfrecords_filepath,
                        args.labels_filepath)