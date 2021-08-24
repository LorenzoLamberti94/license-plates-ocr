# opens a tfrecord file and prints one element at time every time you press Enter
import tensorflow as tf
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet Training')
    parser.add_argument('-p', '--path', help='path to tfrecord file',
                        default='../../open_images_v4_dataset/train.tfrecord/')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # args.path = "/home/lamberti/work/open_images_v4_dataset/Dataset/train.tfrecord" #overwrite config parse

    for example in tf.python_io.tf_record_iterator(args.path):
        print(tf.train.SequenceExample.FromString(example))
        input("Press Enter to continue...")