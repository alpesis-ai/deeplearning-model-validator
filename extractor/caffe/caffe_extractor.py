"""
Caffe Extractor

  Usage:   

    $ workon caffe
    $ python caffe_extractor.py -c <caffe_path> \ 
                                -cn <net_proto> \
                                -cm <model_path> \
                                -wm <extracted_model_path> \
                                -wo <extracted_out_path>
"""

import sys
import ctypes
import struct
import argparse


from tensor_and_model import TensorFloat
from extractor_single import extract_model
from extractor_single import extract_weights
from extractor_single import extract_outputs


def get_args():
    """
    """

    parser = argparse.ArgumentParser("Caffe Extractor")

    parser.add_argument('-c', dest='caffepath',
                              type=str,
                              required=True,
                              help='Caffe path')

    parser.add_argument('-cn', dest='net',
                               type=str,
                               required=True,
                               help='Network prototxt')

    parser.add_argument('-cm', dest='model', 
                               type=str,
                               required=True,
                               help='Caffemodel path')

    parser.add_argument('-wm', dest='modelpath',
                               type=str,
                               required=True,
                               help='Wellframe model path')

    parser.add_argument('-wo', dest='outpath',
                               type=str,
                               required=True,
                               help='Wellframe out path')

    return parser.parse_args()



if __name__ == "__main__":

    args = get_args()
    sys.path.append(args.caffepath)
    import caffe

    net = caffe.Net(args.net, args.model, caffe.TEST)
    extract_model(net, args.modelpath)
    extract_weights(net, args.modelpath)
    net.forward()
    extract_outputs(net, args.outpath)
