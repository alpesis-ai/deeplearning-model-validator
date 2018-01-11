"""
    Usage:
"""


import sys
import argparse

from extractor.extractor_caffe import CaffeExtractor


def get_args():
    parser = argparse.ArgumentParser("DeepLearning Model Validator")

    parser.add_argument('--framework', dest="framework",
                                       type=str,
                                       required=True,
                                       help='Framework: [caffe, tensorflow]')

    parser.add_argument('--caffepath', dest="caffepath",
                                       type=str,
                                       required=True,
                                       help="Caffe Path")

    parser.add_argument('--modulepath', dest='modulepath',
                                        type=str,
                                        required=False,
                                        help="Extract module path")


    parser.add_argument('--netproto', dest="netproto",
                                      type=str,
                                      required=True,
                                      help="Network prototxt")

    parser.add_argument('--model', dest="model",
                                   type=str,
                                   required=True,
                                   help="Caffemodel path")

    parser.add_argument('--outpath', dest="outpath",
                                     type=str,
                                     required=True,
                                     help="Output path")


    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    if args.framework == 'caffe':
        if (args.modulepath):
            sys.path.append(args.caffepath)
            sys.path.append(args.modulepath)
            sys.path.append(args.modulepath + "lib/")
            sys.path.append(args.modulepath + "tools/")
            import caffe
            import demo

            caffe.set_mode_cpu()
            net = caffe.Net(args.netproto, args.model, caffe.TEST)
        else:
            sys.path.append(args.caffepath)
            import caffe
            net = caffe.Net(args.netproto, args.model, caffe.TEST)

        extractor = CaffeExtractor(net, args.netproto, args.model, args.outpath)
        extractor.extract_model()
        extractor.extract_weights()
        extractor.extract_outputs()
