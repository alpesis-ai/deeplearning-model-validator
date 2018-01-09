import sys


class Caffe:

    def __init__(self, caffe_path, net_pt, model, out_path):
        self.caffe_path = caffe_path
        self.net_pt = net_pt
        self.model = model
        self.out_path = out_path


    def extract(self):
        sys.path.append(self.caffe_path)
        import caffe
        
        net = caffe.Net(self.net_pt, self.model, caffe.TEST)
       
