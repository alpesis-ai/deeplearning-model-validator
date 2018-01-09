import tensorflow as tf


if __name__ == '__main__':

    graph = '/home/kelly/Studio/westwell/DeepBrain/zoey/tf-frcnn-62-zoey/weights/zf_float_200000_around/sf/ZFnet_iter_200000.ckpt.meta'
    model = '/home/kelly/Studio/westwell/DeepBrain/zoey/tf-frcnn-62-zoey/weights/zf_float_200000_around/sf/ZFnet_iter_200000.ckpt'

    sess = tf.Session()
    saver = tf.train.import_meta_graph(graph)
    saver.restore(sess, model)

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    conv1 = all_vars[0]
    bias1 = all_vars[1]
    conv_w1, bias_1 = sess.run([conv1, bias1])
    print conv_w1

    net = caffe.Net('path/to/conv.prototxt', caffe.TEST)
    net.params['conv_1'][0].data[...] = conv_w1
    net.params['conv_1'][1].data[...] = bias_1
    

    net.save('modelfromtf.caffemodel')
