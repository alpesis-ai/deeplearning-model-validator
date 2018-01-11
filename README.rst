##############################################################################
Deep Learning Model Validator
##############################################################################


Frameworks Support

- Tensorflow
- Caffe

==============================================================================
Features
==============================================================================

- extractor: to extract weights and layer outs from the model;
- data: to convert the extracted weights and layer outs from extractor;
- converter: to convert the data to the specific weights and layer outs;
- visualizer: to visualize the weights and layer outs.

==============================================================================
Getting Started
==============================================================================


Caffe

::

    # lenet
    $ python validator.main.py \
      --framework caffe \
      --caffepath <path/to/caffe/python> \
      --netproto <path/to/net/prototxt> \
      --model <path/to/caffemodel> \
      --outpath <path/to/outdir>

    # fasterrcnn
    $ python validator.main.py \
      --framework caffe \
      --caffepath <path/to/caffe/python> \
      --modulepath <path/to/fasterrcnn> \
      --netproto <path/to/net/prototxt> \
      --model <path/to/caffemodel> \
      --outpath <path/to/outdir>
