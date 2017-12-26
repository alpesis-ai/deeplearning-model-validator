CAFFE_PATH="/path/to/py-faster-rcnn/caffe-fast-rcnn/python"
FRCNN_PATH="/path/to/py-faster-rcnn/"
CAFFE_NET_PATH="/path/to/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt"
CAFFE_MODEL_PATH="/path/to/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel"
EXTRACTED_WEIGHTS_PATH="data/fasterrcnn/"
EXTRACTED_OUTS_PATH="data/fasterrcnn/"

python extractor/fasterrcnn_extractor.py \
-c ${CAFFE_PATH} \
-cf ${FRCNN_PATH} \
-cn ${CAFFE_NET_PATH} \
-cm ${CAFFE_MODEL_PATH} \
-wm ${EXTRACTED_WEIGHTS_PATH} \
-wo ${EXTRACTED_OUTS_PATH}

