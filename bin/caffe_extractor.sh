CAFFE_PATH="/path/to/caffe/python"
CAFFE_NET_PATH="/path/to/prototxt"
CAFFE_MODEL_PATH="/path/to/caffe/model"
EXTRACTED_WEIGHTS_PATH="/path/to/extracted_weights_dir"
EXTRACTED_OUTS_PATH="/path/to/extracted_outs_dir"


python extractor/caffe_extractor.py \
-c ${CAFFE_PATH} \
-cn ${CAFFE_NET_PATH} \
-cm ${CAFFE_MODEL_PATH} \
-wm ${EXTRACTED_WEIGHTS_PATH} \
-wo ${EXTRACTED_OUTS_PATH} \
