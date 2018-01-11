CAFFE_PATH="/path/to/caffe/python"
MODULE_PATH="/path/to/extra/module"
CAFFE_NET_PROTO="/path/to/prototxt"
CAFFE_MODEL="/path/to/caffe/model"
OUT_PATH="/path/to/outdir"


python validator/main.py \
--framework caffe \
--caffepath ${CAFFE_PATH} \
--modulepath ${MODULE_PATH} \
--netproto ${CAFFE_NET_PROTO} \
--model ${CAFFE_MODEL} \
--outpath ${OUT_PATH} \
