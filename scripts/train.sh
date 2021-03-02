export CUDA_VISIBLE_DEVICES=0,1,2,3
python projects/SparseRCNN_det/train_net.py --num-gpus 4 \
    --config-file projects/SparseRCNN_det/configs/sparsercnn.res50.100pro.3x.yaml \
    OUTPUT_DIR ./output/det/