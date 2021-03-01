export CUDA_VISIBLE_DEVICES=0,1,2,3
python projects/SparseRCNN/train_net.py --num-gpus 4 \
    --config-file projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml \
    OUTPUT_DIR ./ins_seg/