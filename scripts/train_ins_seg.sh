export CUDA_VISIBLE_DEVICES=0,1,2,3
python projects/SparseRCNN/train_net.py --num-gpus 4 --dist-url tcp://0.0.0.0:12345 --resume\
    --config-file projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml \
    OUTPUT_DIR ./output/ins_seg_mask5/