export CUDA_VISIBLE_DEVICES=0,1,2,3
python vis/demo.py --config-file projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml \
  --input /shared/xudongliu/bdd100k/10k/val/* \
  --output show/ins_seg_mask5/ \
  --opts SOLVER.IMS_PER_BATCH 8 MODEL.WEIGHTS /shared/xudongliu/code/f_server/SparseR-CNN/output/ins_seg_mask5/model_0015999.pth