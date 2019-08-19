ulimit -n 4096
export NGPUS=1
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "ty_tools/time_lr2x_lam0.2.yaml"
