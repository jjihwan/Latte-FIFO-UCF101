#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nnodes=1 --nproc-per-node=2 sample/sample_ddp.py \
--config ./configs/ucf101/ucf101_sample_ddim256.yaml \
--ckpt  ./share_ckpts/ucf101.pt \
--save_video_path ./test_ddim256
