#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --nnodes=1 --nproc-per-node=4 sample/sample_ddp.py \
--config ./configs/ucf101/ucf101_sample_ddpm256.yaml \
--ckpt  ./share_ckpts/ucf101.pt \
--save_video_path ./test_ddpm256 \
--ddim_eta 1.0
