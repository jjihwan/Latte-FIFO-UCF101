#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
torchrun --nnodes=1 --nproc-per-node=8 sample/sample_ddp_fifo.py \
--config ./configs/ucf101/ucf101_sample_fifo.yaml \
--ckpt  ./share_ckpts/ucf101.pt \
--save_video_path ./test_fifo_n1 \
--num_partitions 1