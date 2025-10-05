#!/bin/bash
apt-get update
apt install tmux -y
python -m venv grpo
source grpo/bin/activate
pip install uv
uv pip install transformers "trl[vllm]" bitsandbytes deepspeed wandb peft datasets 
wandb login f7f547ca6185e68d76bb3c9ac443b46ffcaabdae

source grpo/bin/activate
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --tensor_parallel_size 1

source grpo/bin/activate
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup accelerate launch --num-processes 4 --config-file deepspeed_zero3.yaml without_external_rewards_grpo.py > output_grpo.log 2>&1 &