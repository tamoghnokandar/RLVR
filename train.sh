python -m venv grpo
source grpo/bin/activate
pip install uv
uv pip install transformers trl vllm bitsandbytes deepspeed wandb peft datasets 
pip install -U datasets
pip install "trl[vllm]"
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --tensor_parallel_size 1
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup accelerate launch --num-processes 4 --config-file deepspeed_zero3.yaml without_external_rewards_grpo.py > output_grpo.log 2>&1 &
