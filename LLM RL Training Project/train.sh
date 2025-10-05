#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:5         # Total GPUs for the entire job
#SBATCH --cpus-per-task=16   # Allocate more CPUs for the training step
#SBATCH --job-name=grpo_vllm
#SBATCH --output=logs/grpo_vllm_%j.out
#SBATCH --error=logs/grpo_vllm_%j.err

# Create a logs directory if it doesn't exist
mkdir -p logs

# --- Environment Setup ---
# Activate the virtual environment you created earlier
echo "Activating Python virtual environment..."
source ~/grpo/bin/activate

# BEST PRACTICE: Set WANDB_API_KEY as an environment variable
# You can do this in your ~/.bashrc or pass it with `sbatch --export=WANDB_API_KEY=...`
# If the variable is already exported, this line is not needed.
export WANDB_API_KEY="f7f547ca6185e68d76bb3c9ac443b46ffcaabdae"
wandb login

# --- Job Logic ---

# Define a function to clean up the background server process
cleanup() {
    echo "Cleaning up vLLM server..."
    # The '-' before $VLLM_PID sends the signal to the entire process group
    kill -9 -$VLLM_PID
    wait $VLLM_PID 2>/dev/null
}

# Set a trap to run the cleanup function on exit, failure, or job cancellation
trap cleanup EXIT SIGINT SIGTERM

# --- 1. Launch vLLM server in the background on one GPU ---
# srun will allocate one specific GPU and set CUDA_VISIBLE_DEVICES for this step.
echo "Launching vLLM server in the background..."
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=8 \
    trl vllm-serve \
        --model Qwen/Qwen3-1.7B \
        --tensor_parallel_size 1 &

# Capture the Process ID (PID) of the srun command which is the leader of the process group
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Give the server a moment to start up
sleep 30

# --- 2. Launch training on the remaining GPUs ---
# Request the other 4 GPUs for this step. `accelerate` will handle the multi-GPU setup.
# We launch ONE task, and accelerate starts multiple processes within that allocation.
echo "Launching training script..."
srun --ntasks=1 --gres=gpu:4 --cpus-per-task=16 \
    accelerate launch \
        --num-processes 4 \
        --config_file deepspeed_zero3.yaml \
        without_external_rewards_grpo.py

echo "Training finished."

# The trap will automatically call the cleanup function now to kill the server.
# The `wait` command at the end is no longer strictly necessary because of the trap,
# but can be kept for clarity.
wait
echo "Job completed."