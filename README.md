# Reinforcement Learning from Verifiable Rewards
In this project, I finetune Qwen3-1.7B model using the Group Relative Policy Optimization (GRPO) algorithm. I also use LORA for finetuning. The dataset was taken from [dataset](https://huggingface.co/datasets/tm21cy/NYT-Connections)

 Both the installation and training are done in one script.
## Training
```bash
bash train.sh
```
After training, we merge the weights of the LORA adapter with the base model.
## Merging
```bash
python3 post_train_merge.py
```
Now, we can evaluate the model on the test set.
## Evaluation
```bash
bash eval.sh
```
