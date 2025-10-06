python -m venv grpo_evaluate
source grpo_evaluate/bin/activate
pip install uv
uv pip install "sglang[all]>=0.5.3rc0"
uv pip install transformers datasets
python3 eval_grpo.py
