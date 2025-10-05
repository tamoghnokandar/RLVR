# scp -i private_key.pem -P 18600 eval_grpo.py root@78.130.201.2:/root
# python -m venv grpo_evaluate
# source grpo_evaluate/bin/activate
# pip install uv
# uv pip install "sglang[all]>=0.5.1.post3"
# uv pip install transformers datasets
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import json
from tqdm import tqdm
import re
# launch the offline engine
import asyncio
import io
import os

import requests
import sglang as sgl

from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge
from datasets import load_dataset
import ast


if is_in_ci():
    import patch
else:
    import nest_asyncio

    nest_asyncio.apply()

# --- 6. Process Outputs and Evaluate ---
def canonicalize_groups(groups):
    """Sorts words within each group and sorts the groups themselves
    to allow for direct comparison, regardless of order."""
    if not isinstance(groups, list):
        return None
    try:
        # Sort words inside each group
        sorted_word_groups = [sorted(group) for group in groups]
        # Sort the groups themselves (represented as tuples to be hashable)
        return sorted(tuple(group) for group in sorted_word_groups)
    except (TypeError, AttributeError):
        # Handle cases where the prediction is not a list of lists
        return None
def structured_reward(generated_lists, ground_truth_lists):
  """
  Calculates a structured reward based on element, list, and goal-level achievements.
  """
  total_reward = 0
  num_perfect_lists = 0

  for gen_list, gt_list in zip(generated_lists, ground_truth_lists):
    gen_set = set(gen_list)
    gt_set = set(gt_list)
    print(gen_set, gt_set)


    # Check for a perfect list match
    if gen_set == gt_set:
      total_reward += 2  # List-level bonus
      num_perfect_lists += 1
  
    # Element-level rewards and penalties
    correct_elements = len(gen_set.intersection(gt_set))
    incorrect_elements = len(gen_set.difference(gt_set))
    total_reward += correct_elements * 0.25  # Reward for correct elements
#   total_reward -= incorrect_elements * 1 # Penalty for incorrect elements

  # Check for the ultimate goal

  if num_perfect_lists == len(ground_truth_lists):
    total_reward += 5 # Goal-level reward

  return total_reward

if __name__ == "__main__":

    dataset = load_dataset("tm21cy/NYT-Connections", split='train')
    reasoning_start = "<think>" # Acts as <think>
    reasoning_end   = "</think>"   # Acts as </think>
    solution_start  = "<answer>"
    solution_end    = "</answer>"

    system_prompt = \
    f"""You will be playing a categorization game. You will be given a list of 16 words that are all loosely related. However, these words can be grouped into 4 distinct categories of 4 words each, such that the words in each group are more closely related to each other than to the rest. Your task is to identify these 4 groups and return them in the following JSON format: [[\"word1\", \"word2\", \"word3\", \"word4\"], [\"word5\", \"word6\", \"word7\", \"word8\"]].
    Important rules:
    - Each word must appear in exactly one group.
    You will be provided with an example on how to solve it. Provide your answer between {solution_start} and {solution_end}. Don't forget to add quotes on each word and never add anything extra. Think about the problem and provide your reasoning steps between {reasoning_start} and {reasoning_end}."""
    one_shot_prompt = \
    f"""{reasoning_start}The first group is "AWESOME", since they are all words related to awesomeness, although they can also describe something else, it is better to rank them like this. The second group is "VARIETY", the third one is "GIST", since the words are related to getting gist of things, and the last one is "FRIED APPETIZER INFORMALLY", since those words are used to refer to fried foods informally. Understood, now I will output json.{reasoning_end}{solution_start}[["COOL", "NICE", "SICK", "SWEET"], ["KIND", "SORT", "STYLE", "TYPE"], ["DRIFT", "IDEA", "MESSAGE", "POINT"], ["RING", "STICK", "TENDER", "WING"]]{solution_end}"""
    dataset = dataset.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\"KIND\", \"DRIFT\", \"TENDER\", \"NICE\", \"IDEA\", \"WING\", \"SORT\", \"RING\", \"TYPE\", \"STICK\", \"STYLE\", \"SWEET\", \"MESSAGE\", \"SICK\", \"COOL\", \"POINT\""},
            {"role": "assistant", "content": one_shot_prompt},
            {"role": "user",   "content": ", ".join([f'"{word}"' for word in x["words"]])},
        ],
        "answer": [words['words'] for words in x["answers"]],
    })

    split_ratio = 0.1
    split_index = int(len(dataset) * (1 - split_ratio))

    train_dataset = dataset.select(range(split_index))
    eval_dataset = dataset.select(range(split_index, len(dataset)))

    # IMPORTANT: Replace this with the path to your fine-tuned model
    # or any other instruction-following model you want to evaluate.
    MODEL_PATH = "outputs-merged"

    # Load the tokenizer to format the prompts
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load the LLM with vLLM
    # For multi-GPU, change tensor_parallel_size to the number of GPUs
    llm = sgl.Engine(model_path=MODEL_PATH)

    # --- 3. Prepare Prompts and Ground Truths ---
    # Convert the chat-style prompts into a single string for the model
    prompts = [
        tokenizer.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
        for item in eval_dataset
    ]
    ground_truths = [item["answer"] for item in eval_dataset]


    # --- 4. Configure Sampling Parameters ---
    # We want the model to stop right after it generates the solution.
    sampling_params = {"temperature": 0.0, "top_p": 1.0, "stop": [tokenizer.eos_token], "skip_special_tokens" : False, "max_new_tokens" : 10000}
    print(f"Using SamplingParams: {sampling_params}")


    # --- 5. Run vLLM Batch Inference ---
    print("\nStarting vLLM batch inference...")
    outputs = llm.generate(prompts, sampling_params)
    print("outputs", outputs)
    print("Inference complete.")




    correct_predictions = 0
    results = []

    print("\nProcessing and evaluating results...")
    for i, output in tqdm(enumerate(outputs), total=len(outputs)):
        generated_text = output['text']


        # Extract the JSON content between <SOLUTION> and </SOLUTION>
        try:
            # Find the content after the start token
            solution_end_regex = r"</answer>[\s]{0,}" + \
            "(?:" + re.escape(tokenizer.eos_token) + ")?"
            match_format = re.compile(
            rf"{reasoning_end}.*?"\
            rf"{solution_start}(.+?){solution_end_regex}"\
            rf"[\s]{{0,}}$",
            flags = re.MULTILINE | re.DOTALL
    )
            solution_text = ast.literal_eval(match_format.findall(generated_text)[0].strip())

            # The stop token ensures we don't have text after the solution
            predicted_groups = solution_text
        except (json.JSONDecodeError, IndexError):
            # Handle cases where the model output is malformed
            print("GENERATED_TEXT", generated_text)
            print("TRUE TEXT", ground_truths[i])
            predicted_groups = None

        # Get the ground truth for comparison
        true_groups = ground_truths[i]

        # Canonicalize both prediction and ground truth for fair comparison
        canonical_prediction = canonicalize_groups(predicted_groups)
        canonical_truth = canonicalize_groups(true_groups)

        is_correct = (canonical_prediction == canonical_truth)
        # print("Rewards", structured_reward(canonical_prediction, canonical_truth))
        if is_correct:
            correct_predictions += 1

        results.append({
            "id": i,
            "prediction": predicted_groups,
            "ground_truth": true_groups,
            "is_correct": is_correct,
            "raw_output": generated_text
        })

    # --- 7. Display Results ---
    print("\n--- Evaluation Summary ---")
    accuracy = (correct_predictions / len(eval_dataset)) * 100
    print(f"Total Samples: {len(eval_dataset)}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Print a few examples (e.g., the first 3)
    print("\n--- Example Outputs ---")
    for i in range(min(3, len(results))):
        print(f"\n--- Example {i+1} ---")
        print(f"Prediction: {results[i]['prediction']}")
        print(f"Ground Truth: {results[i]['ground_truth']}")
        print(f"Result: {'CORRECT' if results[i]['is_correct'] else 'INCORRECT'}")
        if not results[i]['is_correct']:
            print(f"Model's Raw Output Fragment: {results[i]['raw_output'][:200]}...")
