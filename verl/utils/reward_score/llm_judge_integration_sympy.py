"""
Make a copy of this file and modify it for your custom reward function.
Keep in mind that this reward function takes in a batch of trajectories and returns a tensor of scores, unlike
the standard compute_score functions verl provies which take in a single solution and ground truth and return a single score.
"""

from verl.utils.reward_score.utils.llm_judge_base import judge
import re, torch, os, json
import random, sys
from verl.utils.reward_score.integration_numeric import compute_score as compute_score_numeric
import sympy as sp
from verl.utils.reward_score.utils.integration_utils import extract_candidate_solution, extract_integral, preprocess_candidate_solution, sympy_correct_formatting
from verl.utils.reward_score.utils.llm_judge_utils import extract_judge_score

def compute_score(solutions_batch, 
                  ground_truth_batch, 
                  valid_response_lengths, 
                  reward_tensor,
                  max_response_length=None,
                  tokenizer=None):

    ############################################################################
    ################### STEP 1: CREATE YOUR PROMPTS ############################
    ############################################################################
    
    system_prompt = "You are an expert at mathematical differentiation."
    prompt_template = "Please check if the following is a valid function: {}. If it is, differentiate it and determine if it is functionally equal to {}. Output <JUDGE_SCORE>1</JUDGE_SCORE> if they are equal. Output <JUDGE_SCORE>0</JUDGE_SCORE> if they are not equal or if it is not a valid function. Ignore constants of integration."

    processed_solutions = [extract_candidate_solution(sol) for sol in solutions_batch]
    processed_ground_truth = [extract_integral(gt) for gt in ground_truth_batch]

    prompts = []
    for sol, gt in zip(processed_solutions, processed_ground_truth):
        prompt = prompt_template.format(sol, gt)
        prompts.append(prompt)

    ############################################################################
    ################### STEP 2: PASS TO THE LLM JUDGE #######################
    ############################################################################
    
    local_model = False # We want to use the API model
    async_reward = False # We want to use the synchronous reward
    api_model = "gpt-4o-2024-08-06"
    client_service = "openai"
    max_tokens = 1000
    temperature = 0.7

    judge_responses = judge(model=api_model,  # Either model name or path to model 
                            client_service=client_service,
                            system_prompt=system_prompt,
                            prompts=prompts,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            local_model=local_model,
                            async_reward=async_reward)
            
    ############################################################################
    ################### STEP 3: PARSE JUDGE RESPONSE #########################
    ############################################################################

    # Print 10 random responses for debugging
    num_samples = min(10, len(judge_responses))
    sample_indices = random.sample(range(len(judge_responses)), num_samples)
    print("\nSample of judge responses:")
    for idx in sample_indices:
        print(f"\nSolution {idx}:")
        print(solutions_batch[idx])
        print(f"\nGround Truth {idx}:")
        print(ground_truth_batch[idx])
        print(f"\nJudge Response {idx}:")
        print(judge_responses[idx])
        print("-" * 80)

    # Logging proportion of correctly formatted solutions for this step
    correctly_formatted = [sympy_correct_formatting(sol) for sol in processed_solutions]
    num_correctly_formatted = sum(correctly_formatted)


    total_scores = []

    # Uses the gold standard format score instead of the LLM judge's format score.
    correct_scores = [extract_judge_score(response) if format_score else 0 for response, format_score in zip(judge_responses, correctly_formatted)]
    
    # Only add the correct_score from the LLM judge if the output response is formatted correctly.
    # This way, we don't reward the model for outputting the wrong format.
    total_scores = [0.05*float(format_score) + correct_score for format_score, correct_score in zip(correctly_formatted, correct_scores)]

    # Step 4: Convert the scores to a reward tensor
    for i, score in enumerate(total_scores):
        reward_tensor[i, valid_response_lengths[i] - 1] = score
    
    ############################################################################
    ################### STEP 4: LOGGING EXTRA METRICS ##########################
    ############################################################################

    extra_logs_path = "/home/ubuntu/o1-replication-usmid/CustomTinyZero/checkpoints/llmjudge_experiments/r1_distill_7b_ladder_sympyscore_gpt4o"

    # Integration numeric scores (golden scoring metric)
    gold_scores = [compute_score_numeric(solution_str=sol, ground_truth=gt) for sol, gt in zip(solutions_batch, ground_truth_batch)]
    
    # Calculate misclassification error by comparing total_scores and gold_scores
    num_correctly_scored = sum(1 for ts, gs in zip(total_scores, gold_scores) if ts == gs)
    
    custom_metrics = {
        "batch_size": len(solutions_batch),
        "num_correct_sympy_formatting": num_correctly_formatted,
        "num_correctly_scored": num_correctly_scored
    }
    
    metrics_file = os.path.join(extra_logs_path, "failure_metrics.json")
    if not os.path.exists(metrics_file):
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, "w") as f:
            f.write("[]")
    metrics = json.load(open(metrics_file))
    metrics.append(custom_metrics) if isinstance(metrics, list) else json.dump([custom_metrics], open(metrics_file, "w"))
    json.dump(metrics, open(metrics_file, "w"), indent=4)

    # Create dictionary mapping question IDs to details
    question_details = {}
    for idx in range(len(solutions_batch)):
        question_id = f"q{idx+1}"
        question_dict = {
            "model_solution": solutions_batch[idx],
            "ground_truth": ground_truth_batch[idx],
            "processed_solution": processed_solutions[idx],
            "processed_ground_truth": extract_integral(ground_truth_batch[idx]),
            "judge_response": judge_responses[idx],
            "extracted_judge_score": correct_scores[idx],
            "total_reward_score": total_scores[idx],
            "gold_score": gold_scores[idx],
            "format_score": correctly_formatted[idx]
        }
        question_details[question_id] = question_dict

    # Load existing details or create new list
    details_file = os.path.join(extra_logs_path, "question_logs.json")
    if not os.path.exists(details_file):
        existing_details = []
    else:
        with open(details_file, 'r') as f:
            existing_details = json.load(f)

    # Append new details and save
    existing_details.append(question_details)
    with open(details_file, 'w') as f:
        json.dump(existing_details, f, indent=4)

    return reward_tensor