"""
Make a copy of this file and modify it for your custom reward function.
Keep in mind that this reward function takes in a batch of trajectories and returns a tensor of scores, unlike
the standard compute_score functions verl provies which take in a single solution and ground truth and return a single score.
"""

from verl.utils.reward_score.llm_judge_base import judge
import re, torch
import random, sys
from verl.utils.reward_score.integration_numeric import compute_score as compute_score_numeric

def compute_score(solutions_batch, 
                  ground_truth_batch, 
                  valid_response_lengths, 
                  reward_tensor,
                  max_response_length=None,
                  tokenizer=None):

    # Step 1: Create your prompts
    
    system_prompt = "You are an expert at mathematical differentiation."
    prompt_template = "Please check if the following is a valid function: {}. If it is, differentiate it and determine if it is functionally equal to {}. Output <JUDGE_SCORE>1</JUDGE_SCORE> if they are equal. Output <JUDGE_SCORE>0</JUDGE_SCORE> if they are not equal or if it is not a valid function. Ignore constants of integration."

    processed_solutions = [extract_candidate_solution(sol) for sol in solutions_batch]
    processed_ground_truth = [extract_integral(gt) for gt in ground_truth_batch]

    prompts = []
    for sol, gt in zip(processed_solutions, processed_ground_truth):
        prompt = prompt_template.format(sol, gt)
        prompts.append(prompt)

    # Step 2: Pass to the LLM judge with the correct parameters
    
    local_model = False # We want to use the API model
    async_reward = False # We want to use the synchronous reward
    api_model = "Qwen/Qwen2.5-7B-Instruct-Turbo"
    client_service = "together"
    max_tokens = 700
    temperature = 0.7

    judge_responses = judge(model=api_model,  # Either model name or path to model 
                            client_service=client_service,
                            system_prompt=system_prompt,
                            prompts=prompts,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            local_model=local_model,
                            async_reward=async_reward)
            
    # Step 3: Parse the judge response and gather a score for each solution (including formatting score)

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

    total_scores = []
    format_scores = [0.05 if (sol is not None) and (sol is not "") else 0 for sol in processed_solutions]
    correct_scores = [extract_judge_score(response) if format_score > 0 else 0 for response, format_score in zip(judge_responses, format_scores)]
    
    # Only add the correct_score from the LLM judge if the output response is formatted correctly.
    # This way, we don't reward the model for outputting the wrong format.
    total_scores = [format_score + correct_score for format_score, correct_score in zip(format_scores, correct_scores)]

    # Step 4: Convert the scores to a reward tensor

    for i, score in enumerate(total_scores):
        reward_tensor[i, valid_response_lengths[i] - 1] = score

    return reward_tensor

################################################################################
# Extraction functions
################################################################################

def extract_integral(ground_truth: str) -> str:
    return ground_truth[10:-4]

def extract_candidate_solution(solution_str: str, method: str = 'strict') -> str:
    """
    Extracts the candidate integration solution from the provided solution string.
    Also filters out any candidate that directly contains an integration command.
    """

    solution_str = solution_str.split("</instruction>")[-1] if "</instruction>" in solution_str else solution_str
    if not solution_str or not isinstance(solution_str, str):
        return None
        
    assert method in ['strict', 'flexible'], "Method must be 'strict' or 'flexible'"
    candidate = None
    if method == 'strict':
        try:
            matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.IGNORECASE | re.DOTALL)
            candidate = matches[-1].strip() if matches else None
        except Exception:
            return None
    else:
        candidate = solution_str.strip()

    # Filter out candidates that contain the word 'integrate' (in any case)
    if candidate and re.search(r'\bintegrate\b', candidate, re.IGNORECASE):
        return None

    return candidate

def extract_judge_score(response_str: str, method: str = 'strict') -> str:
    """
    Extracts the candidate integration solution from the provided solution string.
    Also filters out any candidate that directly contains an integration command.
    """
    if not response_str or not isinstance(response_str, str):
        return 0
        
    assert method in ['strict', 'flexible'], "Method must be 'strict' or 'flexible'"
    return_score = None
    if method == 'strict':
        try:
            matches = re.findall(r"<JUDGE_SCORE>(.*?)</JUDGE_SCORE>", response_str, re.IGNORECASE | re.DOTALL)
            return_score = matches[-1].strip() if matches else None
        except Exception:
            return 0
    else:
        return_score = response_str.strip()
    
    try:
        return_score = int(return_score)
    except Exception:
        return 0

    return return_score