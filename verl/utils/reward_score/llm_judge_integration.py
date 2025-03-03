"""
Make a copy of this file and modify it for your custom reward function.
Keep in mind that this reward function takes in a batch of trajectories and returns a tensor of scores, unlike
the standard compute_score functions verl provies which take in a single solution and ground truth and return a single score.
"""

from verl.utils.reward_score.llm_judge_base import judge
import re, torch
import random

def compute_score(solutions_batch, 
                  ground_truth_batch, 
                  valid_response_lengths, 
                  reward_tensor,
                  max_response_length=None,
                  tokenizer=None):

    # Step 1: Create your prompts
    
    system_prompt = "You are an expert at mathematical differentiation."
    prompt_template = "Please differentiate the following function, {} and determine if it is functionally equal to {}. Output <JUDGE_SCORE>1</JUDGE_SCORE> if they are equal, and <JUDGE_SCORE>0</JUDGE_SCORE> if they are not equal."

    processed_solutions = [extract_candidate_solution(sol) for sol in solutions_batch]
    processed_ground_truth = [extract_integral(gt) for gt in ground_truth_batch]

    prompts = []
    for sol, gt in zip(processed_solutions, processed_ground_truth):
        prompt = prompt_template.format(sol, gt)
        prompts.append(prompt)

    # Step 2: Pass to the LLM judge with the correct parameters
    
    local_model = False # We want to use the API model
    async_reward = False # We want to use the synchronous reward
    api_model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    client_service = "together"
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

    correct_scores = [extract_judge_score(response) for response in judge_responses]
    format_scores = [0.05 if sol is not None else 0 for sol in processed_solutions]
    total_scores = [correct + format for correct, format in zip(correct_scores, format_scores)]

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