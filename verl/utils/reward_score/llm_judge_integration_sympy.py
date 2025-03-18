"""
Make a copy of this file and modify it for your custom reward function.
Keep in mind that this reward function takes in a batch of trajectories and returns a tensor of scores, unlike
the standard compute_score functions verl provies which take in a single solution and ground truth and return a single score.
"""

from verl.utils.reward_score.llm_judge_base import judge
import re, torch, os, json
import random, sys
from verl.utils.reward_score.integration_numeric import compute_score as compute_score_numeric
import sympy as sp

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
    api_model = "Qwen/Qwen2.5-7B-Instruct"
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
    correctly_formatted = [correct_formatting(sol) for sol in processed_solutions]
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
    ################### STEP 4: LOGGING EXTRA METRICS #######################
    ############################################################################

    extra_logs_path = "/home/ubuntu/o1-replication-usmid/CustomTinyZero/checkpoints/llmjudge_experiments/qwen2.5_7b_integration_sympyscore"

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
            "gold_score": gold_scores[idx]
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

def preprocess_candidate_solution(solution: str) -> str:
    solution = re.sub(r'\*\s*\.\.\.', '', solution)
    solution = solution.replace("...", "")
    solution = solution.replace(r"\(", "").replace(r"\)", "")
    solution = solution.replace("$", "")
    solution = solution.replace("\\arctan", "atan")
    solution = solution.replace("\\arcsin", "asin")
    solution = solution.replace("\\arccos", "acos")
    solution = solution.replace("arccos", "acos")
    solution = solution.replace("arcsin", "asin")
    solution = solution.replace("arctan", "atan")
    solution = solution.replace("e^", "2.718**")
    solution = solution.replace("^", "**")
    solution = solution.replace("\\ln", "log")
    solution = re.sub(r'e\*\*([^*]+)', r'exp(\1)', solution)
    solution = re.sub(r"\+?\s*C\b", "", solution)
    return solution.strip() or "0"

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

def correct_formatting(solution: str) -> bool:
    x, k = sp.symbols('x k')
    locals_dict = {
        'x': x,
        'k': k,
        'C': 0,
        'integrate': sp.integrate,
        'Sum': sp.Sum,   # Use Sum for symbolic summation.
        'sum': sp.Sum,   # Allow 'sum' to be an alias.
        'pi': sp.pi,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'log': sp.log,
        'exp': sp.exp,
        'sqrt': sp.sqrt,
        'atan': sp.atan,
        'asin': sp.asin,
        'acos': sp.acos
    }
    
    try:
        candidate = preprocess_candidate_solution(solution)
        candidate_expr = sp.parse_expr(candidate, local_dict=locals_dict)
        candidate_func = sp.lambdify(x, candidate_expr, "mpmath")
        test_result = candidate_func(10)
    except NameError:
        return False
    except SyntaxError:
        return False
    except Exception:
        return True
    return True