"""
Make a copy of this file and modify it for your custom reward function.
Keep in mind that this reward function takes in a batch of trajectories and returns a tensor of scores, unlike
the standard compute_score functions verl provies which take in a single solution and ground truth and return a single score.
"""

from verl.utils.reward_score.utils.llm_judge_base import judge
import re, torch, os, json
import random, sys
from verl.utils.reward_score.utils.llm_judge_utils import extract_judge_score
from verl.utils.reward_score.utils.judge_sync import run_prompts_sync_pool


from verl.utils.reward_score.inference_utils import run_prompts
import asyncio
from typing import Optional, List

"""
Assumptions:
- All data in the dataset have the same data_source (being an LLM as a judge). This is necessary.
- The reward function name must have "llm_judge" in it for it to be considered an LLM as a judge.
- The test parquet file has a data_source which is not the LLM judge (for a proper evaluation). For example, for integration val parquets the data_source is "numeric_integration"
"""


def judge(model: str,  # Either model name or path to model
          client_service: Optional[str],
          system_prompt: Optional[str],
          prompts: List[str],  # The prompt to use for judging
          max_tokens: int,
          temperature: float,
          local_model: bool = False,
          async_reward: bool = False) -> List[str]:
    
    
    # Perform judging using a locally run model
    if local_model:
        pass

    # Perform judging using an API model from inference_utils
    if not local_model:
        judge_responses = asyncio.run(run_prompts(client_service=client_service, 
                                                  model=model,
                                                  system_prompt=system_prompt, 
                                                  prompts=prompts, 
                                                  max_tokens=max_tokens, 
                                                  temperature=temperature))
    
    assert len(judge_responses) == len(prompts), "Judge responses not the same length as list of prompts."
    return judge_responses

def extract_candidate_solution(text):
    """
    Extract everything after </think> tag as the solution.
    If no </think> tag is found, return the original text.
    """
    # Look for </think> tag (case insensitive)
    think_pattern = r'</think>(.*)'
    match = re.search(think_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    else:
        # If no </think> tag found, return the original text
        return text.strip()

def compute_score(solutions_batch, 
                  ground_truth_batch, 
                  valid_response_lengths, 
                  reward_tensor,
                  max_response_length=None,
                  tokenizer=None):

    ############################################################################
    ################### STEP 1: CREATE YOUR PROMPTS ############################
    ############################################################################
    
    system_prompt = """Think Low. You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to mark and assess the provided mathematical solution. You will be given ground truth sample solutions as reference and a marking rubric. You should follow the marking rubric closely marking at a level of rigor. Refer to the ground truth solution to see the level or rigor required for each stage
"""

    prompt_template = """

Instructions
1) Carefully read students solution
2) Apply the rubric systematiacaly checking each marking criterion
3) Award marks basesd on the rpesence and corectness of required steps/reasoning
4) Provide specific feedback explaning why marks wer eawarded or deudcted
5) Give a final score out of 6

======================================================================
### Problem ###
{}
======================================================================
### Ground Truth Solution ###
{}
======================================================================
### Solution to Evaluate ###
{}
======================================================================
**JUDGE_SCORE Output:**
- <JUDGE_SCORE>score</JUDGE_SCORE> if CORRECT
Example: <JUDGE_SCORE>4</JUDGE_SCORE>
"""

    processed_solutions = [extract_candidate_solution(sol) for sol in solutions_batch]
    
    prompts = []
    for sol, gt in zip(processed_solutions, ground_truth_batch):
        # For mathematical proofs, we'll use a generic problem statement
        problem_statement = "Solve the given mathematical problem and provide a complete proof."
        prompt = prompt_template.format(problem_statement, gt, sol)
        prompts.append(prompt)

    ############################################################################
    ################### STEP 2: PASS TO THE LLM JUDGE #######################
    ############################################################################
    
    local_model = False # We want to use the API model
    async_reward = False # We want to use the synchronous reward
    api_model = "gpt5-nano"
    client_service = "openai"
    max_tokens = 16000
    temperature = 0.7

    print("THIS IS THE MODEL", async_reward)
    print("THIS IS THE CLIENT SERVICE", client_service)

    print("About to call judge function...")


    judge_responses = run_prompts_sync_pool(
                        client_service=client_service,
                        model=api_model,
                        system_prompt=system_prompt,
                        prompts=prompts,
                        max_tokens=max_tokens,         # consider something smaller than 16000 unless your model supports it
                        temperature=temperature,
                        # png_base64_images=...,        # only if you actually use images
                        max_workers=8,                  # tune for throughput
                        timeout=60,                     # per-request
                    )
    
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

    # Extract scores directly from judge responses
    total_scores = [extract_judge_score(response) for response in judge_responses]

    # Step 4: Convert the scores to a reward tensor
    for i, score in enumerate(total_scores):
        reward_tensor[i, valid_response_lengths[i] - 1] = score
    
    ############################################################################
    ################### STEP 4: LOGGING EXTRA METRICS ##########################
    ############################################################################
    
    # Optional: Add logging for proof evaluation metrics
    print(f"Batch processed: {len(solutions_batch)} solutions")
    print(f"Average score: {sum(total_scores) / len(total_scores):.3f}")
    print(f"Scores distribution: {total_scores}")

    return reward_tensor


if __name__ == "__main__":
    # Super simple test to see if compute_score runs
    print("Testing compute_score function...")
    
    # Mock data
    solutions_batch = [
        "<think>Let me think about this...</think>This is the actual solution",
        "Solution without think tags"
    ]
    ground_truth_batch = [
        "Ground truth 1",
        "Ground truth 2" 
    ]
    valid_response_lengths = [5, 8]
    
    # Create a simple reward tensor
    import torch
    reward_tensor = torch.zeros(2, 10)
    
    try:
        # This will fail because it tries to call the judge API, but at least we can see if the setup works
        result = compute_score(
            solutions_batch=solutions_batch,
            ground_truth_batch=ground_truth_batch, 
            valid_response_lengths=valid_response_lengths,
            reward_tensor=reward_tensor
        )
        print("✅ compute_score ran successfully!")
        print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"❌ Error running compute_score: {e}")
        print("This is expected if API keys are not set up")
    
    # Test extract_candidate_solution separately
    print("\nTesting extract_candidate_solution...")
    test_cases = [
        "No think tags here",
        "<think>Some thinking process</think>This is the solution after thinking",
        "<THINK>Case insensitive</THINK>Solution after uppercase think",
        "Before <think>Multi\nline\nthinking</think>Final solution here"
    ]
    
    for i, test in enumerate(test_cases):
        result = extract_candidate_solution(test)
        print(f"Test {i+1}: {result[:50]}...")
    
    print("✅ extract_candidate_solution tests completed!")
    print("✅ Solution parsing: Everything after </think> tag is extracted as the solution")