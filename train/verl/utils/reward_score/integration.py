"""
reward_model_integration.py

This module implements a reward model for integration problems.
The candidate solution is expected to be an antiderivative provided (optionally)
in the following format:

    <answer> ... </answer>

The reward model:
  1. Extracts the candidate solution (using a strict or flexible method).
  2. Preprocesses the candidate string to remove common LaTeX delimiters and extraneous constants.
  3. Parses both the candidate solution and the ground truth into sympy expressions.
  4. Evaluates both expressions at three (random) x-points.
  5. To cancel any constant of integration, it subtracts the evaluation at a base point.
  6. Returns full points if all evaluations agree within tolerance, or a partial score otherwise.

Example usage with an integration problem:
    candidate_solution = "<answer>\\ln(x)</answer>"
    ground_truth_solution = "\\ln(x)"
    reward = compute_score(candidate_solution, ground_truth_solution, method='strict', tol=1e-5)
    print("Reward:", reward)
"""

import re
import random
import sympy as sp

def extract_candidate_solution(solution_str: str, method: str = 'strict') -> str:
    """
    Extracts the candidate integration solution from the provided solution string.
    Also filters out any candidate that directly contains an integration command.
    Accepts both <answer>...</answer> and \boxed{...} formats.
    """
    if not solution_str or not isinstance(solution_str, str):
        return None
        
    assert method in ['strict', 'flexible'], "Method must be 'strict' or 'flexible'"
    candidate = None
    if method == 'strict':
        try:
            # Check for <answer> tags first
            answer_matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.IGNORECASE | re.DOTALL)
            # Check for \boxed{} format
            boxed_matches = re.findall(r"\\boxed{(.*?)}", solution_str, re.DOTALL)
            # Combine matches and take the last one
            matches = answer_matches + boxed_matches
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
    """
    Preprocesses a solution string to remove common LaTeX delimiters and extraneous terms.
    """
    if not solution or not isinstance(solution, str):
        return "0"  # Return a safe default that will parse to 0
        
    try:
        solution = solution.replace(r"\(", "").replace(r"\)", "")
        solution = solution.replace("$", "")
        solution = solution.replace("\\arctan", "atan")
        solution = solution.replace("\\arcsin", "asin")
        solution = solution.replace("\\arccos", "acos")
        solution = solution.replace("^", "**")
        solution = solution.replace("dx", "")
        
        solution = solution.replace("\\ln", "log")
        solution = re.sub(r"\+?\s*C\b", "", solution)
        return solution.strip() or "0"  # Return "0" if empty after processing
    except Exception:
        return "0"

def compute_score(solution_str: str,
                  ground_truth: str,
                  method: str = 'strict',
                  tol: float = 1e-5,
                  score: float = 1.0,
                  format_score: float = 0.05,
                  evaluation_points: list = None,
                  max_response_length: int = None,
                  tokenizer = None) -> float:
    """
    Computes the reward for a candidate integration solution.
    """
    

    # Early returns for invalid inputs
    if not solution_str or not ground_truth:
        return 0.0

    # 1. Extract the candidate solution
    candidate = extract_candidate_solution(solution_str, method=method)
    if not candidate:
        return 0.0

    # 2. Preprocess solutions
    try:
        candidate = preprocess_candidate_solution(candidate)
        ground_truth_processed = preprocess_candidate_solution(ground_truth)

        # 3. Parse expressions
        x = sp.symbols('x')
        candidate_expr = sp.sympify(candidate, locals={'C': 0})
        ground_truth_expr = sp.sympify(ground_truth_processed, locals={'C': 0})

        # 4. Set up evaluation points
        if not evaluation_points:
            evaluation_points = [random.uniform(1, 10) for _ in range(3)]
        
        # Use first point as base
        base_point = evaluation_points[0]
        candidate_base = float(candidate_expr.subs(x, base_point).evalf())
        ground_truth_base = float(ground_truth_expr.subs(x, base_point).evalf())

        # 5. Evaluate at points
        total_points = len(evaluation_points)
        correct_points = 0

        for pt in evaluation_points:
            try:
                cand_val = float(candidate_expr.subs(x, pt).evalf()) - candidate_base
                gt_val = float(ground_truth_expr.subs(x, pt).evalf()) - ground_truth_base
                if abs(cand_val - gt_val) <= tol:
                    correct_points += 1
            except Exception:
                continue

        # 6. Return score
        if correct_points == total_points:
            return score + format_score
        elif candidate:
            return format_score
        else:
            return 0.0
        

    except Exception:
        return 0.0

# Example usage:
if __name__ == "__main__":
    # For example, consider the definite integral problem:
    #    ∫[0 to 1] x dx = 1/2
    # A candidate solution (possibly coming from a model) might be:
    candidate_solution = "Random text before <answer>1/2</answer> Extra text after"
    
    # The ground truth solution is:
    ground_truth_solution = "1/2"
    
    # Compute the reward (this will work for definite integrals since we're just comparing values)
    reward = compute_score(candidate_solution, ground_truth_solution, method='strict', tol=1e-2)
    print("Reward:", reward)
