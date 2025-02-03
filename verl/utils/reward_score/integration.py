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
    """
    if not solution_str or not isinstance(solution_str, str):
        return None
        
    assert method in ['strict', 'flexible'], "Method must be 'strict' or 'flexible'"
    if method == 'strict':
        try:
            matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.IGNORECASE | re.DOTALL)
            return matches[-1].strip() if matches else None
        except Exception:
            return None
    else:
        return solution_str.strip()

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
                  max_response_length: int = None) -> float:
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
        return score if correct_points == total_points else 0.0

    except Exception:
        return 0.0

# Example usage:
if __name__ == "__main__":
    # For example, consider the antiderivative problem:
    #    ∫1/x dx = log(x) + C
    # A candidate solution (possibly coming from a model) might be:
    candidate_solution = "Random text before <answer>cos(x**2)+ ln(x)+0.333333*x^3 </answer> Extra text after"
    
    # The ground truth solution is provided as:
    ground_truth_solution = "cos(x**2)+ ln(x)+0.333333*x^3 + c"
    
    # Compute the reward (using three random evaluation points)
    reward = compute_score(candidate_solution, ground_truth_solution, method='strict', tol=1e-2)
    print("Reward:", reward)
