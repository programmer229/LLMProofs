"""
reward_model_integration.py

This module implements a reward model for integration problems.
The candidate solution is expected to be an antiderivative provided (optionally)
in the following format:

    <answer> ... </answer>

The reward model:
  1. Extracts the candidate solution (using a strict or flexible method).
  2. Preprocesses the candidate string to remove common LaTeX delimiters and extraneous constants.
  3. Parses both the candidate solution and the ground truth (which is the original integral)
     into sympy expressions (the ground truth is automatically integrated).
  4. Evaluates both expressions at three (random) x-points.
  5. To cancel any constant of integration, it subtracts the evaluation at a base point.
  6. Returns full points if all evaluations agree within tolerance, or a partial score otherwise.

Some integrals (e.g., those with singularities or rapidly oscillating functions)
may cause numerical evaluations to hang. To mitigate this, we add a timeout (using
Python's signal module) that skips any test that takes too long.
"""

import re
import random
import sympy as sp
import mpmath as mp
import signal

# Timeout exception for integration evaluation
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Integration evaluation timed out.")

# Set the signal handler (Note: signal.alarm works on Unix-like systems)
signal.signal(signal.SIGALRM, timeout_handler)

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

def preprocess_candidate_solution(solution: str) -> str:
    """
    Preprocesses a solution string to remove common LaTeX delimiters and extraneous terms.
    Returns "0" if the solution is empty or not a string.
    """
    if not solution or not isinstance(solution, str):
        return "0"  # Return a safe default that will parse to 0
        
    try:
        # Remove LaTeX delimiters and dollar signs.
        solution = solution.replace(r"\(", "").replace(r"\)", "")
        solution = solution.replace("$", "")
        # Replace some common LaTeX commands with sympy-compatible ones.
        solution = solution.replace("\\arctan", "atan")
        solution = solution.replace("\\ln", "log")
        # Replace e** notation with exp()
        solution = re.sub(r'e\*\*([^*]+)', r'exp(\1)', solution)
        # Remove any trailing "+ C" or similar constant expressions.
        solution = re.sub(r"\+?\s*C\b", "", solution)
        return solution.strip() or "0"  # Return "0" if empty after processing
    except Exception:
        return "0"

def compute_score(solution_str: str,
                  ground_truth: str,
                  method: str = 'strict',
                  tol: float = 1e-2,
                  score: float = 1.0,
                  format_score: float = 0.05,
                  num_tests: int = 3,
                  timeout_secs: int = 1) -> float:
    """
    Computes the reward for a candidate integration solution using numerical evaluation.
    
    For each of a number of tests:
      - Two random points are selected.
      - The candidate antiderivative is evaluated at these points, and their difference computed.
      - The definite integral of the integrand (extracted from the ground truth) is computed.
      - If the differences are within tolerance, the test passes.
    
    A timeout is imposed on each test evaluation to skip those that hang due to difficult integrals.
    """
    print("--------------------------------")
    
    if not solution_str or not ground_truth:
        return 0.0

    candidate = extract_candidate_solution(solution_str, method=method)
    if not candidate:
        return 0.0

    candidate = preprocess_candidate_solution(candidate)
    ground_truth_processed = preprocess_candidate_solution(ground_truth)
    
    x = sp.symbols('x')
    locals_dict = {
        'x': x,
        'C': 0,
        'integrate': sp.integrate,
        'pi': sp.pi,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'log': sp.log,
        'exp': sp.exp
    }
    
    try:
        candidate_expr = sp.parse_expr(candidate, local_dict=locals_dict)
        # Extract the integrand from ground_truth by removing 'integrate(' and splitting at the comma.
        integrand_str = ground_truth_processed.replace('integrate(', '').split(',')[0]
        integrand_expr = sp.parse_expr(integrand_str, local_dict=locals_dict)
        
        # Create lambda functions for numerical evaluation.
        candidate_func = sp.lambdify(x, candidate_expr, "mpmath")
        integrand_func = sp.lambdify(x, integrand_expr, "mpmath")
        
        is_correct = True
        successful_tests = 0
        for test_num in range(num_tests):
            a_val = random.uniform(-10, 10)
            b_val = random.uniform(-10, 10)
          
            if abs(b_val - a_val) < 1e-3:
                # Skip tests where the evaluation points are too close.
                continue
                
            try:
                # Set an alarm for the timeout.
                signal.alarm(timeout_secs)
                candidate_diff = candidate_func(b_val) - candidate_func(a_val)
                definite_integral = mp.quad(integrand_func, [a_val, b_val])
                # Cancel the alarm.
                signal.alarm(0)
                
                if abs(candidate_diff - definite_integral) > tol:
                    is_correct = False
                    break
                successful_tests += 1
            except TimeoutException as te:
                print(f"Test {test_num + 1}: Timeout during evaluation: {te}")
                signal.alarm(0)
                continue  # Skip this test and try the next one.
            except Exception as e:
                signal.alarm(0)
                print(f"Test {test_num + 1}: Error during evaluation: {str(e)}")
                continue
        
        # Only award full score if at least one test succeeded and all passed.
        final_score = (score + format_score) if (is_correct and successful_tests > 0) else format_score
        print(f"Final score: {final_score}")
        return final_score
    except Exception as e:
        print(f"Error during computation: {str(e)}")
        return 0.0

# Example usage:
if __name__ == "__main__":
    # Example 1: Integral with potential singularity
    # Antiderivative of 1/sqrt(x) is 2*sqrt(x) (ignoring the constant of integration).
    candidate_solution = "<answer>2*sqrt(x)</answer>"
    ground_truth_solution = "integrate(1/sqrt(x), x)"
    
    result = compute_score(candidate_solution, ground_truth_solution, tol=1e-2)
    print("Reward:", result)
    
    # Example 2: A more standard integral (exponential function)
    candidate_solution2 = "<answer>exp(x)</answer>"
    ground_truth_solution2 = "integrate(exp(x), x)"
    
    result2 = compute_score(candidate_solution2, ground_truth_solution2, tol=1e-2)
    print("Reward:", result2)
