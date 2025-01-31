import re
import random

def extract_solution(solution_str):
    """
    Extract the last answer grid from a solution string.
    """
    if not solution_str:
        return None
        
    # Simpler regex to find content between answer tags
    answers = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    
    if not answers:
        return None
        
    last_answer = answers[-1]
    
    # Extract only the grid numbers
    try:
        grid_lines = [line.strip() for line in last_answer.split('\n') if '[' in line]
        grid_numbers = ''.join(filter(str.isdigit, ''.join(grid_lines)))
        return grid_numbers if grid_numbers else None
    except Exception:
        return None

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1., max_response_length=None):
    """
    Scoring function for ARC grid transformation.
    
    Args:
        solution_str (str): The solution text containing the ARC grid
        ground_truth (list): List of lists of integers, representing the correct grid
        method (str, optional): Scoring method (currently unused). Defaults to 'strict'.
        format_score (float, optional): Score given for wrong grid but valid format. Defaults to 0.1.
        score (float, optional): Score given for correctness of move. Defaults to 1.
    
    Returns:
        float: Computed score between 0 and (format_score*4 + score + 1)
    """
    # Extract answer and ground truth
    answer_str = extract_solution(solution_str=solution_str)
    if not isinstance(ground_truth_str := ''.join(filter(str.isdigit, str(ground_truth))), str):
        return 0
        
    # Debug printing (1/16 chance)
    do_print = random.randint(1, 8) == 1
    if do_print:
        print("--------------------------------")
        print(f"Solution grid: {solution_str}")
        print(f"Ground truth string: {ground_truth_str} | Extracted answer string: {answer_str}")
    
    # Score computation
    if answer_str is None:
        if do_print:
            print("No answer found")
        return 0
        
    if len(answer_str) == len(ground_truth_str):
        percent_squares_correct = sum(1 for a, b in zip(answer_str, ground_truth_str) if a == b) / len(answer_str)
        if percent_squares_correct == 1:
            return percent_squares_correct + format_score
        else:
            return format_score
        
    return 0