import re
import random


def extract_solution(solution_str):
    # Remove everything before the first "Assistant:"
 

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    if final_answer is not None:
        try:
            int_final_answer = int(final_answer)
        except ValueError:
            final_answer = None
    return final_answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for chess moves.

    Args:
        solution_str: the solution text containing the chess move
        ground_truth: comma-separated string of valid chess moves
        method: unused parameter kept for consistency
        format_score: score given for wrong but valid format
        score: score given for correct move
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Valid moves: {ground_truth} | Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if do_print:
            print(f"No answer found")
        return 0
    else:
        # Split ground truth into list of valid moves and strip whitespace
        valid_moves = [move.strip() for move in ground_truth.split(',')]
        if answer in valid_moves:
            if do_print:
                print(f"Correct move: {answer}")
            return score
        else:
            if do_print:
                print(f"Incorrect move {answer} | Valid moves: {ground_truth}")
            return format_score
