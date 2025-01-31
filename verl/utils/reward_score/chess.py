import re
import random


def extract_solution(solution_str):
    # Look for a chess move format inside answer tags
    # This will match standard chess notation like e2e4, e4, e2-e4
    answer_pattern = r'<answer>\s*([a-h][1-8]-?[a-h][1-8]|[a-h][1-8])\s*</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
        # Clean up the move format by removing hyphens
        final_answer = final_answer.replace('-', '')
        return final_answer
    return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1., max_response_length=None):
    """The scoring function for chess moves.

    Args:
        solution_str: the solution text containing the chess move
        ground_truth: comma-separated string of valid moves, first move is considered best
        method: unused parameter kept for consistency
        format_score: score given for wrong but valid format
        score: score given for correct move, with bonus for best move
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
            # Add bonus reward based on move position (first move gets highest bonus)
            move_index = valid_moves.index(answer)
            bonus = 1 * (len(valid_moves) - move_index) / len(valid_moves)
            if do_print:
                print(f"Correct move: {answer} with bonus {bonus}")
            return score + bonus
        else:
            if do_print:
                print(f"Incorrect move {answer} | Valid moves: {ground_truth}")
            return format_score
