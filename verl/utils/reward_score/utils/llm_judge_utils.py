import re

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