"""
Make a copy of this file and modify it for your custom reward function.
Keep in mind that this reward function takes in a batch of trajectories and returns a tensor of scores, unlike
the standard compute_score functions verl provies which take in a single solution and ground truth and return a single score.
"""

from verl.utils.reward_score.utils.llm_judge_base import judge
import re, torch, os, json
import random, sys
from verl.utils.reward_score.integration_numeric import compute_score as compute_score_numeric
import sympy as sp
import logging
from verl.utils.reward_score.utils.llm_judge_utils import extract_judge_score
from verl.utils.reward_score.utils.svg_utils import is_svg_parsable, convert_svg_to_png, convert_svg_to_base64
import ray

logger = logging.getLogger(__name__)

def log_unique(message):
    if not hasattr(log_unique, 'last_message'):
        log_unique.last_message = None
    if message != log_unique.last_message:
        logger.info(message)
        log_unique.last_message = message

def compute_score(solutions_batch, 
                  ground_truth_batch, 
                  valid_response_lengths, 
                  reward_tensor,
                  max_response_length=None,
                  tokenizer=None):

    ############################################################################
    ################### STEP 1: CREATE YOUR PROMPTS ############################
    ############################################################################

    print("Using the SVG compute score function.")
    
    system_prompt = "You are an expert at marking SVG image generation."
    prompt_template = """

    Attached is an SVG image. Follow the following rubric to score the SVG image:

Detailed Rubric (0–100 Points)

Category 1: Structural Validity & Syntax (0–10 points)
0–2:
- The SVG is broken, with missing or invalid tags.
- Does not render properly due to syntax errors.
3–4:
- Some minor syntax issues, but the image mostly renders.
- Some missing closing tags or incorrect attributes.
5–6:
- Properly structured, though some redundant or inefficient code is present.
- The image is mostly functional.
7–8:
- Fully valid SVG structure with clean syntax.
- Well-formed and adheres to SVG specifications.
9–10:
- Highly optimized, well-structured, and cleanly formatted.
- Follows best practices for SVG syntax with no unnecessary elements.

Category 2: Visual Aesthetics & Design Quality (0–10 points)
0–2:
- Image is visually unappealing or lacks coherence.
- Poor use of colors, shapes, and alignment.
3–4:
- Basic design elements are present but lack refinement.
- Poor spacing, alignment, or color choices.
5–6:
- Visually acceptable but could use better balance or composition.
- Colors and elements are somewhat harmonious.
7–8:
- Well-designed, with attention to spacing, colors, and composition.
- Elements are arranged aesthetically and work well together.
9–10:
- Exceptional design quality with a strong sense of aesthetics.
- Colors, balance, and layout create a visually pleasing and professional image.

Category 3: Complexity & Detail (0–10 points)
0–2:
- Too simple, lacking meaningful detail.
- Appears unfinished or overly minimal.
3–4:
- Some effort at detail, but too basic or lacking refinement.
5–6:
- Moderate level of detail; visually interesting but not intricate.
7–8:
- Rich in detail with carefully designed elements.
- Clearly goes beyond basic shapes and adds depth.
9–10:
- Highly intricate with exceptional attention to detail.
- Subtle touches make the image feel polished and sophisticated.

Category 4: Efficient Use of SVG Features (0–10 points)
0–2:
- Inefficient or redundant code (e.g., unnecessary paths, overuse of inline styles).
- Poorly structured elements, using raster-like techniques instead of vector features.
3–4:
- Some optimization but excessive use of unnecessary groups or redundant attributes.
5–6:
- Reasonably optimized, though there is room for improvement.
- Uses some SVG features like `defs`, `use`, or `path` efficiently.
7–8:
- Well-optimized, using reusable elements and proper structuring.
- Avoids unnecessary duplication or excessive markup.
9–10:
- Highly efficient and elegant use of SVG features.
- Uses `defs`, `symbols`, `gradients`, `filters`, or `clipPath` optimally.

Category 5: Scalability & Resolution Independence (0–10 points)
0–2:
- The image scales poorly and loses quality at different sizes.
- Elements become unreadable or distorted when resized.
3–4:
- Some elements scale well, but parts may break or look awkward.
5–6:
- Generally scalable, though some minor issues at extreme sizes.
7–8:
- Works well across different sizes without distortion.
- Proper use of `viewBox` and responsive design elements.
9–10:
- Perfectly scalable with no loss of quality.
- Fully responsive and adapts cleanly at any resolution.

Category 6: Use of Colors & Gradients (0–10 points)
0–2:
- Colors are clashing, poorly chosen, or completely absent.
- No meaningful use of shading or gradients.
3–4:
- Basic color selection but lacks refinement or balance.
5–6:
- Acceptable color choices with some attention to contrast.
7–8:
- Thoughtful and appealing color selection.
- Smooth gradients or shading add depth.
9–10:
- Masterful use of color, creating depth, contrast, and harmony.
- Subtle gradients, shadows, or highlights enhance the visual appeal.

Category 7: Use of Paths & Shapes (0–10 points)
0–2:
- Poorly constructed paths and shapes.
- Over-reliance on basic primitives without refinement.
3–4:
- Some structured paths, but lacks smoothness or precision.
5–6:
- Good use of paths and curves, though some inefficiencies.
7–8:
- Excellent use of vector paths with smooth, clean lines.
- Shapes are well-formed and purposeful.
9–10:
- Masterful use of `path` elements, Bezier curves, and complex vector techniques.
- Every shape and stroke is intentional and refined.

Category 8: Animation & Interactivity (0–10 points)
0–2:
- No animation or interactivity where expected.
- If animation is present, it is broken or choppy.
3–4:
- Some basic animation, but it is unpolished or inconsistent.
5–6:
- Functional animations but could be smoother or more engaging.
7–8:
- Well-executed animation with smooth transitions.
- Meaningful interactivity adds to the design.
9–10:
- Outstanding use of animation and interactivity.
- Uses `animate`, `SMIL`, or CSS animations effectively.

Category 9: Faithfulness to Prompt (0–10 points)
0–2:
- Image does not resemble what was requested.
- Key elements are missing or completely inaccurate.
3–4:
- Some aspects match the prompt, but major features are incorrect.
5–6:
- Moderately faithful to the prompt; most elements are present but may be misinterpreted.
7–8:
- Strong alignment with the prompt, with only minor deviations.
- Includes most requested details with reasonable accuracy.
9–10:
- Perfectly captures the intent and details of the prompt.
- High accuracy in representing requested elements, style, and composition.

Category 10: Overall Creativity & Originality (0–10 points)
0–2:
- Generic, unoriginal, or copied design.
- Lacks uniqueness or creativity.
3–4:
- Some originality but relies on common templates or styles.
5–6:
- Moderately unique; shows an attempt at creative expression.
7–8:
- Strong originality with a distinct artistic voice.
9–10:
- Exceptionally creative and original.
- The SVG stands out as a unique, memorable work.

Scoring & Guidelines
Each of the 10 categories is worth up to 10 points, for a total of 0–100 possible points.
Use the descriptors in each category to choose an appropriate integer score from 0 to 10 (no half points).
Sum the category scores for a total mark out of 100.

Because this is a high-standard evaluation, scores above 90 should be exceedingly rare, reserved for exceptionally well-crafted, near-professional SVGs.

Output your score in this exact format: 
<JUDGE_SCORE>SCORE</JUDGE_SCORE>
"""


    extracted_solutions = [extract_candidate_solution(sol) for sol in solutions_batch]
    base64_svg_solutions = [convert_svg_to_base64(sol) for sol in extracted_solutions]

    # Select prompts and base64_svg_solutions to pass to the LLM (the ones where the base64_svg_solution is not -1)
    # This is because the LLM will return a score for each SVG, and we only want to return the scores for the valid SVGs
    valid_base64_svg_solutions = [sol for sol in base64_svg_solutions if sol != -1]
    valid_svg_indices = [i for i, sol in enumerate(base64_svg_solutions) if sol != -1]

    prompts = [prompt_template]*len(base64_svg_solutions)
    breakpoint()
    assert len(prompts) == len(valid_base64_svg_solutions), "Prompts and valid_base64_svg_solutions must be the same length."

    ############################################################################
    ################### STEP 2: PASS TO THE LLM JUDGE ##########################
    ############################################################################
    
    local_model = False # We want to use the API model
    async_reward = False # We want to use the synchronous reward
    api_model = "gpt-4o-mini-2024-07-18"
    client_service = "openai"
    max_tokens = 2000
    temperature = 0.7

    try:
        judge_responses = judge(model=api_model,  # Either model name or path to model 
                                client_service=client_service,
                                system_prompt=system_prompt,
                                prompts=prompts,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                local_model=local_model,
                                png_base64_images=valid_base64_svg_solutions,
                                async_reward=async_reward)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Judge responses: {judge_responses}")
        ray.shutdown()
        sys.exit()
            
    ############################################################################
    ################### STEP 3: PARSE JUDGE RESPONSE #########################
    ############################################################################

    # Print 10 random responses for debugging
    num_samples = min(10, len(judge_responses))
    sample_indices = random.sample(range(len(judge_responses)), num_samples)
    print("\nSample of judge responses:")
    for idx in sample_indices:            
        print(f"\nSolution {idx}:")
        print("-" * 80)
        print(solutions_batch[idx])
        print("-" * 80)
        print(f"\nJudge Response {idx}:")
        print(judge_responses[idx])
        print("-" * 80)

    judge_scores = [extract_judge_score(response) for response in judge_responses]
    
    # Place the judge scores back in the right indices in total_scores according to the valid_svg_indices, and set the rest to 0
    total_scores = [0]*len(solutions_batch)
    for i, score in enumerate(judge_scores):
        total_scores[valid_svg_indices[i]] = score

    # Step 4: Convert the scores to a reward tensor
    for i, score in enumerate(total_scores):
        reward_tensor[i, valid_response_lengths[i] - 1] = score
    
    ############################################################################
    ################### STEP 5: LOGGING EXTRA METRICS #######################
    ############################################################################

    extra_logs_path = "/home/ubuntu/o1-replication-central/CustomTinyZero/checkpoints/svg_judge_experiments/qwen2.5_7b_svg_gpt4o_mini"


    logged_judge_responses = ["Did not call judge"]*len(solutions_batch)
    for i, idx in enumerate(valid_svg_indices):
        logged_judge_responses[idx] = judge_responses[i]

    print(f"Valid SVG indices: {valid_svg_indices}")
    print(f"Total scores: {total_scores}")
    ray.shutdown()
    sys.exit()

    # Create dictionary mapping question IDs to details
    question_details = {}
    for idx in range(len(solutions_batch)):
        question_id = f"q{idx+1}"
        question_dict = {
            "model_solution": solutions_batch[idx],
            "extracted_solution": extracted_solutions[idx],
            "base64_solution": base64_svg_solutions[idx],
            "judge_response": logged_judge_responses[idx],
            "total_score": total_scores[idx]
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
            matches = re.findall(r"<svg_image>(.*?)</svg_image>", solution_str, re.IGNORECASE | re.DOTALL)
            candidate = matches[-1].strip() if matches else None
        except Exception:
            return None
    else:
        candidate = solution_str.strip()

    return candidate