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
import logging

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

    print("Using the right compute score function.")
    
    system_prompt = "You are an expert at marking creative writing."
    prompt_template = """

    Please check if the following is a valid creative writing piece:
    ---
    {}
    ---

If it is not valid creative writing (e.g., it is blank, nonsensical, plagiarized verbatim, etc.), output:
<JUDGE_SCORE>0</JUDGE_SCORE>

Follow the following rubric to score the creative writing piece:

Detailed Rubric (0–100 Points)
Category 1: Originality & Inventiveness (0–10 points)
0–2:
The work is heavily derivative or cliché.
Very little new or surprising in the concept, style, or approach.
Feels like rehashed ideas with minimal personal stamp.
3–4:
Some attempts at originality, but the piece relies on common tropes or predictable plot elements.
Minor sparks of unique imagery or perspective.
Overall effect still feels quite familiar.
5–6:
Moderately original, with clear attempts to develop unique ideas or scenarios.
Some interesting twists or creative flourishes.
Certain sections stand out as inventive, though not consistently.
7–8:
Strongly inventive and often surprising; the writer's personal vision is evident.
Elements of the story or poem break new ground or subvert expectations.
The overall voice, style, or concept is distinct.
9–10:
Remarkably original, challenging conventional boundaries.
The piece stands out as highly creative and distinctive from most contemporary writing.
A consistent sense of innovation throughout.
Achieving a 10 here is exceptionally rare.
Category 2: Depth & Complexity of Themes (0–10 points)
0–2:
Themes are absent, trivial, or muddled.
Writing may be purely surface-level or lack coherence in its core message.
3–4:
Themes are present but simplistic or underdeveloped.
An attempt at deeper issues, but execution is perfunctory or shallow.
5–6:
Themes are reasonably clear and somewhat explored.
Offers more than a surface reading, though insights may remain basic.
7–8:
Themes are well-defined and approached with complexity.
The work encourages readers to consider multiple layers or perspectives.
Evidence of thoughtful engagement with philosophical, social, or emotional issues.
9–10:
Rich thematic tapestry woven with subtlety and nuance.
Themes resonate profoundly, inviting extended reflection or discussion.
Writer demonstrates deep insight and sophistication in handling big ideas.
Category 3: Character Development (0–10 points)
(For non-narrative forms such as certain types of poetry, "character" can be interpreted broadly as the persona or speaker's voice.)

0–2:
Characters (or persona) feel flat, generic, or inconsistent.
No meaningful growth, motivation, or backstory.
3–4:
Some basic definition of characters, but they remain mostly one-dimensional.
Partial or inconsistent development that fails to engage the reader deeply.
5–6:
Characters have identifiable traits or conflicts.
The writer attempts to give them arcs or emotional journeys, though not fully realized.
7–8:
Characters are significantly developed, with consistent motivations, backgrounds, and internal conflicts.
They exhibit believable growth or change throughout the piece.
9–10:
Characters feel fully alive, with nuanced psychology and layered motivations.
Their voices or perspectives profoundly shape the narrative or poem.
Readers can strongly empathize and see genuine transformation or revelation.
Category 4: Plot / Narrative Arc (0–10 points)
(For poetry or experimental writing, interpret this as the structural movement or "journey" the piece takes the reader on.)

0–2:
Little sense of progression; events or ideas are scattered, incoherent, or incomplete.
3–4:
A basic beginning, middle, and end exist, but transitions may be choppy or underdeveloped.
The structure might be muddled or rely on contrivances.
5–6:
A coherent arc is present, though it may be somewhat predictable or lack dramatic tension.
The work demonstrates some structural planning.
7–8:
A well-crafted arc, whether linear or experimental, that engages and guides the reader effectively.
Pacing and transitions generally feel smooth and purposeful.
9–10:
Exceptionally cohesive and compelling structure, whether traditional or avant-garde.
Each segment builds or resonates with the next, maintaining tension or thematic unity.
The journey feels masterfully orchestrated.
Category 5: Language & Prose Style (0–10 points)
0–2:
Language is frequently awkward, ungrammatical, or unpolished.
Style is inconsistent or jarring in a way that hinders comprehension.
3–4:
Writing is mostly functional but lacks elegance or intentionality in word choice.
Occasional awkward phrasing or tonal shifts.
5–6:
Generally clear and coherent prose; some evidence of stylistic flair.
Vocabulary and syntax are competent but may not consistently shine.
7–8:
Skillful prose that demonstrates strong control of tone, diction, and rhythm.
Word choice often enhances the work's impact; language is precise and evocative.
9–10:
Lush, captivating language with near-professional or professional-level craftsmanship.
Consistent mastery of voice, style, and rhetorical effect.
Each sentence or line feels intentional and compelling.
Category 6: Literary Devices & Techniques (0–10 points)
0–2:
Rare or misapplied use of imagery, metaphor, symbolism, or other devices.
If present, they feel forced or clichéd.
3–4:
Some attempts at devices like simile, metaphor, alliteration, etc., but usage is sporadic.
Devices occasionally enrich the text but often lack refinement.
5–6:
Solid use of literary techniques that generally support the narrative or theme.
Examples of effective imagery or symbolism may appear, though not always consistently.
7–8:
Literary devices are well-chosen and enhance depth, creating vivid or meaningful layers.
Symbols, motifs, or figurative language add complexity without overshadowing clarity.
9–10:
Sophisticated, seamless integration of various literary techniques.
The text is enriched by memorable imagery, resonant symbolism, or subtle figurative language.
Every device feels purposefully placed, contributing to an immersive or profound experience.
Category 7: Clarity, Coherence & Flow (0–10 points)
0–2:
Disjointed writing that is hard to follow.
Paragraphs or stanzas show no logical sequence; numerous confusing transitions.
3–4:
Basic coherence, but sections may jump abruptly or meander.
Readers can follow the main idea with effort, but the text feels uneven or messy.
5–6:
Generally coherent structure at sentence and paragraph/stanza level.
Flow is acceptable, though some transitions may be abrupt or repetitive.
7–8:
Smooth and logical flow that guides the reader well.
Paragraphs or stanzas connect effectively, creating a sense of unity.
9–10:
Incredibly fluid, polished flow; each sentence or stanza transitions naturally to the next.
Readers remain continuously engaged, with minimal friction.
Category 8: Emotional & Intellectual Impact (0–10 points)
0–2:
Writing evokes little to no emotional or intellectual response.
Feels rote or hollow, with minimal resonance.
3–4:
Some emotional moments or ideas spark mild interest, but they are fleeting or underexplored.
The piece tries to engage readers, but the effect is inconsistent.
5–6:
Moderately engaging; readers feel something or think critically at certain points.
With more refinement, it could be deeply affecting.
7–8:
Strongly moves or provokes the reader—whether through emotion, thought, or reflection.
The piece creates a memorable experience.
9–10:
Profoundly stirring on an emotional or intellectual level.
Lingers with the reader after finishing, prompting ongoing reflection or discussion.
Masterfully crafted to connect with deeper human experiences or big questions.
Category 9: Integration of Setting & Atmosphere (0–10 points)
(Even if the setting is minimal or abstract, how well does the writing situate readers in its world/space/mood?)

0–2:
Setting is absent, confused, or contradictory.
Atmosphere is flat or unintentional.
3–4:
There is a sense of place or environment but it's vague or clichéd.
Limited sensory details that do not significantly enhance the piece.
5–6:
Setting and atmosphere are adequately conveyed and somewhat immersive.
Basic sensory descriptions are used to ground the reader.
7–8:
A well-realized environment or mood that meaningfully supports the narrative or theme.
Vivid descriptions engage multiple senses.
9–10:
The setting or atmosphere feels integral, with meticulous or evocative details.
The work's mood or sense of place actively shapes readers' emotional and intellectual experience.
Category 10: Professional Presentation & Polish (0–10 points)
0–2:
Numerous distracting errors in grammar, punctuation, or formatting that impede reading.
Manuscript is unrefined, clearly lacking basic editing.
3–4:
Frequent but not overwhelming mechanical issues.
Writing is understandable but sloppy, suggesting insufficient revision.
5–6:
Some minor errors appear, but they do not severely disrupt comprehension.
The piece could benefit from further proofreading or formatting consistency.
7–8:
Well-edited and generally polished; only a few mistakes here and there.
Formatting is professional and consistent.
9–10:
Impeccably proofread; errors are nearly non-existent.
Professional-level attention to detail in formatting, citations (if needed), and layout.
Demonstrates high standards typical of a graduate or professional setting.
Scoring & Guidelines
Each of the 10 categories is worth up to 10 points, for a total of 0–100 possible points.
Use the descriptors in each category to choose an appropriate integer score from 0 to 10 (no half points).
Sum the category scores for a total mark out of 100.
Because this is a tough graduate-level rubric, achieving scores above 90 should be exceedingly rare, reserved for near-publishable or professional-quality work.

Output your score in this exact format: 
<JUDGE_SCORE>SCORE</JUDGE_SCORE>
"""

    processed_solutions = [extract_candidate_solution(sol) for sol in solutions_batch]

    prompts = []
    for sol in processed_solutions:
        # If solution is None or empty, use a placeholder
        if sol is None or sol.strip() == "":
            prompt = prompt_template.format("[Invalid or empty creative writing piece]")
        else:
            prompt = prompt_template.format(sol)
        prompts.append(prompt)

    ############################################################################
    ################### STEP 2: PASS TO THE LLM JUDGE ##########################
    ############################################################################
    
    local_model = False # We want to use the API model
    async_reward = False # We want to use the synchronous reward
    api_model = "gpt-4o"
    client_service = "openai"
    max_tokens = 4000
    temperature = 0.7

    try:
        judge_responses = judge(model=api_model,  # Either model name or path to model 
                                client_service=client_service,
                                system_prompt=system_prompt,
                                prompts=prompts,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                local_model=local_model,
                                async_reward=async_reward)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Judge responses: {judge_responses}")
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

    total_scores = []
    format_scores = [0.05 if (sol != None) and (sol != "") else 0 for sol in solutions_batch]
    correct_scores = [extract_judge_score(response) if format_score > 0 else 0 for response, format_score in zip(judge_responses, format_scores)]
    
    # Only add the correct_score from the LLM judge if the output response is formatted correctly.
    # This way, we don't reward the model for outputting the wrong format.
    total_scores = [format_score + correct_score for format_score, correct_score in zip(format_scores, correct_scores)]

    # Step 4: Convert the scores to a reward tensor
    for i, score in enumerate(total_scores):
        reward_tensor[i, valid_response_lengths[i] - 1] = score
    
    ############################################################################
    ################### STEP 5: LOGGING EXTRA METRICS #######################
    ############################################################################

    extra_logs_path = "/home/ubuntu/o1-replication/CustomTinyZero/checkpoints/llmjudge_experiments/creative_writting_7b_1"

    # Logging proportion of correctly formatted solutions for this step
    correctly_formatted = [correct_formatting(sol) for sol in solutions_batch]
    num_correctly_formatted = sum(correctly_formatted)

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
            "processed_solution": solutions_batch[idx],
            "processed_ground_truth": ground_truth_batch[idx],
            "judge_response": judge_responses[idx],
            "extracted_judge_score": correct_scores[idx],
            "format_score": format_scores[idx], 
            "total_score": total_scores[idx],
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
            matches = re.findall(r"<writing>(.*?)</writing>", solution_str, re.IGNORECASE | re.DOTALL)
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