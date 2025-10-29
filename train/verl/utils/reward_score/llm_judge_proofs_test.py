"""
Group-based LLM judge reward for validation/evaluation runs.

This mirrors the training reward but keeps a smaller worker pool and uses the
evaluation model configuration.
"""

from collections import defaultdict
import json
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

from verl.utils.reward_score.utils.judge_sync import run_prompts_sync_pool

DEFAULT_GROUP_POINTS: List[float] = [1.0, 0.6, 0.2]
DEFAULT_GROUP_SIZE: int = 3
DEFAULT_GROUP_ROUNDS: int = 5


def extract_candidate_solution(text: str) -> str:
    """
    Extract everything after </think> tag as the solution.
    If no </think> tag is found, look for <proof> tag instead.
    If neither tag is found, return the original text.
    """
    think_pattern = r"</think>(.*)"
    match = re.search(think_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    proof_pattern = r"<proof>(.*)"
    match = re.search(proof_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text


def _points_for_position(position: int, schedule: Sequence[float]) -> float:
    if 0 <= position < len(schedule):
        return float(schedule[position])
    return 0.0


def _normalize_extra_info(extra_info: object) -> dict:
    if hasattr(extra_info, "item"):
        try:
            extra_info = extra_info.item()
        except Exception:
            pass
    if isinstance(extra_info, dict):
        return extra_info
    return {}


def _extract_question_id(extra_info: object, fallback_idx: int) -> str:
    info_dict = _normalize_extra_info(extra_info)
    for key in ("question_id", "problem_id", "prompt_id", "id"):
        value = info_dict.get(key)
        if value not in (None, ""):
            return str(value)
    split = info_dict.get("split")
    index = info_dict.get("index")
    if split not in (None, "") and index not in (None, ""):
        return f"{split}::{index}"
    return f"sample_{fallback_idx}"


def _parse_group_response(response_text: str, labels: Sequence[str]) -> Tuple[List[int], bool]:
    json_text = (response_text or "").strip()
    start = json_text.find("{")
    end = json_text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        json_text = json_text[start : end + 1]

    ranks_by_label: dict[str, int] = {}
    parsed = False

    if json_text:
        try:
            data = json.loads(json_text)
            placements = data.get("placements", [])
            if isinstance(placements, list):
                for entry in placements:
                    if not isinstance(entry, dict):
                        continue
                    raw_label = entry.get("label")
                    raw_rank = entry.get("rank")
                    if isinstance(raw_label, str) and isinstance(raw_rank, (int, float)):
                        label = raw_label.strip().upper()
                        rank_value = max(1, int(raw_rank))
                        ranks_by_label[label] = rank_value
                        parsed = True
        except json.JSONDecodeError:
            parsed = False

    default_rank = len(labels)
    ranks = [
        max(1, ranks_by_label.get(label.upper(), default_rank))
        for label in labels
    ]
    return ranks, parsed


def _normalize_reward_model(info: object) -> Dict:
    if hasattr(info, "item"):
        try:
            info = info.item()
        except Exception:
            pass
    if isinstance(info, dict):
        return info
    return {}


def compute_score(
    solutions_batch,
    ground_truth_batch,
    valid_response_lengths,
    reward_tensor,
    max_response_length=None,
    tokenizer=None,
    extra_info_batch=None,
    reward_model_batch=None,
):
    """
    Compute validation rewards using repeated group rankings.
    """

    system_prompt = (
        "You are an expert mathematician and meticulous IMO grader. You will receive reference "
        "solutions and, when available, an explicit grading rubric. Evaluate each candidate with "
        "rigorous step-by-step verification, citing rubric criteria whenever provided. Base all "
        "comparisons solely on correctness, completeness, and adherence to the rubric."
    )

    processed_solutions = [extract_candidate_solution(sol) for sol in solutions_batch]

    if extra_info_batch is None:
        extra_info_batch = [{} for _ in processed_solutions]
    if reward_model_batch is None:
        reward_model_batch = [{} for _ in processed_solutions]

    question_ids = [
        _extract_question_id(extra_info_batch[idx], idx)
        for idx in range(len(processed_solutions))
    ]

    rubric_texts: List[str] = []
    for idx in range(len(processed_solutions)):
        reward_model_info = _normalize_reward_model(reward_model_batch[idx])
        rubric_text = reward_model_info.get("rubric") or reward_model_info.get("rubric_text")
        if not rubric_text:
            extra_info = _normalize_extra_info(extra_info_batch[idx])
            rubric_text = extra_info.get("rubric") or extra_info.get("rubric_text")
        rubric_texts.append(str(rubric_text).strip() if rubric_text else "")

    grouped_indices: defaultdict[str, List[int]] = defaultdict(list)
    for idx, question_id in enumerate(question_ids):
        grouped_indices[question_id].append(idx)

    rng = random.Random()
    points_schedule = list(DEFAULT_GROUP_POINTS)
    problem_statement = "Solve the given mathematical problem and provide a complete proof."

    group_prompts: List[str] = []
    group_contexts: List[dict] = []
    rewards_by_idx: defaultdict[int, List[float]] = defaultdict(list)

    for question_id, indices in grouped_indices.items():
        if not indices:
            continue

        if len(indices) == 1:
            for _ in range(DEFAULT_GROUP_ROUNDS):
                rewards_by_idx[indices[0]].append(points_schedule[0])
            continue

        reference_ground_truth = ground_truth_batch[indices[0]]

        for round_idx in range(DEFAULT_GROUP_ROUNDS):
            shuffled = list(indices)
            rng.shuffle(shuffled)

            for start in range(0, len(shuffled), DEFAULT_GROUP_SIZE):
                chunk = shuffled[start : start + DEFAULT_GROUP_SIZE]

                if len(chunk) == 1:
                    rewards_by_idx[chunk[0]].append(points_schedule[0])
                    continue

                labels = [chr(ord("A") + i) for i in range(len(chunk))]
                solutions_block = "\n\n".join(
                    f"### Solution {label} ###\n{processed_solutions[idx]}"
                    for label, idx in zip(labels, chunk)
                )
                points_lines = "\n".join(
                    f"Rank {pos + 1}: {_points_for_position(pos, points_schedule)}"
                    for pos in range(len(points_schedule))
                )
                json_example_lines = ",\n    ".join(
                    f'{{"label": "{label}", "rank": {pos + 1}, "reason": "one short sentence"}}'
                    for pos, label in enumerate(labels)
                )
                json_example = "{\n  \"placements\": [\n    " + json_example_lines + "\n  ]\n}"

                rubric_text = rubric_texts[chunk[0]]
                rubric_section = ""
                if rubric_text:
                    rubric_section = (
                        "### Rubric ###\n"
                        f"{rubric_text}\n"
                        "======================================================================\n"
                    )

                prompt = f"""Instructions
1) Read each candidate solution carefully (labels A-{labels[-1]}).
2) Perform a rigorous IMO-style verification for each solution independently.
3) Rank the solutions from best to worst based on correctness, completeness, and rigor.

======================================================================
### Question Metadata ###
Question ID: {question_id}
Round: {round_idx + 1} of {DEFAULT_GROUP_ROUNDS}
Group Size: {len(chunk)}
======================================================================
### Problem ###
{problem_statement}
======================================================================
### Ground Truth Solution ###
{reference_ground_truth}
======================================================================
{rubric_section if rubric_section else ""}
{solutions_block}
======================================================================
### Ranking Task Reminder ###
After completing the independent checks, provide a holistic ranking of the solutions.
Ties are allowed by assigning the same rank to multiple labels.
Use the following target point schedule when ordering:
{points_lines}

Return only a JSON object with this structure:
{json_example}
"""

                group_prompts.append(prompt)
                group_contexts.append(
                    {
                        "indices": chunk,
                        "labels": labels,
                    }
                )

    judge_responses: List[str] = []
    if group_prompts:
        judge_responses = run_prompts_sync_pool(
            client_service="openai",
            model="gpt-5-mini",
            system_prompt=system_prompt,
            prompts=group_prompts,
            max_tokens=32000,
            temperature=0.7,
            max_workers=16,
            timeout=60,
        )

    fallback_groups = 0

    for response, context in zip(judge_responses, group_contexts):
        ranks, parsed_ok = _parse_group_response(response, context["labels"])
        if not parsed_ok:
            fallback_groups += 1

        ordered = sorted(
            zip(context["indices"], ranks),
            key=lambda pair: (pair[1], pair[0]),
        )
        rank_position = 0
        processed = 0
        while processed < len(ordered):
            current_rank = ordered[processed][1]
            tie_group = [ordered[processed][0]]
            processed += 1
            while processed < len(ordered) and ordered[processed][1] == current_rank:
                tie_group.append(ordered[processed][0])
                processed += 1

            span_positions = list(range(rank_position, rank_position + len(tie_group)))
            total_points = sum(_points_for_position(pos, points_schedule) for pos in span_positions)
            shared_points = total_points / len(tie_group) if tie_group else 0.0

            for sample_idx in tie_group:
                rewards_by_idx[sample_idx].append(shared_points)

            rank_position += len(tie_group)

    if len(judge_responses) < len(group_contexts):
        fallback_groups += len(group_contexts) - len(judge_responses)
        for context in group_contexts[len(judge_responses) :]:
            ordered = sorted(
                zip(context["indices"], range(1, len(context["indices"]) + 1)),
                key=lambda pair: (pair[1], pair[0]),
            )
            rank_position = 0
            processed = 0
            while processed < len(ordered):
                current_rank = ordered[processed][1]
                tie_group = [ordered[processed][0]]
                processed += 1
                while processed < len(ordered) and ordered[processed][1] == current_rank:
                    tie_group.append(ordered[processed][0])
                    processed += 1
                span_positions = list(range(rank_position, rank_position + len(tie_group)))
                total_points = sum(_points_for_position(pos, points_schedule) for pos in span_positions)
                shared_points = total_points / len(tie_group) if tie_group else 0.0
                for sample_idx in tie_group:
                    rewards_by_idx[sample_idx].append(shared_points)
                rank_position += len(tie_group)

    total_reward_sum = 0.0
    num_samples = len(processed_solutions)

    for idx in range(num_samples):
        reward_history = rewards_by_idx.get(idx, [])
        if reward_history:
            avg_reward = float(sum(reward_history) / len(reward_history))
        else:
            avg_reward = 0.0
        reward_tensor[idx, valid_response_lengths[idx] - 1] = avg_reward
        total_reward_sum += avg_reward

    mean_reward = total_reward_sum / num_samples if num_samples else 0.0

    if group_prompts:
        print(
            f"[llm_judge_proofs_test] prompts={len(group_prompts)}, questions={len(grouped_indices)}, "
            f"fallback_groups={fallback_groups}, mean_reward={mean_reward:.3f}"
        )
    else:
        print(f"[llm_judge_proofs_test] all solo completions, mean_reward={mean_reward:.3f}")

    return reward_tensor
