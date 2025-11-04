"""
Creative-writing reward function backed by an LLM judge.

Compared to the math proof variant, this version:
  * Uses human-written exemplar stories as the reference "ground truth".
  * Generates a grading rubric dynamically with an LLM the first time each prompt appears.
  * Performs group-based ranking (GRPO-style) to obtain relative rewards.
"""

from __future__ import annotations

from collections import defaultdict
import random
import re
from typing import Dict, List, Sequence, Tuple

from verl.utils.reward_score.utils.judge_sync import run_prompts_sync_pool


DEFAULT_GROUP_POINTS: List[float] = [1.0, 0.6, 0.2]
DEFAULT_GROUP_SIZE: int = 3
DEFAULT_GROUP_ROUNDS: int = 5
MAX_REFERENCE_CHARS: int = 2500
MAX_STORY_PREVIEW_CHARS: int = 2500

_RUBRIC_CACHE: Dict[str, str] = {}


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
    for key in ("question_id", "prompt_id", "id", "assignment_id"):
        value = info_dict.get(key)
        if value not in (None, ""):
            return str(value)
    split = info_dict.get("split")
    index = info_dict.get("index")
    if split not in (None, "") and index not in (None, ""):
        return f"{split}::{index}"
    return f"creative_{fallback_idx}"


def _normalize_reward_model(info: object) -> Dict:
    if hasattr(info, "item"):
        try:
            info = info.item()
        except Exception:
            pass
    if isinstance(info, dict):
        return info
    return {}


def extract_candidate_story(text: str) -> str:
    """
    Extract the candidate's story.
    Preference order:
      1. Content after a </think> tag.
      2. Content inside <writing>...</writing>.
      3. The raw text.
    """
    if not isinstance(text, str):
        return ""

    think_match = re.search(r"</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        return think_match.group(1).strip()

    writing_match = re.search(r"<writing>(.*?)</writing>", text, re.DOTALL | re.IGNORECASE)
    if writing_match:
        return writing_match.group(1).strip()

    return text.strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[TRUNCATED]"


def _build_rubric_prompt(assignment: str, exemplar_story: str) -> str:
    exemplar_excerpt = _truncate(exemplar_story, MAX_REFERENCE_CHARS)
    return (
        "You are a creative-writing professor asked to produce a concise scoring rubric.\n"
        "Create clear evaluation criteria (3-5 bullet points) tailored to the assignment below. "
        "Cap each criterion with guidance for awarding high, medium, and low scores on a 0-10 scale. "
        "Avoid generic advice; tie the rubric to the assignment's goals.\n\n"
        f"Assignment:\n{assignment.strip()}\n\n"
        "Reference exemplar (do not copy; use only to understand expectations):\n"
        f"{exemplar_excerpt}\n\n"
        "Return the rubric as bullet points (e.g., '- Criterion: guidance...')."
    )


def _ensure_rubric(question_id: str, assignment: str, exemplar_story: str) -> str:
    """
    Ensure a rubric is cached for the question_id. Generates one if necessary.
    """
    if question_id in _RUBRIC_CACHE:
        return _RUBRIC_CACHE[question_id]

    prompt = _build_rubric_prompt(assignment=assignment, exemplar_story=exemplar_story)
    responses = run_prompts_sync_pool(
        client_service="openai",
        model="gpt-4.1-mini",
        system_prompt="You design grading rubrics for creative writing MFA workshops.",
        prompts=[prompt],
        max_tokens=1024,
        temperature=0.4,
        timeout=60,
        max_workers=1,
    )
    rubric = responses[0].strip() if responses else ""
    if not rubric:
        rubric = (
            "- Concept & originality: Reward distinctive ideas that fulfill the brief. Penalise clichés or drift.\n"
            "- Narrative craft: Assess structure, pacing, and clarity of storytelling.\n"
            "- Voice & language: Evaluate prose quality, imagery, and emotional resonance.\n"
            "- Alignment to assignment: Ensure the response delivers the requested tone, genre, and focus."
        )
    _RUBRIC_CACHE[question_id] = rubric
    return rubric


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
            import json

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
        except Exception:
            parsed = False

    default_rank = len(labels)
    ranks = [
        max(1, ranks_by_label.get(label.upper(), default_rank))
        for label in labels
    ]
    return ranks, parsed


def compute_score(
    solutions_batch,
    ground_truth_batch,
    valid_response_lengths,
    reward_tensor,
    max_response_length=None,
    tokenizer=None,
    extra_info_batch=None,
    reward_model_batch=None,
    reward_conversion_mode: str = "group_points",
):
    """
    Compute rewards by repeatedly ranking groups of creative-writing responses.
    """

    processed_stories = [extract_candidate_story(sol) for sol in solutions_batch]

    if extra_info_batch is None:
        extra_info_batch = [{} for _ in processed_stories]
    if reward_model_batch is None:
        reward_model_batch = [{} for _ in processed_stories]

    question_ids = []
    assignments = []
    reference_stories = []

    for idx in range(len(processed_stories)):
        extra = _normalize_extra_info(extra_info_batch[idx])
        reward_model_info = _normalize_reward_model(reward_model_batch[idx])

        question_ids.append(_extract_question_id(extra, idx))

        assignment = reward_model_info.get("assignment") or extra.get("assignment") or extra.get("title")
        if not assignment:
            assignment = "Write a polished, original creative piece that fulfills the given brief."
        assignments.append(assignment)

        reference_story = reward_model_info.get("reference_story") or reward_model_info.get("ground_truth") or ground_truth_batch[idx]
        reference_stories.append(str(reference_story))

    grouped_indices: defaultdict[str, List[int]] = defaultdict(list)
    for idx, question_id in enumerate(question_ids):
        grouped_indices[question_id].append(idx)

    rng = random.Random()
    points_schedule = list(DEFAULT_GROUP_POINTS)

    group_prompts: List[str] = []
    group_contexts: List[dict] = []
    rewards_by_idx: defaultdict[int, List[float]] = defaultdict(list)

    for question_id, indices in grouped_indices.items():
        if not indices:
            continue

        assignment = assignments[indices[0]]
        reference_story = reference_stories[indices[0]]
        rubric_text = _ensure_rubric(question_id=question_id, assignment=assignment, exemplar_story=reference_story)

        if len(indices) == 1:
            for _ in range(DEFAULT_GROUP_ROUNDS):
                rewards_by_idx[indices[0]].append(points_schedule[0])
            continue

        reference_excerpt = _truncate(reference_story, MAX_REFERENCE_CHARS)

        for round_idx in range(DEFAULT_GROUP_ROUNDS):
            shuffled = list(indices)
            rng.shuffle(shuffled)

            for start in range(0, len(shuffled), DEFAULT_GROUP_SIZE):
                chunk = shuffled[start : start + DEFAULT_GROUP_SIZE]

                if len(chunk) == 1:
                    rewards_by_idx[chunk[0]].append(points_schedule[0])
                    continue

                labels = [chr(ord("A") + i) for i in range(len(chunk))]
                stories_block = "\n\n".join(
                    f"### Story {label} ###\n{_truncate(processed_stories[idx], MAX_STORY_PREVIEW_CHARS)}"
                    for label, idx in zip(labels, chunk)
                )
                points_lines = "\n".join(
                    f"Rank {pos + 1}: {_points_for_position(pos, points_schedule)}"
                    for pos in range(len(points_schedule))
                )
                json_example_lines = ",\n    ".join(
                    f'{{"label": "{label}", "rank": {pos + 1}, "reason": "short justification"}}'
                    for pos, label in enumerate(labels)
                )
                json_example = "{\n  \"placements\": [\n    " + json_example_lines + "\n  ]\n}"

                prompt = f"""You are adjudicating responses for a graduate creative-writing workshop.

Assignment:
{assignment}

Exemplar human story (for calibration, not to copy):
{reference_excerpt}

Rubric:
{rubric_text}

Candidate stories to evaluate:
{stories_block}

Guidelines:
1. Judge originality, emotional depth, structure, voice, and alignment with the assignment.
2. Do not reward plagiarism or close paraphrases of the exemplar.
3. Ties are allowed by assigning the same rank.
4. Use this reward schedule when ordering:
{points_lines}

Return ONLY a JSON object following this schema:
{json_example}
"""
                group_prompts.append(prompt)
                group_contexts.append({"indices": chunk, "labels": labels})

    judge_responses: List[str] = []
    if group_prompts:
        judge_responses = run_prompts_sync_pool(
            client_service="openai",
            model="gpt-4.1-nano",
            system_prompt="You are an expert creative-writing judge delivering meticulous rankings.",
            prompts=group_prompts,
            max_tokens=2048,
            temperature=0.5,
            timeout=60,
            max_workers=64,
        )
        for debug_idx, response in enumerate(judge_responses[:3]):
            print(f"[llm_judge_creative] sample judge response {debug_idx}:\n{response}\n{'-' * 40}")

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
    num_samples = len(processed_stories)
    final_rewards = [0.0 for _ in range(num_samples)]

    for idx in range(num_samples):
        reward_history = rewards_by_idx.get(idx, [])
        if reward_history:
            avg_reward = float(sum(reward_history) / len(reward_history))
        else:
            avg_reward = 0.0
        final_rewards[idx] = avg_reward

    if reward_conversion_mode == "harmonic_rank":
        for question_id, indices in grouped_indices.items():
            if not indices:
                continue
            ordered = sorted(
                indices,
                key=lambda sample_idx: (-final_rewards[sample_idx], sample_idx),
            )
            for rank_position, sample_idx in enumerate(ordered):
                final_rewards[sample_idx] = 1.0 / float(rank_position + 1)
    elif reward_conversion_mode == "squared":
        for idx in range(num_samples):
            value = final_rewards[idx]
            final_rewards[idx] = value * value

    for idx in range(num_samples):
        reward_tensor[idx, valid_response_lengths[idx] - 1] = final_rewards[idx]
        total_reward_sum += final_rewards[idx]

    mean_reward = total_reward_sum / num_samples if num_samples else 0.0

    if group_prompts:
        print(
            f"Creative reward: prompts={len(group_prompts)}, questions={len(grouped_indices)}, "
            f"fallback_groups={fallback_groups}, conversion={reward_conversion_mode}, mean_reward={mean_reward:.3f}"
        )
    else:
        print(f"Creative reward: all solo completions, conversion={reward_conversion_mode}, mean_reward={mean_reward:.3f}")

    return reward_tensor
