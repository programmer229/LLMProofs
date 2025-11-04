"""
Evaluation-time creative-writing reward that scores each story independently via an LLM judge.

This reuses the rubric-generation logic from the training reward but skips GRPO ranking,
emitting a normalized scalar reward per sample for validation/test metrics.
"""

from __future__ import annotations

import random
from typing import List

from .llm_judge_creative import (
    MAX_REFERENCE_CHARS,
    _ensure_rubric,
    _extract_question_id,
    _normalize_extra_info,
    _normalize_reward_model,
    _truncate,
    extract_candidate_story,
)
from verl.utils.reward_score.utils.judge_sync import run_prompts_sync_pool
from verl.utils.reward_score.utils.llm_judge_utils import extract_judge_score


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
    Score each generated story with the creative-writing judge.
    """

    processed_stories = [extract_candidate_story(sol) for sol in solutions_batch]

    if extra_info_batch is None:
        extra_info_batch = [{} for _ in processed_stories]
    if reward_model_batch is None:
        reward_model_batch = [{} for _ in processed_stories]

    assignments: List[str] = []
    reference_stories: List[str] = []
    question_ids: List[str] = []

    for idx in range(len(processed_stories)):
        extra = _normalize_extra_info(extra_info_batch[idx])
        reward_info = _normalize_reward_model(reward_model_batch[idx])

        question_ids.append(_extract_question_id(extra, idx))

        assignment = reward_info.get("assignment") or extra.get("assignment") or extra.get("title")
        if not assignment:
            assignment = "Write a polished, original creative piece that fulfills the given brief."
        assignments.append(assignment)

        reference_story = reward_info.get("reference_story") or reward_info.get("ground_truth") or ground_truth_batch[idx]
        reference_stories.append(str(reference_story))

    system_prompt = (
        "You are an MFA-level creative writing instructor asked to holistically grade student submissions. "
        "Apply the rubric faithfully, reward originality and craft, and output both feedback and a final score."
    )

    prompts: List[str] = []
    for idx, (assignment, reference_story, story) in enumerate(zip(assignments, reference_stories, processed_stories)):
        question_id = question_ids[idx]
        rubric_text = _ensure_rubric(question_id=question_id, assignment=assignment, exemplar_story=reference_story)

        reference_excerpt = _truncate(reference_story, MAX_REFERENCE_CHARS)

        prompt = f"""Assignment:
{assignment}

Exemplar human story (for calibration, not to copy):
{reference_excerpt}

Rubric:
{rubric_text}

Student story to evaluate:
{story}

Instructions:
1. Assess how well the student story fulfills each rubric criterion.
2. Provide concise feedback that references specific strengths or issues.
3. Output the final mark as an integer from 0 to 100 wrapped in <JUDGE_SCORE> tags.
"""
        prompts.append(prompt)

    judge_responses = run_prompts_sync_pool(
        client_service="openai",
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        prompts=prompts,
        max_tokens=2048,
        temperature=0.4,
        timeout=60,
        max_workers=32,
    )

    sample_count = len(processed_stories)
    preview = min(3, sample_count)
    if sample_count:
        for idx in random.sample(range(sample_count), preview):
            print(f"[llm_judge_creative_test] story {idx} sample:\n{processed_stories[idx][:500]}\n---")
            print(f"[llm_judge_creative_test] judge response {idx}:\n{judge_responses[idx]}\n{'-' * 40}")

    total_normalized = 0.0
    for idx, response in enumerate(judge_responses):
        raw_score = extract_judge_score(response)
        raw_score = max(0, min(raw_score, 100))
        normalized = raw_score / 100.0
        reward_tensor[idx, valid_response_lengths[idx] - 1] = normalized
        total_normalized += normalized

    mean_reward = total_normalized / sample_count if sample_count else 0.0
    print(f"[llm_judge_creative_test] mean normalized score: {mean_reward:.3f}")

    return reward_tensor
