from verl.utils.reward_score.utils.inference_utils import run_prompts
import asyncio
import threading
from typing import Optional, List

"""
Assumptions:
- All data in the dataset have the same data_source (being an LLM as a judge).
- The reward function name must have "llm_judge" in it for it to be considered an LLM as a judge.
- The test parquet file has a data_source which is not the LLM judge (for proper evaluation).
"""

def _run_async_safely(coro):
    """
    Always run the coroutine on a dedicated, standard-asyncio event loop
    in a background thread. This avoids interfering with an already-running
    loop (e.g., Ray/Jupyter) and sidesteps uvloop shutdown quirks.
    """
    # Fast path: no running loop in this thread
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Slow path: we're already inside a running loop -> spin a fresh loop in a thread
    result_box = {}
    err_box = {}

    def _runner():
        # Force a stdlib loop (not uvloop)
        try:
            if hasattr(asyncio, "SelectorEventLoop"):
                loop = asyncio.SelectorEventLoop()
            else:
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Optional: add a global timeout so hung providers don't block teardown
            fut = asyncio.wait_for(coro, timeout=None)  # set e.g. 120 if you want
            result_box["v"] = loop.run_until_complete(fut)
        except Exception as e:
            err_box["e"] = e
        finally:
            try:
                loop.run_until_complete(asyncio.sleep(0))
            except Exception:
                pass
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "e" in err_box:
        # Make it Ray-safe by collapsing complex exception fields
        e = err_box["e"]
        raise RuntimeError(f"{type(e).__name__}: {e}")
    return result_box.get("v")

def judge(model: str,  # Either model name or path to model
          client_service: Optional[str],
          system_prompt: Optional[str],
          prompts: List[str],  # The prompts to use for judging
          max_tokens: int,
          temperature: float,
          local_model: bool = False,
          png_base64_images: Optional[List[str]] = None,
          async_reward: bool = False) -> List[str]:

    # Local model path (stub)
    if local_model:
        return [""] * len(prompts)

    # API path via run_prompts
    try:
        coro = run_prompts(
            client_service=client_service,
            model=model,
            system_prompt=system_prompt,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            png_base64_images=png_base64_images,
        )
        # Run in a dedicated thread+loop (never poke the current loop)
        judge_responses = _run_async_safely(coro)
    except Exception as e:
        # Never crash the caller; keep errors Ray-pickleable
        print(f"[judge] run_prompts failed: {type(e).__name__}: {e}")
        judge_responses = [""] * len(prompts)

    # Normalize to list[str] and match length exactly
    if not isinstance(judge_responses, list):
        judge_responses = [str(judge_responses)]

    if len(judge_responses) != len(prompts):
        judge_responses = [
            judge_responses[i] if i < len(judge_responses) else ""
            for i in range(len(prompts))
        ]

    return [r if isinstance(r, str) else str(r) for r in judge_responses]


if __name__ == "__main__":
    client_service = "openai"
    model = "gpt-3.5-turbo"
    system_prompt = "You are a helpful assistant."
    prompts = ["Hello, how are you?", "What is your name?"]

    judge_responses = judge(
        model=model,
        client_service=client_service,
        system_prompt=system_prompt,
        prompts=prompts,
        max_tokens=30,
        temperature=0.5,
    )
    print(judge_responses)
