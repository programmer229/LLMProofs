from verl.utils.reward_score.inference_utils import run_prompts
import asyncio
from typing import Optional, List

"""
Assumptions:
- All data in the dataset have the same data_source (being an LLM as a judge). This is necessary.
- The reward function name must have "llm_judge" in it for it to be considered an LLM as a judge.
- The test parquet file has a data_source which is not the LLM judge (for a proper evaluation). For example, for integration val parquets the data_source is "numeric_integration"
"""



def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Schedule on existing loop
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)
        
def judge(model: str,  # Either model name or path to model
          client_service: Optional[str],
          system_prompt: Optional[str],
          prompts: List[str],  # The prompt to use for judging
          max_tokens: int,
          temperature: float,
          local_model: bool = False,
          async_reward: bool = False) -> List[str]:
    
    
    # Perform judging using a locally run model
    if local_model:
        pass

    # Perform judging using an API model from inference_utils
    if not local_model:
        judge_responses = _run_sync(run_prompts(
                                client_service=client_service,
                                model=model,
                                system_prompt=system_prompt,
                                prompts=prompts,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            ))
    
    assert len(judge_responses) == len(prompts), "Judge responses not the same length as list of prompts."
    return judge_responses

if __name__ == "__main__":
    client_service = "openrouter"
    model = "gpt-5-nano"
    system_prompt = "Reasoning Level High"
    prompts = ["What's 8^8^8", "What is your name?"]

    judge_responses = judge(model=model, 
                            client_service=client_service, 
                            system_prompt=system_prompt, 
                            prompts=prompts, 
                            max_tokens=3000, 
                            temperature=0.5)
    
    print(judge_responses)