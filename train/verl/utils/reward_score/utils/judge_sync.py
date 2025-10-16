# judge_sync.py
import os, json, concurrent.futures, requests
from typing import List, Optional, Tuple
from openai import OpenAI
import anthropic
from together import Together

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def _mk_client(client_service: str):
    if client_service == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        return ("openai", OpenAI(api_key=OPENAI_API_KEY))
    if client_service == "openrouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        # OpenRouter is OpenAI-compatible
        return ("openrouter", OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"))
    if client_service == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return ("anthropic", anthropic.Anthropic(api_key=ANTHROPIC_API_KEY))
    if client_service == "together":
        if not TOGETHER_API_KEY:
            raise RuntimeError("TOGETHER_API_KEY not set")
        return ("together", Together(api_key=TOGETHER_API_KEY))
    if client_service == "huggingface":
        if not HF_API_TOKEN:
            raise RuntimeError("HF_API_TOKEN not set")
        return ("huggingface", None)
    if client_service.startswith("simplejudge"):
        host = client_service.split("-", 1)[1]
        return ("simplejudge", OpenAI(api_key="EMPTY", base_url=f"http://{host}:8000"))
    raise RuntimeError(f"Unknown client_service: {client_service}")

def _call_one(kind, client, model, system_prompt, prompt, max_tokens, temperature, png_b64=None, timeout=30) -> str:
    try:
        if kind in {"openai","openrouter","simplejudge"}:
            msgs = [{"role":"system","content":system_prompt}]
            if png_b64 is None:
                msgs.append({"role":"user","content":prompt})
            else:
                msgs.append({"role":"user","content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{png_b64}"}}
                ]})
            if kind in {"openai"}:
                r = client.chat.completions.create(
                    model=model, messages=msgs
                )
            else:
                r = client.chat.completions.create(
                    model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature, timeout=timeout
                )
            return r.choices[0].message.content.strip()

        if kind == "anthropic":
            if model.startswith("claude-3"):
                r = client.messages.create(
                    model=model, system=system_prompt, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role":"user","content":prompt}],
                )
                for b in r.content:
                    if hasattr(b, "text"):
                        return b.text.strip()
                return json.dumps(r.to_dict())
            else:
                r = client.completions.create(
                    model=model, prompt=f"Human: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=max_tokens, temperature=temperature, timeout=timeout
                )
                return r.completion.strip()

        if kind == "together":
            r = client.chat.completions.create(
                model=model, messages=[{"role":"system","content":system_prompt},{"role":"user","content":prompt}],
                max_tokens=max_tokens, temperature=temperature, timeout=timeout
            )
            return r.choices[0].message.content.strip()

        if kind == "huggingface":
            url = f"https://api-inference.huggingface.co/models/{model}"
            hdr = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}}
            resp = requests.post(url, headers=hdr, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            return json.dumps(data)

        raise RuntimeError(f"Unsupported kind {kind}")

    except Exception as e:
        # Keep Ray-friendly; your caller can handle "ERROR:" prefixes.
        return f"ERROR: {type(e).__name__}: {e}"

def run_prompts_sync_pool(
    client_service: str,
    model: str,
    system_prompt: str,
    prompts: List[str],
    *,
    max_tokens: int = 4000,
    temperature: float = 0.0,
    png_base64_images: Optional[List[Optional[str]]] = None,
    max_workers: int = 8,
    timeout: int = 30,
) -> List[str]:
    kind, client = _mk_client(client_service)
    if png_base64_images is None:
        png_base64_images = [None] * len(prompts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_call_one, kind, client, model, system_prompt, p, max_tokens, temperature, img, timeout)
            for p, img in zip(prompts, png_base64_images)
        ]
        out = [f.result() for f in futs]

    # normalize
    return [o if isinstance(o, str) else str(o) for o in out]

if __name__ == "__main__":
    # Simple sanity test
    import pprint

    system_prompt = "You are a strict grader. reply with <JUDGE_SCORE>n</JUDGE_SCORE>."
    prompts = ["Hello how are you"]

    try:
        responses = run_prompts_sync_pool(
            client_service="openrouter",
            model="deepseek/deepseek-r1-distill-qwen-14b",
            system_prompt=system_prompt,
            prompts=prompts,
            max_tokens=512,
            temperature=0.0,
        )
        print("=== Test Responses ===")
        pprint.pprint(responses)
    except Exception as e:
        print(f"❌ Test failed: {type(e).__name__}: {e}")
        print("Make sure OPENROUTER_API_KEY is set in your environment.")

