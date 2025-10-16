import os
import asyncio
import json
from typing import Optional, List, Tuple

import aiohttp

# OpenAI-compatible SDK
from openai import AsyncOpenAI

# Anthropic SDK is sync; we'll call it via asyncio.to_thread
import anthropic

# Together has an async client
from together import AsyncTogether


# ---------- Config / env ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")


# ---------- Client factory & teardown ----------

async def make_client(client_service: str):
    """
    Create one client instance per run/worker and reuse it.
    """
    if client_service == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return ("openai", AsyncOpenAI(api_key=OPENAI_API_KEY))

    if client_service == "openrouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        # OpenRouter is OpenAI-compatible
        return ("openrouter", AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"))

    if client_service == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        # anthropic.Anthropic is sync
        return ("anthropic", anthropic.Anthropic(api_key=ANTHROPIC_API_KEY))

    if client_service == "together":
        if not TOGETHER_API_KEY:
            raise RuntimeError("TOGETHER_API_KEY is not set")
        return ("together", AsyncTogether(api_key=TOGETHER_API_KEY))

    if client_service == "huggingface":
        if not HF_API_TOKEN:
            raise RuntimeError("HF_API_TOKEN is not set")
        # No persistent client needed; we use aiohttp per-call
        return ("huggingface", None)

    if client_service.startswith("simplejudge"):
        # OpenAI-compatible proxy hosted by you
        parts = client_service.split("-", 1)
        if len(parts) != 2:
            raise RuntimeError("Expected format simplejudge-<ip_or_host>")
        base = parts[1]
        return ("simplejudge", AsyncOpenAI(api_key="EMPTY", base_url=f"http://{base}:8000", max_retries=0))

    raise RuntimeError(
        "Unknown client_service. Use one of: openai, openrouter, anthropic, together, huggingface, simplejudge-<host>"
    )


async def close_client(kind: str, client):
    """
    Close async clients once at shutdown. Anthropic is sync and does not need closing.
    """
    try:
        if kind in {"openai", "openrouter", "simplejudge"}:
            # AsyncOpenAI has .close() (async)
            await client.close()
        elif kind == "together":
            # AsyncTogether exposes .close() (async)
            await client.close()
        # huggingface: no client to close
    except Exception:
        # Swallow close exceptions; shutdown should be best-effort
        pass


# ---------- Core generation ----------

async def _generate_openai_compatible(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    prompt: str,
    *,
    png_base64_image: Optional[str],
    max_tokens: int,
    temperature: float,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if png_base64_image is None:
        messages.append({"role": "user", "content": prompt})
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_base64_image}"}}
            ]
        })

    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


async def _generate_together(
    client: AsyncTogether,
    model: str,
    system_prompt: str,
    prompt: str,
    *,
    png_base64_image: Optional[str],
    max_tokens: int,
    temperature: float,
) -> str:
    # Together async API (OpenAI-like)
    messages = [{"role": "system", "content": system_prompt}]
    if png_base64_image is None:
        messages.append({"role": "user", "content": prompt})
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_base64_image}"}}
            ]
        })

    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


async def _generate_anthropic(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    # Anthropic SDK is sync; wrap in thread to avoid blocking
    def _call():
        # Use Messages API for Claude 3 family
        if model.startswith("claude-3"):
            resp = client.messages.create(
                model=model,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            # content is a list of blocks; take first text block
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text.strip()
            # Fallback
            return json.dumps(resp.to_dict())
        else:
            # Legacy completions
            resp = client.completions.create(
                model=model,
                prompt=f"Human: {prompt}\n\nAssistant:",
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            return resp.completion.strip()
    return await asyncio.to_thread(_call)


async def _generate_huggingface(
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    # Stateless per-call usage; no persistent client
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload, timeout=120) as r:
            if r.status != 200:
                text = await r.text()
                raise RuntimeError(f"HuggingFace HTTP {r.status}: {text}")
            result = await r.json()
    if isinstance(result, list) and result and "generated_text" in result[0]:
        return result[0]["generated_text"].strip()
    # Models sometimes return dicts with 'error' or token streams
    return json.dumps(result)


async def generate_text(
    client_kind: str,
    client,
    model: str,
    system_prompt: str,
    prompt: str,
    *,
    png_base64_image: Optional[str] = None,
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> str:
    """
    Provider-agnostic generation with Ray-safe error conversion.
    """
    try:
        if client_kind in {"openai", "openrouter", "simplejudge"}:
            return await _generate_openai_compatible(
                client, model, system_prompt, prompt,
                png_base64_image=png_base64_image,
                max_tokens=max_tokens, temperature=temperature
            )
        if client_kind == "together":
            return await _generate_together(
                client, model, system_prompt, prompt,
                png_base64_image=png_base64_image,
                max_tokens=max_tokens, temperature=temperature
            )
        if client_kind == "anthropic":
            return await _generate_anthropic(
                client, model, system_prompt, prompt,
                max_tokens=max_tokens, temperature=temperature
            )
        if client_kind == "huggingface":
            return await _generate_huggingface(
                model, prompt, max_tokens=max_tokens, temperature=temperature
            )
        raise RuntimeError(f"Unsupported client kind: {client_kind}")

    except Exception as e:
        # Make exceptions pickle-friendly for Ray (avoid complex .response/.body attrs)
        raise RuntimeError(f"{type(e).__name__}: {str(e)}") from None


# ---------- Batch runner ----------

async def run_prompts(
    client_service: str,
    model: str,
    system_prompt: str,
    prompts: List[str],
    *,
    max_tokens: int = 8000,
    temperature: float = 0.0,
    png_base64_images: Optional[List[Optional[str]]] = None,
) -> List[str]:
    client_kind, client = await make_client(client_service)
    try:
        if png_base64_images is None:
            tasks = [
                generate_text(
                    client_kind, client, model, system_prompt, p,
                    max_tokens=max_tokens, temperature=temperature
                ) for p in prompts
            ]
        else:
            if len(png_base64_images) != len(prompts):
                raise ValueError("png_base64_images length must match prompts length")
            tasks = [
                generate_text(
                    client_kind, client, model, system_prompt, p,
                    png_base64_image=img,
                    max_tokens=max_tokens, temperature=temperature
                ) for p, img in zip(prompts, png_base64_images)
            ]

        # If you want errors to bubble immediately, keep return_exceptions=False
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    finally:
        await close_client(client_kind, client)


# ---------- CLI / demo ----------

async def _main():
    client_service = "simplejudge-150.136.45.212"  # or "openai", "openrouter", "anthropic", "together", "huggingface"
    model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    system_prompt = "You are a helpful assistant."
    base_prompt = "Please write me a short story."
    prompts = [base_prompt] * 2

    texts = await run_prompts(
        client_service=client_service,
        model=model,
        system_prompt=system_prompt,
        prompts=prompts,
        max_tokens=200,
        temperature=0.5,
    )
    for i, t in enumerate(texts, 1):
        print(f"\n--- Output {i} ---\n{t}\n")


if __name__ == "__main__":
    asyncio.run(_main())
