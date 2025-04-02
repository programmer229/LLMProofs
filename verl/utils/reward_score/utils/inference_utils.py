import os
import openai
import anthropic
import aiohttp
import json
from together import AsyncTogether
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
import asyncio
import requests
from huggingface_hub import InferenceClient
from together import Together

# Load environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
together_api_key = os.getenv('TOGETHER_API_KEY')
hf_api_token = os.getenv('HF_API_TOKEN')

# Choose an inference client
def get_inference_client(client_service: str):
    """
    Returns a client object for the specified client service.
    """

    if client_service == "together":
        client = Together()
        return client
    if client_service == "openai":
        client = AsyncOpenAI(api_key=openai_api_key)
        return client
    if client_service == "anthropic":
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        return client
    else:
        print("Please specify a valid client service. The available options are: 'deepinfra', 'openai', 'anthropic'.")
        exit()

async def generate_text(client_service: str, model: str, system_prompt : str, prompt: str, png_base64_image: str = None, max_tokens: int = 8000, temperature: float = 0) -> str:
    """
    Asynchronously generate text using various AI models.
    
    :param model: The name of the model to use (e.g., "gpt-3.5-turbo", "claude-2", "meta-llama/Llama-2-70b-chat-hf")
    :param prompt: The input prompt for text generation
    :param max_tokens: Maximum number of tokens to generate
    :param temperature: Controls randomness in generation (0.0 to 1.0)
    :return: Generated text as a string
    """
    
    # HuggingFace Inference API
    if client_service == "huggingface":
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {hf_api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
        async def query_huggingface():
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                            return result[0]['generated_text'].strip()
                        else:
                            return str(result)
                    else:
                        raise Exception(f"Error with Hugging Face Inference API for model {model}: {await response.text()}")
        
        return await query_huggingface()
    
    # Returns a client object for the specified client service
    client = get_inference_client(client_service=client_service)

    # Perform the inference on the returned client object:

    # OpenAI or DeepInfra
    if client_service == "openai":
     
        # If the client object has been setup
        if png_base64_image is None:
            response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
            )
        
        else:
            response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt}, 
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_base64_image}"}}
                            ]
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
            )
        
        content = response.choices[0].message.content.strip()
        return content
       

    if client_service == "together":
        # If the client object has been setup
        if png_base64_image is None:
            print(f"Generating text with Together model {model}")
            response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            print(f"Received response")
            
            content = response.choices[0].message.content
            return content
        else:
            print(f"Generating text with image with Together model {model}")
            response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt}, 
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_base64_image}"}}
                            ]
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            print(f"Received response")
            
            content = response.choices[0].message.content
            return content

    # Anthropic
    if client_service == "anthropic":
        async def run_anthropic():
            if model.startswith("claude-3"):
                response = client.messages.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.content[0].text.strip()
            else:
                response = client.completions.create(
                    model=model,
                    prompt=f"Human: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature
                )
                return response.completion.strip()
        
        return await run_anthropic()

async def run_prompts(client_service, model, system_prompt, prompts, max_tokens, temperature, png_base64_images = None):
    tasks = [
        generate_text(
            client_service=client_service,
            model=model, 
            system_prompt=system_prompt,
            prompt=prompt,
            png_base64_image=png_base64_image,
            max_tokens=max_tokens,
            temperature=temperature
        ) for (prompt, png_base64_image) in zip(prompts, png_base64_images)
    ]
    return_texts = await asyncio.gather(*tasks)
    return return_texts

if __name__ == "__main__":
    client_service = "openai"
    model = "gpt-4o-2024-11-20"
    system_prompt = "You are a helpful assistant."
    base_prompt = "Please write me a short story"
    base_prompt_8k = base_prompt*1000
    prompts = [base_prompt_8k]*200
    
    return_texts = asyncio.run(run_prompts(client_service=client_service, model=model, system_prompt=system_prompt, prompts=prompts, max_tokens=4000, temperature=0.5))
    print(return_texts)