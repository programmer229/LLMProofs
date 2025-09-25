from datasets import load_dataset, Dataset
import json
import hashlib
import random
from tqdm import tqdm
import os

from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict

def format_grid(grid: List[List[int]]) -> str:
    # Using string formatting to remove decimal places
    return '\n'.join([str([int(float(num)) for num in row]) for row in grid])

def format_examples(examples: List[Dict]) -> str:
    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.extend([
            f"Training example {i}:",
            "Input grid:",
            format_grid(ex['input']),
            "Output grid:",
            format_grid(ex['output']),
            ""
        ])
    return '\n'.join(formatted)

def create_prompt(formatted_train_examples: str, formatted_test_examples: str) -> str:
    return f"""Given these training examples:
{formatted_train_examples}
Please solve this puzzle and provide both the transformation rule and the predicted output grid.
Format your response with the rule in <rule> tags and the predicted grid in <answer> tags.
Test input:
{formatted_test_examples}"""

def process_item(item):
    # Get all examples except the test example
    available_examples = item["examples"][:-1]
    test_example = item["examples"][-1]
    
    # Randomly select 3-5 examples for training
    num_examples = random.randint(3, 4)
    if len(available_examples) < num_examples:
        return None
    
    training_examples = random.sample(available_examples, num_examples)
    
    # Format the examples
    train_examples = [{"input": ex[0], "output": ex[1]} for ex in training_examples]
    test_input = test_example[0]
    test_output = test_example[1]
    
    # Create prompts
    formatted_train = format_examples(train_examples)
    formatted_test = format_grid(test_input)
    prompt = create_prompt(formatted_train, formatted_test)
    
    # Skip if prompt is too long (> 4096 chars)
    if len(prompt) > 8096:
        return None
    
    # Create single training example with all context
    train_data = [{
        'data_source': 'grid_transform',
        'prompt': [{'role': 'user', 'content': prompt}],
        'ability': 'grid_transform',
        'reward_model': {
            'style': 'rule',
            'ground_truth': test_output
        },
        'extra_info': {
            'split': 'train',
            'index': 0
        }
    }]
    
    # Create test example
    test_data = [{
        'data_source': 'grid_transform',
        'prompt': [{'role': 'user', 'content': prompt}],
        'ability': 'grid_transform',
        'reward_model': {
            'style': 'rule',
            'ground_truth': test_output
        },
        'extra_info': {
            'split': 'test',
            'index': 0
        }
    }]
    
    return train_data, test_data

def save_to_parquet(data: list, output_path: str):
    print(f"Saving {len(data)} examples to {output_path}")
    chunk_size = 10000
    num_chunks = (len(data) + chunk_size - 1) // chunk_size
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Save all chunks as a single dataset
    dataset = Dataset.from_list(data)
    dataset.to_parquet(output_path)

def main(output_dir='output', max_test_samples=1024, max_size=500000):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading BARC dataset...")
    dataset = load_dataset(
        "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
        features_cache_dir=None
    )
    
    if not dataset or 'train' not in dataset:
        print("Failed to load dataset")
        return
    
    # Use the actual dataset size, but cap it at max_size
    num_examples = min(len(dataset['train']), max_size)
    print(f"Processing {num_examples} examples...")
    
    # Initialize lists for train and test data
    all_train_data = []
    all_test_data = []
    
    # Process one example at a time
    for i, item in enumerate(tqdm(dataset['train'].select(range(num_examples)))):
        try:
            result = process_item(item)
            if result is not None:
                train_data, test_data = result
                all_train_data.extend(train_data)
                if len(all_test_data) < max_test_samples:
                    all_test_data.extend(test_data[:max(0, max_test_samples - len(all_test_data))])
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
            continue
    
    # Save to parquet files
    train_path = os.path.join(output_dir, 'train_arc.parquet')
    test_path = os.path.join(output_dir, 'test_arc.parquet')
    
    save_to_parquet(all_train_data, train_path)
    save_to_parquet(all_test_data, test_path)
    
    print(f"Saved {len(all_train_data)} training examples and {len(all_test_data)} test examples")
    print(f"Files saved at:\n{train_path}\n{test_path}")

if __name__ == '__main__':
    main(output_dir='/home/ubuntu/o1-replication/CustomTinyZero/data/arc')
