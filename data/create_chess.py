import os
import datasets

if __name__ == '__main__':
    data_source = 'EleutherAI/lichess-puzzles'
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)["train"]
    
    # Create train-test split
    split_datasets = dataset.train_test_split(test_size=0.001)
    train_dataset = split_datasets['train']
    test_dataset = split_datasets['test']
    
    # Define chess-specific instruction
    instruction_following = """Analyze this chess puzzle carefully. Provide ONLY the next move in algebraic notation within answer tags, like:
        <answer>e4e5</answer>"""
    
    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract puzzle context and solution
            puzzle_context = example.pop('ctx')
            puzzle_solution = example.pop('target')
            
            # Combine context with instruction
            puzzle_with_instruction = puzzle_context + ' ' + instruction_following
            
            data = {
                "data_source": "chess_puzzles",
                "prompt": [{
                    "role": "user",
                    "content": puzzle_with_instruction
                }],
                "ability": "chess",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": puzzle_solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'puzzle_id': example.get('id', f"{split}_{idx}")  # Include original puzzle ID if available
                }
            }
            return data
        return process_fn

    # Apply the mapping function to both datasets
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Save to parquet files
    local_dir = '/home/ubuntu/o1-replication/CustomTinyZero/data/chess'
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    test_dataset = test_dataset.select(range(min(len(test_dataset), 1024)))
    
    train_dataset.to_parquet(os.path.join(local_dir, 'puzzles_train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'puzzles_test.parquet'))
    
    print(f"Lichess puzzle dataset created successfully")