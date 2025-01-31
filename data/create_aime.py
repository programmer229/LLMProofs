"""
Create AIME traina and test datasets (excluding 2024)
"""

import os
import datasets

if __name__ == '__main__':
    data_source = 'di-zhang-fdu/AIME_1983_2024'

    dataset = datasets.load_dataset(data_source, trust_remote_code=True)["train"]
    dataset = dataset.filter(lambda x: not int(x['Year']) == 2024) # Remove problems from 2024

    split_datasets = dataset.train_test_split(test_size=0.1)
    train_dataset = split_datasets['train']
    test_dataset = split_datasets['test']

    instruction_following = "Show your full working out. You should explore and relfect often solving the problem like an expert mathmatician. Checking and reflecting after each step. State your final answer clearly within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('Question')

            question = question + ' ' + instruction_following

            solution = example.pop('Answer')

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "aime",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = '/home/ubuntu/o1-replication/CustomTinyZero/data/aime'

    train_dataset.to_parquet(os.path.join(local_dir, 'train_aime.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_aime.parquet'))

    print(f"AIME dataset created successfully")