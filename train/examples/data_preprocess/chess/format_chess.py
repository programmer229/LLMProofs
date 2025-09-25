# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Lichess puzzles dataset to parquet format
"""

import json
import os
import datasets
from datasets import Dataset
import argparse


def load_puzzles(json_file):
    """Load puzzles from jsonl file"""
    puzzles = []
    with open(json_file, 'r') as f:
        for line in f:
            puzzle = json.loads(line)
            puzzles.append(puzzle)
    return puzzles


def make_map_fn(split):
    def process_fn(example, idx):
        # Extract the first prompt content which contains the puzzle description
        question_raw = example['prompt'][0]['content']
        
        # Get the ground truth move
        answer_raw = example['reward_model']['ground_truth']
        
        data = {
            "data_source": "chess_puzzles",
            "prompt": [{
                "role": "user",
                "content": question_raw
            }],
            "ability": "chess",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer_raw
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'answer': answer_raw,
                'question': question_raw,
                'fen': example['extra_info']['fen'],
                'rating': example['extra_info']['rating'],
                'themes': example['extra_info']['themes'],
                'puzzle_id': example['extra_info']['puzzle_id']
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./lichess_puzzles')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--input_file', default='puzzles.json')
    parser.add_argument('--train_ratio', type=float, default=0.8)

    args = parser.parse_args()

    # Load puzzles
    input_path = os.path.join(args.local_dir, args.input_file)
    puzzles = load_puzzles(input_path)
    
    # Convert to dataset
    dataset = Dataset.from_list(puzzles)

    # Split into train/test
    dataset = dataset.train_test_split(train_size=args.train_ratio)
    
    # Add split and index information and reformat
    train_dataset = dataset['train'].map(function=make_map_fn('train'), with_indices=True)
    test_dataset = dataset['test'].map(function=make_map_fn('test'), with_indices=True)

    # Save to parquet
    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
