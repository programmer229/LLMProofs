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
Preprocess the MATH dataset to parquet format for confidence interval project
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math_utils import remove_boxed, last_boxed_only_string
import random


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    data_source = 'qwedsacf/competition_math'

    dataset = datasets.load_dataset(data_source, trust_remote_code=True)["train"]
    split_datasets = dataset.train_test_split(test_size=0.1)
    train_dataset = split_datasets['train']
    test_dataset = split_datasets['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):            
            question = example.pop('problem')
            instruction_following = "Show your full working out. You should explore and relfect often solving the problem like an expert mathmatician. Checking and reflecting after each step. State your final answer clearly within \\boxed{}. Lastly, state your confidence from 0 to 100 for your answer clearly within \\confidence{}."
            question = question + ' ' + instruction_following

            answer = example.pop('solution')
            try:
                solution = extract_solution(answer)
            except Exception as e:
                solution = answer # These are the proofs questions

            data = {
                "data_source": "conf",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
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

    local_dir = '/home/ubuntu/o1-replication/CustomTinyZero/data/conf'

    train_dataset.to_parquet(os.path.join(local_dir, 'train_conf_math.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_conf_math.parquet'))

    print(f"MATH dataset with confidence prompt appended created successfully")