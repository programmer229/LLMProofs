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
Preprocess the Numina dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'EleutherAI/hendrycks_math'

    ds_algebra = datasets.load_dataset(data_source, 'algebra', trust_remote_code=True)["train"]
    ds_counting = datasets.load_dataset(data_source, 'counting_and_probability', trust_remote_code=True)["train"]
    ds_geometry = datasets.load_dataset(data_source, 'geometry', trust_remote_code=True)["train"]
    ds_interm_algebra = datasets.load_dataset(data_source, 'intermediate_algebra', trust_remote_code=True)["train"]
    ds_prealgebra = datasets.load_dataset(data_source, 'prealgebra', trust_remote_code=True)["train"]
    ds_number = datasets.load_dataset(data_source, 'number_theory', trust_remote_code=True)["train"]
    ds_precalc = datasets.load_dataset(data_source, 'precalculus', trust_remote_code=True)["train"]

    all_train_datasets = [ds_algebra, ds_counting, ds_geometry, ds_interm_algebra, ds_prealgebra, ds_number, ds_precalc]
    train_dataset = datasets.concatenate_datasets(all_train_datasets)

    ds_algebra = datasets.load_dataset(data_source, 'algebra', trust_remote_code=True)["test"]
    ds_counting = datasets.load_dataset(data_source, 'counting_and_probability', trust_remote_code=True)["test"]
    ds_geometry = datasets.load_dataset(data_source, 'geometry', trust_remote_code=True)["test"]
    ds_interm_algebra = datasets.load_dataset(data_source, 'intermediate_algebra', trust_remote_code=True)["test"]
    ds_prealgebra = datasets.load_dataset(data_source, 'prealgebra', trust_remote_code=True)["test"]
    ds_number = datasets.load_dataset(data_source, 'number_theory', trust_remote_code=True)["test"]
    ds_precalc = datasets.load_dataset(data_source, 'precalculus', trust_remote_code=True)["test"]

    all_test_datasets = [ds_algebra, ds_counting, ds_geometry, ds_interm_algebra, ds_prealgebra, ds_number, ds_precalc]
    test_dataset = datasets.concatenate_datasets(all_test_datasets)

    instruction_following = "Show your full working out. You should explore and relfect often solving the problem like an expert mathmatician. Checking and reflecting after each step. State your final answer clearly within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following

            answer = example.pop('solution')
            try:
                solution = extract_solution(answer)
            except Exception as e:
                solution = answer # These are the proofs questions

            data = {
                "data_source": data_source,
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

    local_dir = '/home/ubuntu/o1-replication/CustomTinyZero/data/hendrycks_math'

    train_dataset.to_parquet(os.path.join(local_dir, 'train_hendrycks_math.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_hendrycks_math.parquet'))

    print(f"Hendrycks Math dataset created successfully")