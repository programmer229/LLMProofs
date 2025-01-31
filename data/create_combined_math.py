"""
Create a combined math dataset from Hendrycks and Competition Math datasets, called combined_math
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

    data_source_competition = 'qwedsacf/competition_math'

    dataset_competition = datasets.load_dataset(data_source_competition, trust_remote_code=True)["train"]
    split_datasets_competition = dataset_competition.train_test_split(test_size=0.1)
    train_dataset_competition = split_datasets_competition['train']
    test_dataset_competition = split_datasets_competition['test']

    print(f"Competition Math dataset created successfully")

    data_source_hendrycks = 'EleutherAI/hendrycks_math'

    ds_algebra = datasets.load_dataset(data_source_hendrycks, 'algebra', trust_remote_code=True)["train"]
    ds_counting = datasets.load_dataset(data_source_hendrycks, 'counting_and_probability', trust_remote_code=True)["train"]
    ds_geometry = datasets.load_dataset(data_source_hendrycks, 'geometry', trust_remote_code=True)["train"]
    ds_interm_algebra = datasets.load_dataset(data_source_hendrycks, 'intermediate_algebra', trust_remote_code=True)["train"]
    ds_prealgebra = datasets.load_dataset(data_source_hendrycks, 'prealgebra', trust_remote_code=True)["train"]
    ds_number = datasets.load_dataset(data_source_hendrycks, 'number_theory', trust_remote_code=True)["train"]
    ds_precalc = datasets.load_dataset(data_source_hendrycks, 'precalculus', trust_remote_code=True)["train"]

    ds_algebra_test = datasets.load_dataset(data_source_hendrycks, 'algebra', trust_remote_code=True)["test"]
    ds_counting_test = datasets.load_dataset(data_source_hendrycks, 'counting_and_probability', trust_remote_code=True)["test"]
    ds_geometry_test = datasets.load_dataset(data_source_hendrycks, 'geometry', trust_remote_code=True)["test"]
    ds_interm_algebra_test = datasets.load_dataset(data_source_hendrycks, 'intermediate_algebra', trust_remote_code=True)["test"]
    ds_prealgebra_test = datasets.load_dataset(data_source_hendrycks, 'prealgebra', trust_remote_code=True)["test"]
    ds_number_test = datasets.load_dataset(data_source_hendrycks, 'number_theory', trust_remote_code=True)["test"]
    ds_precalc_test = datasets.load_dataset(data_source_hendrycks, 'precalculus', trust_remote_code=True)["test"]

    print(f"Hendrycks Math dataset created successfully")

    all_train_datasets = [ds_algebra, ds_counting, ds_geometry, ds_interm_algebra, ds_prealgebra, ds_number, ds_precalc, train_dataset_competition]
    train_dataset = datasets.concatenate_datasets(all_train_datasets)

    all_test_datasets = [ds_algebra_test, ds_counting_test, ds_geometry_test, ds_interm_algebra_test, ds_prealgebra_test, ds_number_test, ds_precalc_test, test_dataset_competition]
    test_dataset = datasets.concatenate_datasets(all_test_datasets)

    print(f"Combined Math dataset created successfully")

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
                "data_source": "combined_math",
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

    local_dir = '/home/ubuntu/o1-replication/CustomTinyZero/data/combined_math'

    train_dataset.to_parquet(os.path.join(local_dir, 'train_combined_math.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_combined_math.parquet'))

    print(f"Hendrycks Math dataset created successfully")