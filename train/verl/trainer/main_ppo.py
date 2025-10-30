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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import inspect
from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, multiply, countdown, chess, arc, dial_length, integration, conf, integration_numeric, llm_judge_integration, llm_judge_integration_sympy, llm_judge_svg, llm_judge_creative
from verl.utils.reward_score import llm_judge_proofs_test, llm_judge_proofs_train
from verl.utils.reward_score.utils import math_utils
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import sys
import datetime

def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH' or data_source == 'math' or data_source == 'probability':
        return math_utils.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif "chess" in data_source:
        return chess.compute_score
    elif "grid_transform" in data_source:
        return arc.compute_score
    elif "AI-MO/NuminaMath-CoT" in data_source:
        return math_utils.compute_score
    elif "llm_judge_creative" in data_source:
        return llm_judge_creative.compute_score
    elif "llm_judge_proof_test" in data_source:
        return llm_judge_proofs_test.compute_score
    elif "llm_judge_proof_train" in data_source:
        return llm_judge_proofs_train.compute_score
    elif "combined_math" in data_source:
        return math_utils.compute_score
    elif "di-zhang-fdu/AIME_1983_2024" in data_source:
        return math_utils.compute_score
    elif "dial_length" in data_source:
        return dial_length.compute_score
    elif "integration_numeric" == data_source:
        return integration_numeric.compute_score
    elif "llm_judge_integration" == data_source: # Formatting score comes from just being between <ANSWER> tags
        return llm_judge_integration.compute_score
    elif "llm_judge_integration_sympy" == data_source: # Formatting score comes from sympy parser
        return llm_judge_integration_sympy.compute_score
    elif "integration" == data_source:
        return integration.compute_score
    elif "conf" == data_source:
        return conf.compute_score
    elif "llm_judge_svg" in data_source:
        return llm_judge_svg.compute_score
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, max_response_length, reward_conversion_mode: str = "group_points") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.max_response_length = max_response_length
        self.reward_conversion_mode = reward_conversion_mode

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        data_source = data[0].non_tensor_batch['data_source']

        if "llm_judge" not in data_source:

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts']

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)

                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

                # select rm_score
                data_source = data_item.non_tensor_batch['data_source']
                compute_score_fn = _select_rm_score_fn(data_source)

                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, max_response_length=self.max_response_length, tokenizer=self.tokenizer)
                reward_tensor[i, valid_response_length - 1] = score

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print(sequences_str)

            return reward_tensor
    
        # This enables batch processing of the solutions by the LLM judge
        if "llm_judge" in data_source:

            solutions_batch = []
            ground_truth_batch = []
            valid_response_lengths = []
            extra_info_batch = []
            reward_model_batch = []

            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts']

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

                solutions_batch.append(sequences_str)
                ground_truth_batch.append(ground_truth)
                valid_response_lengths.append(valid_response_length)
                extra_info_batch.append(data_item.non_tensor_batch.get('extra_info', {}))
                reward_model_batch.append(data_item.non_tensor_batch.get('reward_model', {}))

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print(sequences_str)
                
            compute_score_fn = _select_rm_score_fn(data_source)

            compute_kwargs = dict(
                solutions_batch=solutions_batch,
                ground_truth_batch=ground_truth_batch,
                valid_response_lengths=valid_response_lengths,
                max_response_length=self.max_response_length,
                tokenizer=self.tokenizer,
                reward_tensor=reward_tensor,
            )
            if 'extra_info_batch' in inspect.signature(compute_score_fn).parameters:
                compute_kwargs['extra_info_batch'] = extra_info_batch
            if 'reward_model_batch' in inspect.signature(compute_score_fn).parameters:
                compute_kwargs['reward_model_batch'] = reward_model_batch
            if 'reward_conversion_mode' in inspect.signature(compute_score_fn).parameters:
                compute_kwargs['reward_conversion_mode'] = self.reward_conversion_mode

            reward_tensor = compute_score_fn(**compute_kwargs)

            return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id
    
    # if config.judge.location == "local":
    #     model = AutoModelForCausalLM.from_pretrained(config.judge.model)
    #     tokenizer = AutoTokenizer.from_pretrained(config.judge.model)

    train_reward_conversion_mode = getattr(config, "train_reward_conversion_mode", "group_points")
    reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        max_response_length=config.data.max_response_length,
        reward_conversion_mode=train_reward_conversion_mode,
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        max_response_length=config.data.max_response_length,
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
