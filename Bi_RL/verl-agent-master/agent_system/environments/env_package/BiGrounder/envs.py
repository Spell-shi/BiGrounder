# -*- coding: utf-8 -*-
"""BiGrounder parallel environment - each worker loads data independently"""

import ray
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from .env import BiGrounderSingleEnv, BidirectionalConfig


@ray.remote
class BiGrounderWorker:
    """BiGrounder Worker - handles a single environment instance"""

    def __init__(
        self,
        tasks_json: str,
        config: BidirectionalConfig,
        env_id: int = 0,
        seed: int = 0,
        num_workers: int = 1,
        group_n: int = 4,  # new parameter
    ):
        """Initialize Worker"""
        # each worker loads data independently
        print(f"[Worker {env_id}] Loading tasks from {tasks_json}")

        with open(tasks_json, 'r') as f:
            all_tasks = json.load(f)

        # process task format
        tasks = []
        for t in all_tasks:
            if 'messages' in t and 'images' in t and len(t['messages']) >= 3:
                task_dict = {
                    'messages': t['messages'],
                    'images': t['images'],
                    'gt_action': t.get('gt_action', ''),
                    'original_img_path': t.get('original_img_path',
                        t['images'][0].replace('visualization_reverse_action', 'data_unzip_resize')),
                    # task type (paper Eq.6): 1=action grounding, 2=region grounding, 3=point grounding
                    'task_type_id': t.get('task_type_id', None),
                }

                if 'bbox' in t and isinstance(t['bbox'], list) and len(t['bbox']) == 4:
                    task_dict['bbox'] = t['bbox']
                if 'offline_instructions' in t:
                    task_dict['offline_instructions'] = t['offline_instructions']

                tasks.append(task_dict)

        # set up the environment, passing group_n
        self.env = BiGrounderSingleEnv(
            tasks=tasks,
            config=config,
            env_id=env_id,
            seed=seed,
            num_workers=num_workers,
            group_n=group_n,  # pass the group size
            shuffle_tasks=True,
        )
        self.env_id = env_id
        self.completed_episodes = 0
        self.auto_reset = True # set to True to run until max_step

    def reset(self, seed_for_reset: Optional[int] = None):
        """Reset the environment"""
        if seed_for_reset is not None:
            self.env.rng = np.random.RandomState(seed_for_reset)
        obs, info = self.env.reset()
        return obs, info

    def step(self, action: str):
        """Execute an action; automatically reset if the task is complete"""
        obs, reward, done, info = self.env.step(action)

        if done:
            self.completed_episodes += 1

            if self.auto_reset:
                # save completion info
                completion_info = {
                    'task_completed': True,
                    'success': info.get('success', False),
                    'final_step_reward': reward,  # reward of the last step
                    'phase': info.get('phase', 'unknown'),
                    'reverse_reward': info.get('reverse_reward', 0.0),
                    'forward_rewards_history': info.get('forward_rewards_history', []),
                    'avg_forward_reward': info.get('avg_forward_reward', 0.0),
                    'max_forward_reward': info.get('max_forward_reward', 0.0),
                }

                # reset to a new task
                new_seed = np.random.randint(0, 2**32)
                new_obs, new_info = self.reset(seed_for_reset=new_seed)

                # merge info
                merged_info = {
                    **info,
                    'auto_reset_triggered': True,
                    'completion_info': completion_info,
                    'new_task': {
                        'task_idx': new_info['task_idx'],
                        'phase': new_info['phase'],
                    },
                    'completed_episodes': self.completed_episodes,
                }

                # return the observation of the new task, but set done to False (continue training)
                return new_obs, reward, False, merged_info

        return obs, reward, done, info

    def get_env_id(self):
        """Get the environment ID"""
        return self.env_id

    def close(self):
        """Close the environment"""
        pass


class BiGrounderParallelEnvs:
    """BiGrounder multi-process environment"""

    def __init__(
        self,
        *,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        tasks_json: str,
        config: Optional[BidirectionalConfig] = None,
        resources_per_worker: Dict[str, Any] = {"num_cpus": 1.0},
        verbose: bool = False,
        **config_kwargs
    ):
        """Initialize the multi-process environment"""
        self.env_num = int(env_num)
        self.group_n = int(group_n)
        self.num_processes = self.env_num * self.group_n
        self.verbose = verbose

        print(f"[BiGrounderParallelEnvs] Initializing {self.num_processes} workers "
              f"({self.env_num} groups × {self.group_n} workers/group)")

        # validate the file path
        if not os.path.exists(tasks_json):
            raise FileNotFoundError(f"Tasks file not found: {tasks_json}")

        # quickly validate the file format
        print(f"[BiGrounderParallelEnvs] Validating tasks file: {tasks_json}")
        with open(tasks_json, 'r') as f:
            raw_tasks = json.load(f)
            valid_count = sum(1 for t in raw_tasks
                             if 'messages' in t and 'images' in t and len(t.get('messages', [])) >= 3)
            print(f"[BiGrounderParallelEnvs] Found {valid_count}/{len(raw_tasks)} valid tasks")

            # display task assignment info
            tasks_per_group = valid_count // self.env_num
            print(f"[BiGrounderParallelEnvs] Each group will get ~{tasks_per_group} tasks")

        self.tasks_json = tasks_json

        # create config
        if config is None:
            config = BidirectionalConfig(**config_kwargs)
        self.config = config

        # initialize Ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                _system_config={
                    "max_grpc_message_size": 1073741824,
                    "grpc_max_send_message_length": 1073741824,
                    "grpc_max_receive_message_length": 1073741824,
                }
            )
        
        # create workers
        print(f"[BiGrounderParallelEnvs] Creating workers with group-based task assignment")
        self.workers = []
        for i in range(self.num_processes):
            worker = BiGrounderWorker.options(**resources_per_worker).remote(
                tasks_json=self.tasks_json,
                config=config,
                env_id=i,
                seed=seed + i * 1000,
                num_workers=self.num_processes,
                group_n=self.group_n,  # pass the group size
            )
            self.workers.append(worker)
        
        print(f"[BiGrounderParallelEnvs] All {self.num_processes} workers initialized")
        print(f"  - {self.env_num} groups")
        print(f"  - {self.group_n} workers per group")
        print(f"  - Each group shares the same task pool")

    
    def reset(self):
        """Reset all environments"""
        seeds = np.random.randint(0, 2**32, size=self.num_processes).tolist()

        futures = [w.reset.remote(seeds[i]) for i, w in enumerate(self.workers)]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)

        return list(obs_list), list(info_list)

    def step(self, actions: List[str]):
        """Execute actions"""
        assert len(actions) == self.num_processes, \
            f"Expected {self.num_processes} actions, got {len(actions)}"

        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures)

        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        # temporary statistics within the batch
        batch_reverse_success = []
        batch_forward_success = []
        batch_both_success = []

        for i, (obs, r, d, info) in enumerate(results):
            obs_list.append(obs)
            reward_list.append(r)  # r is the immediate reward
            done_list.append(d)
            info_list.append(info)

        # return rewards and dones as numpy arrays
        return (
            obs_list,
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            info_list
        )

    def close(self):
        """Close all workers"""
        try:
            for w in self.workers:
                ray.kill(w)
        except Exception as e:
            print(f"[Warning] Error closing workers: {e}")

    def __del__(self):
        """Destructor"""
        try:
            self.close()
        except:
            pass


def build_bigrounder_envs(
    tasks_json: str,
    config_dict: Dict,
    seed: int,
    env_num: int,
    group_n: int,
    is_train: bool,
    resources_per_worker: Dict = None,
    num_gpus: int = None,
    verbose: bool = False,
) -> BiGrounderParallelEnvs:
    """Build BiGrounder environments"""

    print("\n" + "="*60)
    print("Building BiGrounder Environments")
    print("="*60)
    print(f"Parameters:")
    print(f"  - tasks_json: {tasks_json}")
    print(f"  - env_num: {env_num}")
    print(f"  - group_n: {group_n}")
    print(f"  - total workers: {env_num * group_n}")
    print(f"  - is_train: {is_train}")
    print(f"  - seed: {seed}")
    print(f"  - verbose: {verbose}")
    
    # default resource configuration
    if resources_per_worker is None:
        resources_per_worker = {"num_cpus": 1.0}

    print("\n" + "="*60)
    print("Building BiGrounder Environments")
    print("="*60)

    # add debugging: print the received config
    print(f"Received config_dict keys: {list(config_dict.keys())}")
    print(f"Config values:")
    for k, v in config_dict.items():
        print(f"  - {k}: {v}")

    # create config
    config = BidirectionalConfig(**config_dict)

    # add debugging: print the config actually used
    print(f"\nActual config values:")
    print(f"  - format_reward_weight: {config.format_reward_weight}")
    print(f"  - action_match_reward_weight: {config.action_match_reward_weight}")

    # create environment
    env = BiGrounderParallelEnvs(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        tasks_json=tasks_json,
        config=config,
        resources_per_worker=resources_per_worker,
        verbose=verbose
    )
    return env


def bigrounder_projection(texts: List[str], *args, **kwargs) -> Tuple[List[str], np.ndarray]:
    """
    Action validation - BiGrounder's lenient validation
    Only rejects clearly invalid outputs (empty strings, too short, etc.)
    """
    valids = []

    for text in texts:
        # consider valid as long as it is not empty or too short
        is_valid = bool(text) and isinstance(text, str) and len(text.strip()) > 5
        valids.append(is_valid)
    
    return texts, np.array(valids, dtype=bool)
