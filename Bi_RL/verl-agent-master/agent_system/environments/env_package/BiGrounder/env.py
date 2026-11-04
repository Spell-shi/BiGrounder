# -*- coding: utf-8 -*-
"""
Bidirectionally-verified BiGrounder environment implementation - uses BERTScore to evaluate similarity
Used for reinforcement learning training in the verl-agent framework
"""

import numpy as np
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from dataclasses import dataclass
from bert_score import score as bert_score
import torch


@dataclass
class BidirectionalConfig:
    """Bidirectional verification configuration

    The reward design corresponds to the method section of the paper, Eq.(6)-(8).
    - Reverse phase (semantic reward): BERTScore gating, see the reverse branch in step().
    - Forward phase (grounding reward):
        * Action grounding: R = R_A + R_C + R_F
            R_A (action_type_reward)  correct action type = 0.1
            R_C (coord_hit_reward)    predicted point falls inside GT bbox = 0.9
            R_F (format_penalty)      invalid format = -0.2 (valid format is 0)
        * Region grounding: R_bbox = α·exp(-E_g) + (1-α)·IoU   (Eq.7-8)
            E_g = δ̂cx⁴ + δ̂cy⁴ + γ·δ̂w⁴ + γ·δ̂h⁴,  γ=0.5, α=0.8
            All dimensions use a unified pixel tolerance τ = coord_tolerance = 6.
    Note: the 0.2/0.8 proportion of reverse/forward trajectories in the paper arises
    naturally from seq-mean-token-mean averaging over G reverse + G·K forward trajectories
    (G=K=4); it is not pre-scaled into the reward. Under GRPO the advantage is normalized
    within each group, so a single global scale applied uniformly to every reward in a
    group cancels out and leaves the advantage unchanged — hence no separate
    reverse/forward reward weight is needed.
    """
    similarity_threshold: float = 0.85
    bert_model_path: str = "./bert_models/roberta-large"
    coord_tolerance: float = 6.0          # unified pixel tolerance from the paper τ = 6
    # action grounding reward terms (Eq.6)
    format_reward_weight: float = 0.10    # R_A: correct action type reward
    action_match_reward_weight: float = 0.90  # R_C: coordinate hit (point falls inside bbox) reward
    format_penalty: float = -0.2          # R_F: invalid format penalty
    # region grounding dense reward (Eq.7-8)
    bbox_alpha: float = 0.8               # fusion weight α of geometric score and IoU
    bbox_gamma: float = 0.5               # size error decay coefficient γ
    bert_num_layers: int = 17
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    forward_rollouts: int = 4
    instruction_source: str = 'online'  # online policy, original annotation, or offline bank
    update_reverse: bool = True
    cycle_verification: bool = True
    reverse_reward_mode: str = 'semantic'  # raw BERTScore or binary semantic gate

    def __post_init__(self):
        if type(self.forward_rollouts) is not int or self.forward_rollouts < 1:
            raise ValueError('forward_rollouts must be a positive integer')
        if self.instruction_source not in {'online', 'original', 'offline'}:
            raise ValueError('instruction_source must be online, original, or offline')
        if type(self.update_reverse) is not bool or type(self.cycle_verification) is not bool:
            raise ValueError('update_reverse and cycle_verification must be booleans')
        if self.instruction_source != 'online' and self.update_reverse:
            raise ValueError('Reverse updates require instruction_source=online')
        if self.reverse_reward_mode not in {'semantic', 'binary'}:
            raise ValueError('reverse_reward_mode must be semantic or binary')


class BERTScoreEvaluator:
    """BERTScore evaluator (singleton pattern, avoids reloading the model)"""
    _instance = None
    
    def __new__(cls, model_path: str, num_layers: int, device: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_path: str, num_layers: int, device: str):
        if self._initialized:
            return
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ BERTScore model not found at {model_path}")
        
        self.model_path = model_path
        self.num_layers = num_layers
        self.device = device
        self._initialized = True
        print(f"✓ BERTScore evaluator initialized with {model_path} on {device}")
    
    def calculate_similarity(self, predictions: List[str], references: List[str]) -> np.ndarray:
        """
        Calculate BERTScore F1 score
        Args:
            predictions: list of predicted texts
            references: list of reference texts
        Returns:
            array of F1 scores
        """
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type=self.model_path,
            num_layers=self.num_layers,
            verbose=False,
            device=self.device
        )
        return F1.cpu().numpy()


class BiGrounderSingleEnv:
    """
    A single BiGrounder environment instance
    Implements bidirectional verification: action → instruction → action
    Uses BERTScore to evaluate instruction similarity
    """
    
    def __init__(
        self,
        tasks: List[Dict],
        config: BidirectionalConfig,
        env_id: int = 0,
        seed: int = 0,
        num_workers: int = 1,
        group_n: int = 4,
        shuffle_tasks: bool = True,
    ):
        """
        Args:
            tasks: list of tasks containing both reverse and forward information
            config: configuration object
            env_id: environment ID
            seed: random seed
            num_workers: total number of workers
            group_n: number of workers per group (used for GRPO)
            shuffle_tasks: whether to shuffle the task order
        """
        self.config = config
        self.env_id = env_id
        self.num_workers = num_workers
        self.group_n = group_n
        
        # ========== Key: assign tasks based on group ==========
        # Determine which group this worker belongs to
        self.group_idx = env_id // group_n
        self.worker_idx_in_group = env_id % group_n

        # Compute the number of groups and the tasks for each group
        num_groups = num_workers // group_n
        tasks_per_group = len(tasks) // num_groups
        remainder = len(tasks) % num_groups

        # Assign tasks to this group
        if self.group_idx < remainder:
            start_idx = self.group_idx * (tasks_per_group + 1)
            end_idx = start_idx + tasks_per_group + 1
        else:
            start_idx = self.group_idx * tasks_per_group + remainder
            end_idx = start_idx + tasks_per_group

        # The task pool of this group
        self.group_tasks = tasks[start_idx:end_idx]
        
        print(f"[Worker {env_id}] Group {self.group_idx}, "
              f"Worker {self.worker_idx_in_group}/{group_n}, "
              f"Tasks: {start_idx}-{end_idx-1} ({len(self.group_tasks)} tasks)")
        
        # Check whether there are any tasks
        if not self.group_tasks:
            raise ValueError(f"Worker {env_id} (Group {self.group_idx}) has no tasks assigned!")

        # ========== Simplification: use round-robin instead of random ==========
        # All workers start from task 0 to ensure synchronization
        self.current_task_index = 0  # current task index (within group_tasks)

        # Annotation-only grounding does not need semantic scoring.
        self.bert_evaluator = None if config.instruction_source == 'original' else BERTScoreEvaluator(
            model_path=config.bert_model_path,
            num_layers=config.bert_num_layers,
            device=config.device
        )
        if config.instruction_source == 'offline':
            for task in self.group_tasks:
                bank = task.get('offline_instructions')
                if (not isinstance(bank, list) or len(bank) < group_n
                        or any(not isinstance(x, str) or not x.strip() for x in bank)):
                    raise ValueError(
                        f'Offline mode requires at least {group_n} nonempty strings '
                        'in offline_instructions for every task. Generate them once '
                        'with the frozen checkpoint before training.'
                    )

        # State variable modifications
        self.current_task = None
        self.task_idx = None
        self.phase = 'reverse'
        self.step_count = 0
        self.reverse_instruction = None
        self.similarity_score = 0.0
        self.is_done = False
        self.total_episodes = 0

        # New state variables for tracking the forward phase
        self.forward_step_count = 0  # step count of the forward phase
        self.max_forward_steps = config.forward_rollouts
        self.reverse_completed = False  # marks whether the reverse phase is completed

        # Store rewards
        self.reverse_reward = 0.0
        self.forward_rewards = []  # store multiple forward rewards
        self.use_gt_forward = False

    def reset(self) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        # ========== Simplification: round-robin task selection ==========
        # All workers in the same group will select the same task
        self.current_task = self.group_tasks[self.current_task_index]

        # Compute the global task index (for display)
        if self.group_idx < (self.num_workers // self.group_n):
            tasks_per_group = len(self.group_tasks)
            self.task_idx = self.group_idx * tasks_per_group + self.current_task_index
        else:
            self.task_idx = self.current_task_index  # simplified handling

        # Advance to the next task (cyclically)
        self.current_task_index = (self.current_task_index + 1) % len(self.group_tasks)
        self.total_episodes += 1

        # Reset state
        self.phase = 'reverse'
        self.step_count = 0
        self.forward_step_count = 0  # reset forward step count
        self.reverse_instruction = None
        self.similarity_score = 0.0
        self.is_done = False
        self.reverse_completed = False

        # Reset rewards
        self.reverse_reward = 0.0
        self.forward_rewards = []
        self.use_gt_forward = False


        # Reverse phase: use the visualization image
        vis_image_path = self.current_task['images'][0]
        gt_instruction = self.current_task['messages'][2]['content']
        text_prompt = self.current_task['messages'][1]['content']
        image_path = vis_image_path
        anchor = f"task_{self.task_idx}_reverse"
        if self.config.instruction_source != 'online':
            self.phase = 'forward'
            self.reverse_completed = True
            self.use_gt_forward = True
            instruction = gt_instruction
            if self.config.instruction_source == 'offline':
                self.reverse_instruction = self.current_task['offline_instructions'][self.worker_idx_in_group]
                self.similarity_score = self._calculate_bertscore_similarity(
                    self.reverse_instruction, gt_instruction
                )
                self.use_gt_forward = self.similarity_score < self.config.similarity_threshold
                if not self.use_gt_forward:
                    instruction = self.reverse_instruction
            text_prompt = self._get_forward_prompt(instruction, self._get_task_type())
            image_path = self.current_task['original_img_path']
            anchor = f"task_{self.task_idx}_forward_step_0"
        image = self._load_image(image_path)
        
        obs = {
            'text': text_prompt,
            'image': [image],
            'anchor': anchor
        }
        
        info = {
            'env_id': self.env_id,
            'group_idx': self.group_idx,
            'worker_idx_in_group': self.worker_idx_in_group,
            'task_idx': self.task_idx,
            'phase': self.phase,
            'instruction_source': self.config.instruction_source,
            'step': 0,
            'gt_instruction': gt_instruction,
            'gt_action': self.current_task['gt_action'],
            'vis_image_path': vis_image_path,
            'original_image_path': self.current_task['original_img_path'],
            'data_source': 'bigrounder',
            'total_episodes': self.total_episodes,
            'tasks_in_group': len(self.group_tasks),
        }
        
        print(f"[Worker {self.env_id}] Reset to task {self.task_idx} (group {self.group_idx})")
        
        return obs, info
    
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step"""
    
        if self.is_done:
            dummy_obs = self._get_dummy_obs()
            return dummy_obs, 0.0, True, {'env_id': self.env_id, 'done': True}
        
        info = {
            'env_id': self.env_id,
            'task_idx': self.task_idx,
            'phase': self.phase,
            'step': self.step_count,
            'gt_action': self.current_task['gt_action'],
            'gt_instruction': self.current_task['messages'][2]['content'],
            'data_source': 'bigrounder',
        }
        
        # ===== Reverse phase =====
        if self.phase == 'reverse' and not self.reverse_completed:
            pred_instruction = action.strip()
            gt_instruction = self.current_task['messages'][2]['content']

            # Use BERTScore to compute similarity
            similarity = self._calculate_bertscore_similarity(pred_instruction, gt_instruction)
            self.similarity_score = similarity
            self.reverse_instruction = pred_instruction
            
            info.update({
                'pred_instruction': pred_instruction,
                'gt_instruction': gt_instruction,
                'bertscore_f1': similarity,
                'threshold': self.config.similarity_threshold,
            })

            # Paper Eq. (5): retain the raw BERTScore when the semantic gate
            # passes. Cycle verification is applied after the K forward
            # rollouts in gather_rollout_data().
            if similarity >= self.config.similarity_threshold:
                self.reverse_reward = similarity if self.config.reverse_reward_mode == 'semantic' else 1.0
                self.use_gt_forward = False
            else:
                self.reverse_reward = 0.0
                self.use_gt_forward = True  # use GT when similarity is insufficient

            reward = self.reverse_reward
            self.reverse_completed = True  # mark the reverse phase as completed

            # Immediately transition into the forward phase
            self.phase = 'forward'
            self.forward_step_count = 0

            # Build the forward observation
            instruction_to_use = gt_instruction if self.use_gt_forward else pred_instruction
            original_image_path = self.current_task['original_img_path']
            text_prompt = self._get_forward_prompt(instruction_to_use, self._get_task_type())
            image = self._load_image(original_image_path)
            
            next_obs = {
                'text': text_prompt,
                'image': [image],
                'anchor': f"task_{self.task_idx}_forward_step_{self.forward_step_count}"
            }
            
            info.update({
                'transition_to_forward': True,
                'reverse_success': similarity >= self.config.similarity_threshold,
                'reverse_reward': self.reverse_reward,
                'use_gt_forward': self.use_gt_forward,
                'forward_step': 0,
                'max_forward_steps': self.max_forward_steps,
            })
            
            return next_obs, reward, False, info
        
        # ===== Forward phase =====
        elif self.phase == 'forward':
            pred_text = action.strip()
            gt_action = self.current_task['gt_action']
            task_type = self._get_task_type()
            bbox = self.current_task.get('bbox', None)
            has_bbox = bbox is not None and len(bbox) == 4

            # Route the forward grounding reward by task_type_id (paper Eq.6)
            #   1 -> action grounding: R = R_A + R_C + R_F
            #   2 -> region grounding: R = R_bbox (Eq.7-8, IoU fusion)
            #   3 -> point  grounding: R = R_point (1 if point falls inside GT bbox, else 0)
            # Fall back to action grounding when task_type is missing.
            format_reward = 0.0
            match_reward = 0.0
            format_valid = False
            action_type_match = False
            match = False
            match_detail = {}
            point_reward = 0.0
            bbox_reward = None

            if task_type == 2:
                # ---- Region grounding: dense R_bbox ----
                pred_box = self._parse_box_output(pred_text)
                format_valid = pred_box is not None
                if not format_valid:
                    match_reward = self.config.format_penalty
                elif has_bbox:
                    bbox_reward = self._calculate_bbox_reward(pred_box, bbox)
                    match = True
                    match_reward = bbox_reward
                match_detail = {'type': 'region_grounding', 'pred_box': pred_box,
                                'bbox': bbox, 'bbox_reward': bbox_reward}

            elif task_type == 3:
                # ---- Point grounding: binary hit R_point ----
                pred_point = self._parse_point_output(pred_text)
                format_valid = pred_point is not None
                if not format_valid:
                    match_reward = self.config.format_penalty
                elif has_bbox:
                    px, py = pred_point
                    x1, y1, x2, y2 = bbox
                    hit = (x1 <= px <= x2) and (y1 <= py <= y2)
                    match = bool(hit)
                    point_reward = 1.0 if hit else 0.0
                    match_reward = point_reward
                match_detail = {'type': 'point_grounding', 'pred_point': pred_point,
                                'bbox': bbox, 'point_reward': point_reward}

            else:
                # ---- Action grounding: R_A + R_C + R_F (task_type==1 or missing fallback) ----
                parsed_action = self._parse_action_detail(pred_text)
                format_valid = parsed_action is not None

                gt_parsed = self._parse_action_detail(gt_action)
                if format_valid and gt_parsed:
                    action_type_match = parsed_action['type'] == gt_parsed['type']

                match, match_detail = self._match_actions(pred_text, gt_action)
                point_reward = match_detail.get('point_reward', 0.0)

                if not format_valid:
                    # R_F: invalid format penalty (valid format is 0)
                    format_reward = self.config.format_penalty
                elif action_type_match:
                    # R_A: correct action type
                    format_reward = self.config.format_reward_weight
                    if has_bbox:
                        # R_C: full score when predicted point falls inside GT bbox
                        if match:
                            match_reward = self.config.action_match_reward_weight
                    else:
                        # No-bbox fallback: scale R_C by the point-distance dense reward
                        match_reward = point_reward * self.config.action_match_reward_weight

            current_forward_reward = format_reward + match_reward
            self.forward_rewards.append(current_forward_reward)

            # Update step count
            self.forward_step_count += 1

            info.update({
                'pred_action': pred_text,
                'gt_action': gt_action,
                'task_type_id': task_type,
                'action_match': match,
                'match_detail': match_detail,
                'used_gt_instruction': self.use_gt_forward,
                'format_valid': format_valid,
                'action_type_match': action_type_match,
                'has_bbox': has_bbox,
                'point_reward': point_reward,
                'bbox_reward': bbox_reward if bbox_reward is not None else 0.0,
                'forward_step': self.forward_step_count,
                'max_forward_steps': self.max_forward_steps,
                'forward_reward': current_forward_reward,
                'reverse_reward': self.reverse_reward,
                'reward_breakdown': {
                    'format': format_reward,
                    'match': match_reward,
                    'point_reward': point_reward,
                    'bbox_reward': bbox_reward if bbox_reward is not None else 0.0,
                    'total_forward': current_forward_reward,
                    'task_type_id': task_type,
                    'has_bbox': has_bbox
                }
            })
            
            # Check whether all forward steps are completed
            if self.forward_step_count >= self.max_forward_steps:
                # All forward steps completed
                self.is_done = True

                info.update({
                    'all_forward_completed': True,
                    'forward_rewards_history': self.forward_rewards,
                    'avg_forward_reward': np.mean(self.forward_rewards) if self.forward_rewards else 0.0,
                    'max_forward_reward': max(self.forward_rewards) if self.forward_rewards else 0.0,
                    'completed_steps': self.forward_step_count + (self.config.instruction_source == 'online'),
                    'total_steps': self.max_forward_steps + (self.config.instruction_source == 'online'),
                })

                return self._get_dummy_obs(), current_forward_reward, True, info

            else:
                # Continue to the next forward step
                instruction_to_use = (self.current_task['messages'][2]['content']
                                     if self.use_gt_forward
                                     else self.reverse_instruction)
                
                original_image_path = self.current_task['original_img_path']
                text_prompt = self._get_forward_prompt(instruction_to_use, self._get_task_type())
                image = self._load_image(original_image_path)
                
                next_obs = {
                    'text': text_prompt,
                    'image': [image],
                    'anchor': f"task_{self.task_idx}_forward_step_{self.forward_step_count}"
                }
                
                info['continue_forward'] = True
                
                return next_obs, current_forward_reward, False, info
        
        raise RuntimeError(f"Invalid phase: {self.phase}")

    
    # ========== Helper methods ==========

    def _calculate_bertscore_similarity(self, pred: str, gt: str) -> float:
        """Use BERTScore to compute similarity"""
        try:
            f1_scores = self.bert_evaluator.calculate_similarity([pred], [gt])
            return float(f1_scores[0])
        except Exception as e:
            print(f"⚠️ BERTScore calculation failed: {e}, falling back to word overlap")
            return self._calculate_similarity_fallback(pred, gt)
    
    def _calculate_similarity_fallback(self, pred: str, gt: str) -> float:
        """Fallback similarity calculation (word overlap)"""
        pred_words = set(pred.lower().split())
        gt_words = set(gt.lower().split())
        if not gt_words:
            return 0.0
        intersection = pred_words & gt_words
        return len(intersection) / len(gt_words)

    def _get_task_type(self) -> Optional[int]:
        """Read the task type (paper Eq.6): 1=action grounding, 2=region grounding, 3=point grounding.
        Returns None when missing; the forward phase falls back to action grounding logic."""
        t = self.current_task.get('task_type_id', None)
        if t is None:
            return None
        try:
            return int(t)
        except (ValueError, TypeError):
            return None

    def _parse_box_output(self, text: str) -> Optional[Tuple[float, float, float, float]]:
        """Parse the predicted box for region grounding: of the form (x1,y1),(x2,y2) (may contain box markers)."""
        text = self._clean_action(text)
        m = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
        if m:
            return (float(m.group(1)), float(m.group(2)),
                    float(m.group(3)), float(m.group(4)))
        return None

    def _parse_point_output(self, text: str) -> Optional[Tuple[float, float]]:
        """Parse the predicted point for point grounding: of the form (x,y) (may contain box markers)."""
        text = self._clean_action(text)
        m = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', text)
        if m:
            return (float(m.group(1)), float(m.group(2)))
        return None
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load an image"""
        try:
            img = Image.open(path).convert('RGB')
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"⚠️ Failed to load image {path}: {e}")
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def _get_forward_prompt(self, instruction: str, task_type: Optional[int] = None) -> str:
        """Forward-phase prompt; the output format is decided by task_type_id (the three grounding types in paper Eq.6).

        - task_type == 2 (region grounding): requires outputting the two corner points of the target box.
        - task_type == 3 (point grounding): requires outputting the target point coordinates.
        - otherwise (action grounding or missing): outputs a GUI action.
        """
        if task_type == 2:
            return f"""<image>You are specialized in GUI grounding. Given an instruction, locate the target region on the screenshot.

    ## Output Format
    <|box_start|>(x1,y1),(x2,y2)<|box_end|>

    ## Instruction : {instruction}"""

        if task_type == 3:
            return f"""<image>You are specialized in GUI grounding. Given an instruction, locate the target point on the screenshot.

    ## Output Format
    <|box_start|>(x,y)<|box_end|>

    ## Instruction : {instruction}"""

        return f"""<image>You are a GUI agent. You are given a task, with screenshots. You need to perform the next action to complete the task.

    ## Output Format
    Action: ...

    ## Action Space
    click(start_box='<|box_start|>(x1,y1)<|box_end|>')
    rightClick(start_box='<|box_start|>(x1,y1)<|box_end|>')
    doubleClick(start_box='<|box_start|>(x1,y1)<|box_end|>')
    select(start_box='<|box_start|>(x1,y1)<|box_end|>') # To expand a dropdown menu or select an item from it.
    type(content='', start_box='<|box_start|>(x1,y1)<|box_end|>') # First click on the textbox, and then type the content.
    hotkey(key='')
    stop(reason='') ## If the item can not found in the image, give the reason
    scrolldown() ## If the all screenshot need to scroll down
    scrollup() ## If the all screenshot need to scroll up
    scrollmenu(start_box='<|box_start|>(x1,y1),(x2,y2)<|box_end|>') ## If the part of screenshot need to scroll and give the area need to scroll
    finish()

    ## User Instruction : {instruction}"""
    
    
    def _get_dummy_obs(self) -> Dict:
        """Return a dummy observation"""
        return {
            'text': "",
            'image': [np.zeros((224, 224, 3), dtype=np.uint8)],
            'anchor': ""
        }

    def _calculate_point_reward(self, pred_x: float, pred_y: float, gt_x: float, gt_y: float) -> float:
        """
        Compute the point-distance-based reward (quartic decay function)

        Args:
            pred_x, pred_y: predicted coordinates
            gt_x, gt_y: ground-truth coordinates

        Returns:
            reward value ∈ (0, 1]
        """
        # Get the tolerance threshold
        tau_x = self.config.coord_tolerance
        tau_y = self.config.coord_tolerance

        # Compute the normalized deviation
        delta_x_norm = abs(pred_x - gt_x) / tau_x
        delta_y_norm = abs(pred_y - gt_y) / tau_y

        # Quartic decay function
        reward = np.exp(-(delta_x_norm**4 + delta_y_norm**4))

        return float(reward)

    def _calculate_bbox_reward(self, pred_box, gt_box) -> float:
        """
        Dense reward R_bbox for region grounding (paper Eq.7-8).

        Decouple the bbox into center coordinates (cx, cy) and size (w, h), and apply a
        unified pixel tolerance τ = coord_tolerance to the four dimensions to construct
        the geometric energy term:
            E_g = δ̂cx⁴ + δ̂cy⁴ + γ·δ̂w⁴ + γ·δ̂h⁴,      δ̂k = |k_pred - k_gt| / τ
        The final reward is a weighted fusion of the geometric score and IoU:
            R_bbox = α·exp(-E_g) + (1-α)·IoU

        Args:
            pred_box: predicted box (x1, y1, x2, y2)
            gt_box:   ground-truth box (x1, y1, x2, y2)
        Returns:
            reward value ∈ [0, 1]
        """
        tau = self.config.coord_tolerance
        gamma = self.config.bbox_gamma
        alpha = self.config.bbox_alpha

        px1, py1, px2, py2 = pred_box
        gx1, gy1, gx2, gy2 = gt_box
        # Normalize corner order to avoid negative width/height
        px1, px2 = min(px1, px2), max(px1, px2)
        py1, py2 = min(py1, py2), max(py1, py2)
        gx1, gx2 = min(gx1, gx2), max(gx1, gx2)
        gy1, gy2 = min(gy1, gy2), max(gy1, gy2)

        # Decouple into center coordinates and size
        p_cx, p_cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        p_w, p_h = px2 - px1, py2 - py1
        g_cx, g_cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
        g_w, g_h = gx2 - gx1, gy2 - gy1

        # Tolerance-normalized deviations
        d_cx = abs(p_cx - g_cx) / tau
        d_cy = abs(p_cy - g_cy) / tau
        d_w = abs(p_w - g_w) / tau
        d_h = abs(p_h - g_h) / tau

        # Geometric energy term (size error down-weighted by γ)
        e_geom = d_cx**4 + d_cy**4 + gamma * d_w**4 + gamma * d_h**4
        geom_score = np.exp(-e_geom)

        # IoU
        inter_x1, inter_y1 = max(px1, gx1), max(py1, gy1)
        inter_x2, inter_y2 = min(px2, gx2), min(py2, gy2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        union_area = p_w * p_h + g_w * g_h - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0

        reward = alpha * geom_score + (1.0 - alpha) * iou
        return float(reward)

    def _match_actions(self, pred: str, gt: str) -> Tuple[bool, Dict]:
        """Match the predicted action against the GT action - using bbox verification or point reward"""

        pred_parsed = self._parse_action_detail(pred)
        gt_parsed = self._parse_action_detail(gt)

        if not pred_parsed or not gt_parsed:
            return False, {'type': 'parse_error', 'pred_parsed': pred_parsed, 'gt_parsed': gt_parsed, 'point_reward': 0.0}

        # Check the action type
        if pred_parsed['type'] != gt_parsed['type']:
            return False, {
                'type': 'action_type_mismatch',
                'pred_type': pred_parsed['type'],
                'gt_type': gt_parsed['type'],
                'point_reward': 0.0
            }

        # Get the bbox of the current task (if it exists)
        bbox = self.current_task.get('bbox', None)

        # Check action types that require coordinate verification
        if pred_parsed['type'] in ['click', 'rightClick', 'doubleClick', 'select', 'type']:
            px, py = pred_parsed['coords'][:2] if pred_parsed['coords'] else (None, None)

            if px is None or py is None:
                return False, {'type': 'missing_coordinates', 'point_reward': 0.0}

            # If there is a bbox, use bbox verification
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # Check whether the point is inside the bbox
                if x1 <= px <= x2 and y1 <= py <= y2:
                    return True, {
                        'type': f'{pred_parsed["type"]}_match_bbox',
                        'pred_coords': (px, py),
                        'bbox': bbox,
                        'in_bbox': True
                    }
                else:
                    return False, {
                        'type': f'{pred_parsed["type"]}_outside_bbox',
                        'pred_coords': (px, py),
                        'bbox': bbox,
                        'in_bbox': False
                    }
            
            # If there is no bbox, use the point reward
            else:
                gx, gy = gt_parsed['coords'][:2] if gt_parsed['coords'] else (None, None)
                if gx is None or gy is None:
                    return False, {'type': 'gt_missing_coordinates', 'point_reward': 0.0}

                # Compute the point reward
                point_reward = self._calculate_point_reward(px, py, gx, gy)
                distance = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                
                return True, {
                    'type': f'{pred_parsed["type"]}_match_point',
                    'distance': float(distance),
                    'tolerance': self.config.coord_tolerance,
                    'point_reward': point_reward,
                    'pred_coords': (px, py),
                    'gt_coords': (gx, gy)
                }
        
        # For drag and scrollmenu, check the start and end points
        elif pred_parsed['type'] in ['drag', 'scrollmenu']:
            if pred_parsed['coords'] and len(pred_parsed['coords']) >= 4:
                px1, py1, px2, py2 = pred_parsed['coords'][:4]

                # If there is a bbox, use the dense region reward R_bbox (Eq.7-8)
                if bbox is not None and len(bbox) == 4:
                    bbox_reward = self._calculate_bbox_reward((px1, py1, px2, py2), bbox)
                    return True, {
                        'type': f'{pred_parsed["type"]}_match_bbox',
                        'start_coords': (px1, py1),
                        'end_coords': (px2, py2),
                        'bbox': bbox,
                        'bbox_reward': bbox_reward,
                    }

                # Use the point reward when there is no bbox
                else:
                    if gt_parsed['coords'] and len(gt_parsed['coords']) >= 4:
                        gx1, gy1, gx2, gy2 = gt_parsed['coords'][:4]

                        # Compute the rewards for the start and end points separately
                        start_reward = self._calculate_point_reward(px1, py1, gx1, gy1)
                        end_reward = self._calculate_point_reward(px2, py2, gx2, gy2)

                        # Combined reward (average)
                        point_reward = (start_reward + end_reward) / 2.0
                        
                        distance_start = np.sqrt((px1 - gx1) ** 2 + (py1 - gy1) ** 2)
                        distance_end = np.sqrt((px2 - gx2) ** 2 + (py2 - gy2) ** 2)
                        
                        return True, {
                            'type': f'{pred_parsed["type"]}_match_point',
                            'distance_start': float(distance_start),
                            'distance_end': float(distance_end),
                            'tolerance': self.config.coord_tolerance,
                            'point_reward': point_reward,
                            'start_reward': start_reward,
                            'end_reward': end_reward
                        }
        
        # Actions that do not require coordinates
        elif pred_parsed['type'] in ['scrolldown', 'scrollup', 'finish', 'stop', 'hotkey']:
            return True, {'type': f'{pred_parsed["type"]}_match'}

        return False, {'type': 'unknown_action'}

    def _clean_action(self, action: str) -> str:
        """Clean the action, uniformly removing box markers"""
        # Remove <|box_start|> and <|box_end|> markers
        action = re.sub(r'<\|box_start\|>', '', action)
        action = re.sub(r'<\|box_end\|>', '', action)
        return action.strip()

    def _parse_action_detail(self, action: str) -> Optional[Dict]:
        """Parse action details (must contain the Action: prefix)"""
        # Uniformly clean box markers
        action = self._clean_action(action)

        # Updated regular expressions; must contain the Action: prefix
        # Action: click(start_box='(x1,y1)')
        RE_CLICK = re.compile(r"Action:\s*click\s*\(\s*start_box\s*=\s*['\"]?\((\d+),(\d+)\)['\"]?\s*\)", re.I)
        RE_RIGHT_CLICK = re.compile(r"Action:\s*rightclick\s*\(\s*start_box\s*=\s*['\"]?\((\d+),(\d+)\)['\"]?\s*\)", re.I)
        RE_DOUBLE_CLICK = re.compile(r"Action:\s*doubleclick\s*\(\s*start_box\s*=\s*['\"]?\((\d+),(\d+)\)['\"]?\s*\)", re.I)
        RE_SELECT = re.compile(r"Action:\s*select\s*\(\s*start_box\s*=\s*['\"]?\((\d+),(\d+)\)['\"]?\s*\)", re.I)
        
        # Action: drag(start_box='(x1,y1),(x2,y2)')
        RE_DRAG = re.compile(r"Action:\s*drag\s*\(\s*start_box\s*=\s*['\"]?\((\d+),(\d+)\),\((\d+),(\d+)\)['\"]?\s*\)", re.I)
        
        # Action: scrollmenu(start_box='(x1,y1),(x2,y2)')
        RE_SCROLL_MENU = re.compile(r"Action:\s*scrollmenu\s*\(\s*start_box\s*=\s*['\"]?\((\d+),(\d+)\),\((\d+),(\d+)\)['\"]?\s*\)", re.I)
        
        # Action: type(content='...', start_box='(x1,y1)')
        RE_TYPE = re.compile(r"Action:\s*type\s*\(\s*content\s*=\s*['\"]([^'\"]*)['\"],\s*start_box\s*=\s*['\"]?\((\d+),(\d+)\)['\"]?\s*\)", re.I)
        
        # Action: hotkey(key='...')
        RE_HOTKEY = re.compile(r"Action:\s*hotkey\s*\(\s*key\s*=\s*['\"]([^'\"]+)['\"]?\s*\)", re.I)
        
        # Action: stop(reason='...')
        RE_STOP = re.compile(r"Action:\s*stop\s*\(\s*reason\s*=\s*['\"]([^'\"]*)['\"]?\s*\)", re.I)
        
        # No-argument actions
        RE_SCROLL_DOWN = re.compile(r"Action:\s*scrolldown\s*\(\s*\)", re.I)
        RE_SCROLL_UP = re.compile(r"Action:\s*scrollup\s*\(\s*\)", re.I)
        RE_FINISH = re.compile(r"Action:\s*finish\s*\(\s*\)", re.I)
        
        # click
        match = RE_CLICK.search(action)
        if match:
            return {
                'type': 'click',
                'coords': (float(match.group(1)), float(match.group(2)))
            }
        
        # rightClick
        match = RE_RIGHT_CLICK.search(action)
        if match:
            return {
                'type': 'rightClick',
                'coords': (float(match.group(1)), float(match.group(2)))
            }
        
        # doubleClick
        match = RE_DOUBLE_CLICK.search(action)
        if match:
            return {
                'type': 'doubleClick',
                'coords': (float(match.group(1)), float(match.group(2)))
            }
        
        # select
        match = RE_SELECT.search(action)
        if match:
            return {
                'type': 'select',
                'coords': (float(match.group(1)), float(match.group(2)))
            }
        
        # drag
        match = RE_DRAG.search(action)
        if match:
            return {
                'type': 'drag',
                'coords': (
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4))
                )
            }
        
        # scrollmenu
        match = RE_SCROLL_MENU.search(action)
        if match:
            return {
                'type': 'scrollmenu',
                'coords': (
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4))
                )
            }
        
        # type
        match = RE_TYPE.search(action)
        if match:
            return {
                'type': 'type',
                'content': match.group(1),
                'coords': (float(match.group(2)), float(match.group(3)))
            }
        
        # hotkey
        match = RE_HOTKEY.search(action)
        if match:
            return {
                'type': 'hotkey',
                'key': match.group(1),
                'coords': None
            }
        
        # stop
        match = RE_STOP.search(action)
        if match:
            return {
                'type': 'stop',
                'reason': match.group(1),
                'coords': None
            }
        
        # scrolldown
        match = RE_SCROLL_DOWN.search(action)
        if match:
            return {'type': 'scrolldown', 'coords': None}
        
        # scrollup
        match = RE_SCROLL_UP.search(action)
        if match:
            return {'type': 'scrollup', 'coords': None}
        
        # finish
        match = RE_FINISH.search(action)
        if match:
            return {'type': 'finish', 'coords': None}
        
        return None
