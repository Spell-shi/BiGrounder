"""CPU checks for ablation behavior; no model weights or Ray cluster required.

Run: python -m unittest discover -s tests -p 'test_bigrounder_ablation.py'
Requires numpy and Pillow. Heavy model dependencies are stubbed only in this test.
"""
import ast
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_environment():
    torch = ModuleType('torch')
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    bert = ModuleType('bert_score')
    bert.score = None
    spec = importlib.util.spec_from_file_location(
        '_bigrounder_test_env', ROOT / 'agent_system/environments/env_package/BiGrounder/env.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, {'torch': torch, 'bert_score': bert}):
        spec.loader.exec_module(module)
    return module


ENV = load_environment()


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class Batch:
    @staticmethod
    def from_single_dict(data):
        return SimpleNamespace(rows=data, batch={'input_ids': [0] * len(data)})


def load_collector():
    # Execute the actual constructor and batch-assembly method without importing
    # distributed/model libraries. Their method bodies are left unchanged.
    source = ast.parse((ROOT / 'agent_system/multi_turn_rollout/rollout_loop.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'TrajectoryCollector')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {'__init__', 'gather_rollout_data'}]
    namespace = {'np': np, 'DataProto': Batch, 'collate_fn': lambda rows: rows,
                 'List': list, 'Dict': dict, 'PreTrainedTokenizer': object}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<collector>', 'exec'), namespace)
    return namespace['TrajectoryCollector']


Collector = load_collector()


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.temp = tempfile.TemporaryDirectory()
        image = Path(self.temp.name) / 'image.png'
        Image.new('RGB', (100, 100)).save(image)
        self.task = {
            'messages': [{'role': 'system', 'content': ''},
                         {'role': 'user', 'content': '<image>Describe the target.'},
                         {'role': 'assistant', 'content': 'Click the save button.'}],
            'images': [str(image)], 'original_img_path': str(image),
            'gt_action': 'click(50, 50)', 'bbox': [40, 40, 60, 60], 'task_type_id': 3,
        }
        self.evaluator = patch.object(ENV, 'BERTScoreEvaluator')
        self.scorer = self.evaluator.start()
        self.scorer.return_value.calculate_similarity.return_value = np.array([0.9])

    def tearDown(self):
        self.evaluator.stop()
        self.temp.cleanup()
        self.silence.__exit__(None, None, None)

    def make_env(self, *, worker=0, group=4, task=None, **kwargs):
        return ENV.BiGrounderSingleEnv(
            [task or self.task], ENV.BidirectionalConfig(**kwargs),
            env_id=worker, group_n=group, num_workers=group,
        )

    def finish_forward(self, env, count):
        for step in range(count):
            _, reward, done, info = env.step('(50, 50)')
            self.assertEqual(reward, 1.0)
            self.assertEqual(done, step == count - 1)
        self.assertEqual(len(env.forward_rewards), count)
        return info

    def test_original_instruction_16_and_20_rollouts(self):
        for count in (16, 20):
            with self.subTest(count=count):
                env = self.make_env(group=1, instruction_source='original',
                                    update_reverse=False, forward_rollouts=count)
                obs, info = env.reset()
                self.assertEqual(info['phase'], 'forward')
                self.assertIn('Click the save button.', obs['text'])
                info = self.finish_forward(env, count)
                self.assertEqual(info['completed_steps'], count)
                self.scorer.assert_not_called()

    def test_default_online_and_binary_rewards(self):
        for mode, expected in [('semantic', 0.9), ('binary', 1.0)]:
            with self.subTest(mode=mode):
                env = self.make_env(reverse_reward_mode=mode)
                _, info = env.reset()
                self.assertEqual(info['phase'], 'reverse')
                obs, reward, done, _ = env.step('Select save.')
                self.assertAlmostEqual(reward, expected)
                self.assertFalse(done)
                self.assertIn('Select save.', obs['text'])
                info = self.finish_forward(env, 4)
                self.assertEqual(info['completed_steps'], 5)

    def test_offline_bank_selection_and_semantic_fallback(self):
        task = dict(self.task, offline_instructions=['Variant 0', 'Variant 1', 'Variant 2', 'Variant 3'])
        for worker in range(4):
            env = self.make_env(worker=worker, task=task, instruction_source='offline', update_reverse=False)
            obs, info = env.reset()
            self.assertEqual(info['phase'], 'forward')
            self.assertIn(f'Variant {worker}', obs['text'])
            self.finish_forward(env, 4)
        self.scorer.return_value.calculate_similarity.return_value = np.array([0.5])
        env = self.make_env(task=task, instruction_source='offline', update_reverse=False)
        obs, _ = env.reset()
        self.assertIn('Click the save button.', obs['text'])
        self.assertTrue(env.use_gt_forward)

    def test_missing_offline_bank_fails(self):
        with self.assertRaisesRegex(ValueError, 'offline_instructions'):
            self.make_env(instruction_source='offline', update_reverse=False)

    def test_invalid_settings_fail(self):
        for kwargs in [{'forward_rollouts': 0}, {'forward_rollouts': True},
                       {'instruction_source': 'frozen'}, {'update_reverse': 'false'},
                       {'instruction_source': 'original'}, {'reverse_reward_mode': 'bad'}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ENV.BidirectionalConfig(**kwargs)

    def collector(self, **flags):
        steps = flags.get('forward_rollouts', 4) + (flags.get('instruction_source', 'online') == 'online')
        config = Config(env=Config(env_name='bigrounder', bigrounder=flags,
                                   max_steps=steps, rollout=Config(n=4)))
        return Collector(config, None)

    def gather(self, collector, include_reverse=True, count=4):
        rows = []
        for worker in range(4):
            uid = f'traj_{worker}'
            samples = []
            if include_reverse:
                samples.append(dict(uid='sample_0_abc_reverse', rewards=0.9,
                                    active_masks=True, traj_uid=uid))
            for step in range(count):
                # Only worker 1 has a successful grounding rollout.
                reward = 1.0 if worker == 1 and step == 0 else 0.0
                samples.append(dict(uid=f'sample_0_abc_ins_w{worker}_forward', rewards=reward,
                                    active_masks=True, traj_uid=uid))
            rows.append(samples)
        return collector.gather_rollout_data(
            copy.deepcopy(rows), np.zeros(4), np.full(4, count + include_reverse), {},
            np.array([f'traj_{worker}' for worker in range(4)]),
        ).rows

    def test_cycle_rewards_and_grpo_grouping(self):
        rows = self.gather(self.collector())
        self.assertEqual(len(rows), 20)
        reverse = [r for r in rows if r['phase_type'] == 'reverse']
        self.assertEqual(len({r['uid'] for r in reverse}), 1)
        np.testing.assert_allclose([r['step_reward'] for r in reverse], [0, 0.9, 0, 0])
        forward = [r for r in rows if r['phase_type'] == 'forward']
        self.assertEqual(len({r['uid'] for r in forward}), 4)

    def test_disabling_reverse_updates_removes_reverse_training_rows(self):
        rows = self.gather(self.collector(update_reverse=False))
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(r['phase_type'] == 'forward' for r in rows))

    def test_disabling_cycle_keeps_semantic_reward(self):
        rows = self.gather(self.collector(cycle_verification=False))
        reverse = [r for r in rows if r['phase_type'] == 'reverse']
        np.testing.assert_allclose([r['step_reward'] for r in reverse], [0.9] * 4)

    def test_step_count_guard(self):
        collector = self.collector()
        config = copy.deepcopy(collector.config)
        config.env.max_steps = 6
        with self.assertRaisesRegex(ValueError, 'max_steps'):
            Collector(config, None)


if __name__ == '__main__':
    unittest.main()
