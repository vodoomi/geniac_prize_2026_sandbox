"""LIBERO collection must see and be scored like the gateway evaluation."""
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import yaml

from libero_collect import parity as libero_parity
from libero_collect.plan import (EXP, SUBMISSION_CONFIG, PreflightError, check_task_config, taskpack_tasks)


def make_env_class(success_steps):
    class Handle:
        def __init__(self):
            self.t = 0
            self.env = NS(check_success=lambda: self.t in success_steps)

        def step(self, action):
            self.t += 1
            return None, float(self.t in success_steps), False, {}

    class Env:
        def __init__(self):
            self.handle = Handle()

        def reset(self):
            self.handle.t = 0
            self.handle.step(None)  # settling step, not scored
            self.handle.t = 0

        def compute_reward(self):
            return float(self.handle.t in success_steps)
    return Env


class LatchTests(unittest.TestCase):
    def test_success_reached_then_undone_still_counts(self):
        cls = make_env_class({3})
        libero_parity.install(cls)
        env = cls()
        env.reset()
        for _ in range(5):
            env.handle.step(None)
        self.assertTrue(env.task_completed())
        self.assertEqual(env.compute_reward(), 1.0)

    def test_reset_clears_and_handle_is_wrapped_once(self):
        cls = make_env_class({1})
        libero_parity.install(cls)
        libero_parity.install(cls)
        env = cls()
        env.reset()
        env.handle.step(None)
        self.assertTrue(env.task_completed())
        env.reset()
        self.assertFalse(env.task_completed())
        self.assertEqual(env.compute_reward(), 0.0)
        wrapped = env.handle.step
        env.reset()
        self.assertIs(env.handle.step, wrapped)

    def test_settling_success_at_reset_is_not_scored(self):
        cls = make_env_class({1})
        libero_parity.install(cls)
        env = cls()
        env.reset()  # first settling step "succeeds" before the wrapper is active
        env.reset()  # wrapped now; settling succeeds again but is cleared after reset
        self.assertFalse(env.task_completed())


class TaskConfigTests(unittest.TestCase):
    def setUp(self):
        self.tasks = taskpack_tasks()
        self.submission = yaml.safe_load(SUBMISSION_CONFIG.read_text())

    def test_every_generated_config_matches_the_evaluation(self):
        split = yaml.safe_load((EXP.parents[1] / "evaluation/split.yaml").read_text())
        dev = split["development_task_ids"]["libero"]
        self.assertEqual(len(dev), 104)
        for task in dev:
            cfg = yaml.safe_load((EXP / f"configs/libero_tasks/{task}.yaml").read_text())
            check_task_config(task, cfg, self.tasks, self.submission)
            self.assertNotIn("{libero_environment_goal}", cfg["env"]["cfg"]["prompt"])

    def test_old_style_configs_are_refused(self):
        cfg = yaml.safe_load((EXP / "configs/libero_tasks/libero_90_14.yaml").read_text())
        for mutate in (
            lambda c: c["env"]["cfg"].update(prompt=c["env"]["cfg"]["prompt"].replace(
                "Put the middle black bowl on the plate. Quaternions are WXYZ.", "{libero_environment_goal}")),
            lambda c: c["env"]["cfg"].update(prompt=c["env"]["cfg"]["prompt"].replace(
                "Put the middle black bowl on the plate. Quaternions are WXYZ.",
                "put the black bowl in the middle on the plate")),
            lambda c: c["env"]["cfg"]["low_level"].update(max_steps=4000),
            lambda c: c["env"]["cfg"].update(multi_turn_prompt="x"),
        ):
            bad = copy.deepcopy(cfg)
            mutate(bad)
            with self.assertRaises(PreflightError):
                check_task_config("libero_90_14", bad, self.tasks, self.submission)


if __name__ == "__main__":
    unittest.main()
