"""Score in-process LIBERO collection the way the gateway scores a submission.

The gateway counts a trial successful if ``task_completed()`` (LIBERO's
``check_success``) holds after any control cycle within ``trial_steplimit``
(``capx/gateway/gateway.py``, ``libero_adapter.py``). The in-process low level
reports only the reward of the last simulator step, so a task completed and later
undone would be scored as a failure. This latches success after every
``handle.step`` exactly like the gateway.

The step limit itself is the simulator horizon (``low_level.max_steps``), which
``libero_collect/prepare.py`` sets to the taskpack's ``trial_steplimit``; LIBERO's settling
steps at reset are taken back off that count, as in the gateway. Stepping past it
raises robosuite's "executing action in terminated episode", which
``capx.envs.trial`` treats as truncation.
"""
from __future__ import annotations


def install(env_class) -> None:
    if getattr(env_class, "_exp001_libero_parity", False):
        return
    original_reset = env_class.reset
    original_reward = env_class.compute_reward

    def reset(self, *args, **kwargs):
        self._exp001_success = False
        result = original_reset(self, *args, **kwargs)
        handle = self.handle
        if not getattr(handle, "_exp001_latched", False):
            raw_step = handle.step

            def step(action):
                out = raw_step(action)
                owner = handle._exp001_owner
                if not owner._exp001_success and bool(handle.env.check_success()):
                    owner._exp001_success = True
                return out

            handle.step = step
            handle._exp001_latched = True
        # Steps taken before this point (reset settling) are not scored.
        handle._exp001_owner = self
        self._exp001_success = False
        return result

    def task_completed(self) -> bool:
        return bool(getattr(self, "_exp001_success", False))

    def compute_reward(self) -> float:
        return 1.0 if task_completed(self) else original_reward(self)

    env_class.reset = reset
    env_class.task_completed = task_completed
    env_class.compute_reward = compute_reward
    env_class._exp001_libero_parity = True
