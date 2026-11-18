"""Per-code-block step budget so one program cannot consume the whole trial.

CaP-X only asks the model to regenerate after a code block returns. A block that
keeps moving until the trial's ``max_steps`` leaves no turn for regeneration.
Each block may use at most ``BLOCK_FRACTION`` of the cycles remaining when it
starts (but at least ``MIN_BLOCK_CYCLES``). When the allowance is used up, the
next control cycle raises ``StepBudgetExceeded``. It derives from BaseException
so generated ``except Exception`` blocks cannot swallow it; CaP-X's executor
catches BaseException and returns the traceback to the model. The remaining
budget is appended to the block's stdout either way.

Uses only ``_sim_step_count`` and ``max_steps`` of the low-level env, which the
gateway shim exposes as well.
"""
from __future__ import annotations

BLOCK_FRACTION = 0.5
MIN_BLOCK_CYCLES = 1500


class StepBudgetExceeded(BaseException):
    """The current code block used its share of the trial's control cycles."""


def block_allowance(remaining: int) -> int:
    if remaining <= MIN_BLOCK_CYCLES:
        return max(0, remaining)
    return max(MIN_BLOCK_CYCLES, int(BLOCK_FRACTION * remaining))


def check_block_budget(env) -> None:
    deadline = getattr(env, "_exp001_block_deadline", None)
    if deadline is None or env._sim_step_count < deadline:
        return
    remaining = env.max_steps - env._sim_step_count
    raise StepBudgetExceeded(
        f"This code block used its allowance of {env._exp001_block_allowance} control cycles "
        f"(about half of what remained). {remaining} of {env.max_steps} cycles remain for the "
        "trial. Regenerate a shorter program: reuse object positions already printed, avoid "
        "repeating full searches, and stop retrying the same failing motion.")


def get_step_budget(self) -> dict:
    """Return the control-cycle budget: {"used", "remaining", "limit", "block_remaining"}.

    Every motion consumes control cycles; the trial ends when "remaining" is 0.
    A single code block may use at most about half of the cycles that remained
    when it started ("block_remaining"); after that the block is stopped so the
    program can be regenerated.
    """
    env = self._env
    used, limit = int(env._sim_step_count), int(env.max_steps)
    deadline = getattr(env, "_exp001_block_deadline", None)
    block_remaining = None if deadline is None else max(0, int(deadline) - used)
    return {"used": used, "remaining": max(0, limit - used), "limit": limit,
            "block_remaining": block_remaining}


def install(code_env_class, api_class) -> None:
    original_exec = code_env_class._exec_user_code

    def exec_user_code(self, code):
        env = self.low_level_env
        start = int(getattr(env, "_sim_step_count", 0))
        limit = getattr(env, "max_steps", None)
        if limit is None:
            return original_exec(self, code)
        allowance = block_allowance(int(limit) - start)
        env._exp001_block_allowance = allowance
        env._exp001_block_deadline = start + allowance
        try:
            result = original_exec(self, code)
        finally:
            env._exp001_block_deadline = None
        used = int(env._sim_step_count) - start
        remaining = max(0, int(limit) - int(env._sim_step_count))
        note = (f"[budget] this code block used {used} control cycles; "
                f"{remaining} of {int(limit)} remain for the trial "
                f"(next block may use up to {block_allowance(remaining)}).\n")
        result = dict(result)
        result["stdout"] = (result.get("stdout") or "") + note
        return result

    code_env_class._exec_user_code = exec_user_code
    api_class.get_step_budget = get_step_budget
    original_functions = api_class.functions

    def functions(self):
        fns = dict(original_functions(self))
        fns["get_step_budget"] = self.get_step_budget
        return fns

    api_class.functions = functions
