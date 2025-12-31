"""
ARC-AGI-3 Reward Model for ThetaEvolve

This module provides the async reward function for ARC-AGI-3 game environment,
compatible with ThetaEvolve's rm_hub pattern.
"""

from typing import Any, Dict, Optional

# Globally registered gym (same process as rollout manager)
_GYM = None


def set_gym(gym):
    """Register the ARC-AGI-3 gym instance for reward computation."""
    global _GYM
    _GYM = gym


def get_gym():
    """Get the registered gym instance."""
    return _GYM


async def arc_agi3_rm(args, sample) -> Dict[str, Any]:
    """
    Score LLM output using ARC-AGI-3 gym and return reward dictionary.

    This function is called by the rollout system to compute rewards
    for each LLM response.

    Args:
        args: Training arguments
        sample: Sample object with response and metadata

    Returns:
        Dict containing at least args.reward_key with scalar reward
    """
    default_low_score = -1.0
    reward_key = getattr(args, "reward_key", "reward")

    if _GYM is None:
        return {
            reward_key: default_low_score,
            "error": "gym_not_initialized",
        }

    metadata = sample.metadata or {}
    if isinstance(metadata, str):
        # Safety check
        return {
            reward_key: default_low_score,
            "error": "metadata_should_be_dict",
        }

    # Check if this is an ARC-AGI-3 sample
    if not metadata.get("arc_agi3", False):
        return {
            reward_key: default_low_score,
            "error": "not_arc_agi3_sample",
        }

    try:
        # Call gym's response_scorer
        result = await _GYM.response_scorer(sample.response or "", metadata)

    except Exception as e:
        print(f"[arc_agi3_rm] Exception: {e}")
        return {
            reward_key: default_low_score,
            "error": f"exception:{str(e)[:200]}",
        }

    if result is None or result.child_metrics is None:
        print("[arc_agi3_rm] Result is None or child_metrics is None")
        return {reward_key: default_low_score}

    # Extract reward from child_metrics
    metrics = result.child_metrics
    reward_val = metrics.get("reward", default_low_score)

    # Sanitize reward
    reward_val = _sanitize_reward(reward_val, default_low_score)

    # Build output dict
    out = {
        reward_key: reward_val,
        "metrics": metrics,
        "episode_id": result.episode_id,
        "step_in_episode": result.step_in_episode,
        "action_taken": result.action_taken,
        "score_delta": result.score_delta,
        "episode_finished": result.episode_finished,
        "finish_state": result.finish_state,
    }

    if result.iteration_time is not None:
        out["iteration_time"] = result.iteration_time

    if result.artifacts:
        out["artifacts"] = result.artifacts

    return out


def _sanitize_reward(
    reward_val: float,
    default_low_score: float,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
) -> float:
    """Sanitize reward value: handle NaN/Inf and clip to safe range."""
    import math

    if reward_val is None:
        return default_low_score

    if math.isnan(reward_val) or math.isinf(reward_val):
        print(f"[WARNING] Invalid reward {reward_val}, using {default_low_score}")
        return default_low_score

    clipped = max(clip_min, min(clip_max, reward_val))
    if abs(reward_val - clipped) > 1e-6:
        print(f"[WARNING] Reward clipped: {reward_val:.2e} -> {clipped:.2f}")

    return clipped


# ============================================================================
# Distributed Reward Computation Utilities
# ============================================================================

def compute_episode_distributed_rewards(
    episode_rewards: list,
    gamma: float = 0.99,
    normalize: bool = True,
) -> list:
    """
    Compute distributed rewards across episode actions using discount factor.

    This is a utility function for post-processing episode rewards.
    The actual reward distribution strategy can be customized.

    Args:
        episode_rewards: List of per-step rewards
        gamma: Discount factor for future rewards
        normalize: Whether to normalize rewards

    Returns:
        List of adjusted rewards
    """
    if not episode_rewards:
        return []

    n = len(episode_rewards)
    distributed = [0.0] * n

    # Backward pass: accumulate discounted future rewards
    running = 0.0
    for i in range(n - 1, -1, -1):
        running = episode_rewards[i] + gamma * running
        distributed[i] = running

    if normalize and n > 1:
        mean = sum(distributed) / n
        std = (sum((r - mean) ** 2 for r in distributed) / n) ** 0.5
        if std > 1e-8:
            distributed = [(r - mean) / std for r in distributed]

    return distributed

