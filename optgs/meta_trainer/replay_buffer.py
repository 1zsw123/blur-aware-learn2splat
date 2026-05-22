import random
from dataclasses import dataclass, is_dataclass, fields, replace
from typing import Any

import torch

from optgs.dataset.data_types import BatchedExample
from optgs.model.types import Gaussians
from optgs.scene_trainer.optimizer.optimizer import OptimizerState


def to_device(obj: Any, device: torch.device | str, detach=True) -> Any:
    """
    Recursively moves all tensors (and nested dataclasses) to the given device.
    - Skips None fields
    - Works with nested dataclasses
    - Works with lists/tuples of tensors or dataclasses
    """
    if torch.is_tensor(obj):
        if detach:
            obj = obj.detach()
        return obj.to(device)

    elif is_dataclass(obj):
        kwargs = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            if val is not None:
                kwargs[f.name] = to_device(val, device, detach=detach)
        return replace(obj, **kwargs)

    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device, detach=detach) for v in obj)

    elif isinstance(obj, dict):
        return {k: to_device(v, device, detach=detach) for k, v in obj.items()}

    else:
        return obj  # Leave unchanged (e.g., int, float, str)


@dataclass
class GaussianEpisodeEntry:
    id: int
    t: int
    batch: BatchedExample
    gaussians: Gaussians
    state: OptimizerState | None = None
    info: dict[str, Any] | None = None


@dataclass
class ReplayBufferCfg:
    capacity: int  # number of snapshots to store
    sample_batch_size: int  # number of snapshots to sample when resuming training
    sample_prob: float | int  # probability of sampling from the buffer vs starting fresh
    insert_prob: float | int # probability of pushing to the buffer a new sample
    return_prob: float | int # probability of returning the sampled snapshot (vs discarding it)
    simulate_ahead: bool  # whether to simulate ahead before returning the updated snapshot
    simulate_ahead_min_steps: int  # min steps to simulate ahead
    simulate_ahead_max_steps: int  # max steps to simulate ahead
    simulate_ahead_grow: int  # number of steps to scale up the max steps over meta iterations
    max_t: int | None  # maximum number of inner steps per episode
    push_only_if_not_full: bool  # only push if buffer is not full
    remove_strategy_when_full: str  # strategy to remove entries when buffer is full: "oldest" or "random"


class EpisodeReplayBuffer:
    def __init__(self, cfg: ReplayBufferCfg):
        self.cfg = cfg
        self.buffer = []
        assert self.cfg.sample_batch_size == 1, "Only batch size of 1 is supported for now."

    def push(self, entry, to_cpu=True):
        """Store one snapshot (intermediate state of training).

        If the buffer is full, the oldest snapshot will be removed.
        """
        if to_cpu:
            entry = to_device(entry, 'cpu', detach=True)

        self.buffer.append(entry)
        if len(self.buffer) > self.cfg.capacity:
            if self.cfg.remove_strategy_when_full == "oldest":
                self.buffer.pop(0)  # remove oldest if full
            elif self.cfg.remove_strategy_when_full == "random":
                idx = random.randint(0, len(self.buffer) - 2)  # remove random except the newly added one
                del self.buffer[idx]
            else:
                raise ValueError("Invalid remove strategy when full")

    def sample(self, device, leave_batch_fn=None):
        """Return and remove a random element from the buffer."""
        if len(self.buffer) < self.cfg.sample_batch_size:
            raise ValueError("Not enough elements in the buffer to sample")

        # Sample random entries
        indices = random.sample(range(len(self.buffer)), self.cfg.sample_batch_size)
        sampled_entries = [self.buffer[i] for i in indices]

        # Remove from buffer by index (must go in reverse to avoid shifting)
        for idx in sorted(indices, reverse=True):
            del self.buffer[idx]

        assert self.cfg.sample_batch_size == 1, "Only batch size of 1 is supported for now."
        sampled_entries = sampled_entries[0]

        # Move to device
        if leave_batch_fn is not None:
            batch = sampled_entries.batch
            # should_move_batch = leave_batch_fn(batch)
        sampled_entries = to_device(sampled_entries, device)

        return sampled_entries

    def flipcoin(self, action: str):
        """Flip a coin to decide whether to sample or push."""
        if action == "sample":
            return random.random() < self.cfg.sample_prob
        elif action == "insert":
            return random.random() < self.cfg.insert_prob
        elif action == "return":
            return random.random() < self.cfg.return_prob
        else:
            raise ValueError("sample_or_push must be 'sample' or 'push'")

    def should_sample(self):
        buffer_is_not_full = len(self.buffer) < self.cfg.capacity
        if buffer_is_not_full:
            return False
        return len(self.buffer) >= self.cfg.sample_batch_size and self.flipcoin("sample")

    def should_push(self, new_sample: bool, t: int):
        if self.cfg.push_only_if_not_full and len(self.buffer) >= self.cfg.capacity:
            return  False # do not push if buffer is full

        if self.cfg.max_t is not None:
            if t >= self.cfg.max_t:
                return  # do not store entries beyond max_t

        if len(self.buffer) < self.cfg.capacity:
            # Always fill the buffer if possible
            return True

        if new_sample:
            return self.flipcoin("insert")
        else:
            return self.flipcoin("return")

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()