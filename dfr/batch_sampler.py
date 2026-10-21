"""Deterministic prompt and image-seed schedules for distributed training."""

import hashlib

import torch
from torch.utils.data import BatchSampler


class PromptSeedBatchSampler(BatchSampler):
    """Build a global epoch schedule, then shard complete prompt groups.

    Each yielded item is ``(prompt_index, image_seeds)``.  A sampler instance
    yields one logical batch per epoch: all prompt groups assigned to its rank.
    Sampling minibatches are formed later, after prompts have been encoded once.
    """

    def __init__(
        self,
        dataset_size,
        num_prompts_per_epoch,
        num_samples_per_prompt,
        num_replicas=1,
        rank=0,
        seed=0,
    ):
        if num_prompts_per_epoch % num_replicas != 0:
            raise ValueError(
                f"num_prompts_per_epoch={num_prompts_per_epoch} must be divisible by "
                f"num_replicas={num_replicas}"
            )

        self.dataset_size = dataset_size
        self.num_prompts_per_epoch = num_prompts_per_epoch
        self.num_samples_per_prompt = num_samples_per_prompt
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _image_seed(self, prompt_index, sample_index):
        payload = f"{self.seed}:{self.epoch}:{prompt_index}:{sample_index}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)

    def __iter__(self):
        # Hashing avoids coupling the accepted config range to manual_seed's
        # integer range and keeps this stream separate from image seeds.
        selection_seed = self._image_seed("prompts", "selection")
        generator = torch.Generator().manual_seed(selection_seed)
        prompt_indices = torch.randperm(self.dataset_size, generator=generator)[
            : self.num_prompts_per_epoch
        ].tolist()

        # Strided sharding leaves every prompt group intact and makes the union
        # of rank-local schedules independent of the number of replicas.
        local_indices = prompt_indices[self.rank :: self.num_replicas]
        batch = []
        for prompt_index in local_indices:
            image_seeds = tuple(
                self._image_seed(prompt_index, sample_index)
                for sample_index in range(self.num_samples_per_prompt)
            )
            batch.append((prompt_index, image_seeds))
        yield batch

    def __len__(self):
        return 1
