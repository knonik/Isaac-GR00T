# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.utils.dist_utils import run_or_wait_on_rank0


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        assert self.config.training.eval_strategy == "no", (
            "Sharded dataset does not support evaluation sets"
        )
        if self.config.data.loader == "knonik":
            return self._build_knonik(processor), None

        all_datasets = []
        all_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                # rank-0 writes stats; helper barriers before peers read them.
                with run_or_wait_on_rank0(label=f"generate_stats({dataset_path})") as is_rank0:
                    if is_rank0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                )
                datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        alpha = self.config.data.ds_weights_alpha
        if alpha is not None and len(all_datasets) > 1:
            ds_lengths = np.array([len(dataset) for dataset in all_datasets], dtype=np.float64)
            all_weights = (np.power(ds_lengths, alpha) / np.power(ds_lengths[0], alpha)).tolist()
            print(
                f"Applied ds_weights_alpha={alpha} across {len(all_datasets)} datasets; "
                "this overrides per-dataset mix_ratio sampling weights."
            )

        return (
            ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            ),
            None,
        )

    def _build_knonik(self, processor: BaseProcessor):
        from gr00t.data.dataset.knonik_dataset import KnonikGr00tDataset
        from gr00t.data.knonik import loader as kl, stats as knonik_stats
        from gr00t.data.knonik.source import resolve_manifest
        from gr00t.data.stats import LE_ROBOT_DATA_FILENAME

        specs = self.config.data.datasets
        assert len(specs) == 1 and len(specs[0].dataset_paths) == 1, (
            "data.loader=knonik takes exactly one dataset"
        )
        dataset_path = specs[0].dataset_paths[0]
        embodiment_tag = EmbodimentTag(specs[0].embodiment_tag)
        assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
        k = kl.merged_settings(self.config.data.knonik)
        manifest = resolve_manifest(k["source"], dataset_path, k["hub_cache_dir"] or None)
        meta_root = knonik_stats.meta_root(manifest, dataset_path)
        with run_or_wait_on_rank0(label=f"generate_stats({meta_root})") as is_rank0:
            if is_rank0 and meta_root == manifest.parent:
                knonik_stats.generate_stats(meta_root)
                knonik_stats.generate_rel_stats(meta_root, embodiment_tag)
            elif is_rank0 and next(meta_root.glob(LE_ROBOT_DATA_FILENAME), None) is not None:
                generate_stats(meta_root)
                generate_rel_stats(meta_root, embodiment_tag)
        if not (meta_root / "meta" / "stats.json").exists():
            raise FileNotFoundError(
                f"{meta_root}/meta/stats.json missing: re-run scripts/knonik/prepare_dataset.py"
            )

        training = self.config.training
        per_device = training.per_gpu_batch_size or training.global_batch_size // training.num_gpus
        return KnonikGr00tDataset(
            dataset_path=dataset_path,
            embodiment_tag=embodiment_tag,
            modality_configs=self.config.data.modality_configs[embodiment_tag.value],
            processor=processor,
            knonik=self.config.data.knonik,
            per_device_batch_size=per_device,
            steps_needed=training.max_steps * training.gradient_accumulation_steps,
            baseline_num_workers=training.dataloader_num_workers,
            seed=self.config.data.seed,
            shard_size=self.config.data.shard_size,
            episode_sampling_rate=self.config.data.episode_sampling_rate,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            allow_padding=self.config.data.allow_padding,
        )
