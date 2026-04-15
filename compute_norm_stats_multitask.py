"""Compute normalisation statistics for the multi-task Baxter dataset.

Must be run before train_multitask.py.  Reads local/baxter_multitask and
writes per-joint mean/std for state and actions to:
  openpi/assets/pi05_baxter_multitask/local/baxter_multitask/norm_stats.json

Usage (run from the openpi directory with uv):
  cd openpi
  uv run ../compute_norm_stats_multitask.py
"""

import pathlib

import numpy as np
import tqdm

import openpi.models.pi0_config as pi0_config
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
from openpi.training.config import DataConfig, LeRobotLiberoDataConfig


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main() -> None:
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )

    data_factory = LeRobotLiberoDataConfig(
        repo_id="local/baxter_multitask",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    )

    # assets_dir mirrors what TrainConfig would compute for name="pi05_baxter_multitask"
    assets_dir = pathlib.Path("./assets/pi05_baxter_multitask").resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)

    data_config = data_factory.create(assets_dir, model_config)

    dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    batch_size = 4
    num_batches = len(dataset) // batch_size
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        num_batches=num_batches,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing norm stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = assets_dir / data_config.repo_id
    print(f"\nWriting stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print("Done.")


if __name__ == "__main__":
    main()
