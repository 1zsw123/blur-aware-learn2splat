import warnings

from torch.utils.data import Dataset
from typing import Type

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_dl3dv import DatasetDL3DV, DatasetDL3DVCfg
from .dataset_colmap import DatasetColmap, DatasetColmapCfg
from .dataset_scannet import DatasetScannet, DatasetScannetCfg
from .data_types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Type[Dataset]] = {
    "re10k": DatasetRE10k,
    "dl3dv": DatasetDL3DV,
    "colmap": DatasetColmap,
    "scannet": DatasetScannet,
}


DatasetCfg = DatasetRE10kCfg | DatasetDL3DVCfg | DatasetColmapCfg | DatasetScannetCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    print(f"Using dataset: {cfg.name}")
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )

    return DATASETS[cfg.name](cfg, stage, view_sampler)
