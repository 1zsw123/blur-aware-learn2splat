import os
import sys
import warnings
from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from optgs.misc.io import cyan

# Configure beartype and jaxtyping.
with install_import_hook(
        ("optgs",),
        ("beartype", "beartype"),
):
    from optgs.config import setup_cfg
    from optgs.dataset.data_module import DataModule
    from optgs.misc.step_tracker import StepTracker

# print torch device info
print(cyan(f"Torch version: {torch.__version__}"))
if torch.cuda.is_available():
    print(cyan(f"CUDA is available. Number of devices: {torch.cuda.device_count()}"))
    for i in range(torch.cuda.device_count()):
        print(cyan(f"Device {i}: {torch.cuda.get_device_name(i)}"))
else:
    print(cyan("CUDA is not available."))
    # raise ValueError("CUDA is required to run this code.")


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    # Set up configuration.
    cfg, cfg_dict, eval_cfg = setup_cfg(cfg_dict)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
    )

    if cfg.mode == "train":
        print("train:", len(data_module.train_dataloader()))
        print("val:", len(data_module.val_dataloader()))
        print("test:", len(data_module.test_dataloader()))
    else:
        print("test:", len(data_module.test_dataloader()))

    # DEBUGGING: loop over all data once to catch errors early
    for batch_idx, batch in enumerate(data_module.test_dataloader()):
        extrinsics = batch["context"]["extrinsics"]
        pose_norm = extrinsics.view(extrinsics.shape[0], -1).norm(dim=1)
        if pose_norm > 1e3:
            print(f"Batch {batch_idx}: pose norm {pose_norm.item():.4f} {extrinsics} {batch['scene']} {batch['context']['index']}")

        image = batch["context"]["image"][0, 0].permute(1, 2, 0).cpu().numpy()

        plt.figure()
        plt.imshow(image)
        plt.title(f"Batch {batch_idx}\n{batch['scene'][0]}")
        plt.show()

    print(cyan("DEBUG: Completed one full pass through the data loaders without errors. Exiting now."))
    sys.exit(0)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    if not torch.cuda.is_available():
        print("")
        print(cyan("=" * 80))
        print(cyan("CUDA is not available, running on CPU."))
        print(cyan("=" * 80))
        print("")

        # Print the hostname and current working directory.
    print(cyan("=" * 80))
    print(cyan(f"Starting training on {os.uname().nodename}, slurm job id: {os.environ.get('SLURM_JOB_ID', 'N/A')}"))
    print(cyan(f"Current working directory: {Path.cwd()}"))
    print(cyan("=" * 80))

    train()
