"""Export the trained ReSplat initializer for public release.

Takes the in-repo training run (Lightning checkpoint + its v0 config.yaml) and
produces a clean, release-ready pair under a new directory:

  <out_dir>/config.yaml                      migrated to the current config version
  <out_dir>/checkpoints/<ckpt_name>          slimmed to just the initializer weights

The config is run through ``config_migrate.migrate`` (v0 -> current) and its
``output_dir`` is repointed at the release directory. The checkpoint keeps only
``state_dict`` (the ``initializer.*`` weights), dropping the Lightning training
state (optimizer moments, loops, callbacks, lr schedulers) that inference never
reads. The resulting layout matches what ``_find_config_for_checkpoint`` expects,
so it loads directly as ``pretrained_initializer`` (locally or via ``hf://``).

Usage:
  python -m optgs.scripts.dev.export_resplat_init \
      --src checkpoints/optgs/unified-dl3dv-8views/init \
      --out checkpoints/learn2splat/resplat_init
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from optgs.config_migrate import migrate
from optgs.misc.checkpointing import _rename_optimizer_attrs, find_latest_ckpt
from optgs.misc.io import cyan, read_omega_cfg


def export_config(src_cfg: Path, out_cfg: Path, out_dir: Path) -> None:
    cfg = read_omega_cfg(src_cfg)
    cfg = migrate(cfg)
    # Repoint output_dir at the release directory as a plain string (the source
    # carried a pickled PosixPath, which serializes as an ugly !!python tag).
    cfg.output_dir = str(out_dir)
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_cfg)
    print(cyan(f"Wrote migrated config -> {out_cfg}"))


def export_checkpoint(src_ckpt: Path, out_ckpt: Path) -> None:
    ckpt = torch.load(src_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # Keep only the initializer weights and apply the optimizer-attr renames
    # (a no-op here, since an init-only checkpoint has no optimizer keys).
    state_dict = {k: v for k, v in state_dict.items() if k.startswith("initializer.")}
    state_dict = _rename_optimizer_attrs(state_dict)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, out_ckpt)
    src_mb = src_ckpt.stat().st_size / 1e6
    out_mb = out_ckpt.stat().st_size / 1e6
    print(cyan(f"Wrote slimmed checkpoint -> {out_ckpt} "
               f"({len(state_dict)} keys, {src_mb:.0f}MB -> {out_mb:.0f}MB)"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="Source run dir containing config.yaml and checkpoints/")
    parser.add_argument("--out", type=Path, required=True,
                        help="Release dir to write config.yaml and checkpoints/ into")
    parser.add_argument("--ckpt-name", type=str, default=None,
                        help="Checkpoint filename under src/checkpoints (default: latest by step)")
    args = parser.parse_args()

    src_ckpt = (args.src / "checkpoints" / args.ckpt_name) if args.ckpt_name \
        else find_latest_ckpt(args.src / "checkpoints")
    out_ckpt = args.out / "checkpoints" / src_ckpt.name

    export_config(args.src / "config.yaml", args.out / "config.yaml", args.out)
    export_checkpoint(src_ckpt, out_ckpt)


if __name__ == "__main__":
    main()
