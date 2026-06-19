#!/usr/bin/env bash
# Upload the released ReSplat initializer (config + slimmed checkpoint) to the
# Hugging Face model repo, into the resplat_init/ subfolder.
#
# Pairs with export_resplat_init.py: that script stages the release-ready pair
# under $LOCAL_DIR; this script pushes that folder to the Hub at $REMOTE_DIR so
# it loads as pretrained_initializer=hf://$REPO/$REMOTE_DIR/checkpoints/<ckpt>.
#
# Rerun after retraining: re-stage with export_resplat_init.py, then run this
# again. Uploads are git commits that overwrite the same paths, so no deletion
# is needed between versions.
#
# Requires a WRITE token with access to the autonomousvision org:
#   conda run -n resplat hf auth login        # paste a Write token
#
# Usage:
#   bash optgs/scripts/dev/upload_resplat_init.sh
set -euo pipefail

REPO="autonomousvision/learn2splat"
LOCAL_DIR="checkpoints/learn2splat/resplat_init"
REMOTE_DIR="resplat_init"

# 1. Upload the staged folder. Mirrors $LOCAL_DIR onto $REPO at $REMOTE_DIR:
#      $REMOTE_DIR/config.yaml
#      $REMOTE_DIR/checkpoints/<ckpt_name>
conda run -n resplat hf upload "$REPO" "$LOCAL_DIR" "$REMOTE_DIR" \
  --repo-type model \
  --commit-message "Add ReSplat initializer (resplat_init): ckpt + migrated config"

# 2. List the resulting repo files to confirm the layout.
conda run -n resplat python -c "from huggingface_hub import HfApi; print('\n'.join(HfApi().list_repo_files('$REPO')))"

# --- One-time rename note -----------------------------------------------------
# The initializer originally lived under init/. To drop that stale folder
# (including subdirs) after the first resplat_init/ upload, run once:
#   conda run -n resplat hf repo-files "$REPO" delete "init/**" --repo-type model \
#     --commit-message "Remove misnamed init/ folder (renamed to resplat_init)"