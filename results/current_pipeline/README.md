# Current Pipeline Results

This directory is the compact, paper-facing index for the currently retained
results. It does not move, replace, or relabel the original experiment roots.
Large checkpoints, point clouds, optimizer states, and training caches remain
outside Git and are not duplicated here.

## Result sets

| Dataset | Scope | PSNR | SSIM | LPIPS | Selection |
|---|---:|---:|---:|---:|---|
| Deblur-NeRF Motion | 10 scenes | 45.840594 | 0.993318 | 0.005101 | best-PSNR checkpoint; LPIPS at 50K |
| Deblur-NeRF Defocus | standard 10 scenes | 41.932874 | 0.985798 | 0.013421 | best-PSNR checkpoint; LPIPS at 50K |
| Deblur-NeRF main | Motion10 + Defocus10 | 43.886734 | 0.989558 | 0.009261 | paper-facing 20-scene aggregate |
| TUM-RGBD | 3 official-keyframe scenes | 34.952986 | 0.954836 | withheld | 50K local-joint run |
| PRISM3D | 8 scenes | 37.794834 | 0.937011 | 0.046374 | uniform accepted 10K run |

`defocus_bush` is retained as an additional eleventh Defocus scene. Including
it produces a 21-scene Deblur-NeRF aggregate of 43.731933 PSNR, 0.989167 SSIM,
and 0.009484 LPIPS. It is kept separate from the standard 20-scene main table.

## Protocol boundary

The three result sets share the Learn2Splat/Blur-LeGS reconstruction family,
but they are not a single bit-identical configuration:

- the retained Deblur-NeRF and TUM results use the EVSSM restoration teacher;
- the accepted PRISM3D run uses TURTLE stage-1 step 24K, a 25x25 BPN kernel,
  dilation 2, latent blur assignment, and Laplacian-surplus supervision;
- Deblur-NeRF PSNR/SSIM use the best measured checkpoint per scene while its
  LPIPS column is measured at 50K;
- TUM uses the official-keyframe evaluator and its exact retained run has no
  LPIPS postprocess;
- PRISM3D reports one uniform 10K checkpoint across all eight scenes.

These distinctions must remain visible in tables and paper claims. See each
dataset subdirectory for per-scene values and exact provenance.

## Artifact locations

- Code and compact tables: this Git repository.
- Receipts, selected visualizations, and integrity manifests:
  `qizhangslam/blur-aware-learn2splat-prism3d` on Hugging Face under
  `current_pipeline/`.
- Existing Hugging Face paths such as `accepted/`, `publication_visuals/`, and
  `intermediate_visuals/` are immutable inputs to this index and are not moved.
