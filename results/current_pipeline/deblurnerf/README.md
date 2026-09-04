# Deblur-NeRF Results

The paper-facing scope is Motion10 plus the standard Defocus10, for 20 scenes.
`defocus_bush` is provided as a separately labeled extension and must not be
silently folded into the main aggregate.

- `per_scene_best_psnr.csv`: selected checkpoint and corresponding PSNR/SSIM.
- `per_scene_50k_lpips.csv`: 50K LPIPS with the matching 50K PSNR/SSIM.
- Git-recorded code heads vary by queue: `9e8bb25`, `0f23433`, and the later
  bush/seal supplement. The result manifest on Hugging Face binds each receipt.

The main PSNR/SSIM table keeps the original `defocus_seal` receipt so the
previously accepted aggregate remains unchanged. Its LPIPS was unavailable in
that receipt and is taken from the separately labeled seal supplement; the
supplement's accompanying 50K PSNR/SSIM remain only in the LPIPS table.

For comparison, the retained Unblur-SLAM paper aggregate is 29.49 PSNR,
0.9213 SSIM, and 0.0728 LPIPS. The current 20-scene aggregate is 43.886734,
0.989558, and 0.009261, but protocol comparability must still be stated in the
paper rather than inferred from the numbers alone.
