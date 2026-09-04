# PRISM3D Results

The accepted result is the uniform 10K, eight-scene dilation-2 repair run. It
uses TURTLE stage-1 step 24K, NIMA > 0.6 with w10, hold-identity-blind all-frame
optimization, a true 25x25 BPN kernel, latent blur assignment, and Laplacian
surplus weight 0.1.

The detailed artifacts already exist on Hugging Face under `accepted/10k/`.
This unified directory adds only a compact table, provenance, and links; it
does not duplicate or relocate the existing accepted artifacts.

The accepted aggregate is 37.794834 PSNR, 0.937011 SSIM, and 0.046374 LPIPS.
Although later 50K runs improve aggregate PSNR for some scenes, strong-blur
visual quality can regress, so the eight-scene main table remains uniformly
10K.
