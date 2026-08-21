# TUM Protocol Audit

The benchmark adapter is bound to the public Unblur-SLAM repository at commit
`151ca3b3185b7c56beef3eeb0c0c9034feeeb843`.

Evidence:

- `scripts/fr2_xyz_indices.txt` contains 42 filtered-stream indices and has
  SHA-256 `492f657623c2856b998a4b6031f53a09855266957192b1dd5c60ea7b0471fd71`.
- `thirdparty/glorie_slam/motion_filter.py` has SHA-256
  `85a5b8029d219fb9bdcbe600fd82430029c237982a3be04cc642ef0be28a1641`.
  Lines 90-129 load the scene-specific index file. Lines 270-311 process every
  incoming frame but set `is_keyframe = tstamp in self.sharp_indices`; only a
  true keyframe is appended to the mapping video.
- `src/utils/datasets.py` has SHA-256
  `6a4700c0b73d07f6ffb32b44569c3d26ab2c0671bdc6f0a76153868d85b05f03`.
  Its TUM loader associates RGB, depth, and pose before applying the 32 Hz
  temporal filter.

Consequently, the benchmark reconstruction input is the released 42-keyframe
mapping set. Optimizing all 3,397 associated frames is a useful stress test but
is not a reproduction of the released mapping protocol. TUM RGB-D depth is
camera-axis z-depth; the full-stream diagnostic declares that convention
explicitly.
