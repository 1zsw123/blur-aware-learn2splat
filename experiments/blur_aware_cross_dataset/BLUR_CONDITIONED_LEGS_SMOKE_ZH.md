# Blur-conditioned LeGS smoke 记录

日期：2026-08-22

## 方法边界

`--adc legs` 保留原始 LeGS 的 11 维状态、PPO 和动作逻辑；新方法仅由
`--adc legs_blur` 启用。新控制器在原始逐基元 sensitivity 状态后加入：

1. EVSSM reliability 的均值和离散度；
2. render-over-EVSSM Laplacian surplus；
3. BPN kernel entropy、kernel radius 和 mask strength；
4. 相对初始基元数的 primitive pressure。

所有状态采用有物理范围的无量纲映射到 `[-1, 1]`。EVSSM reliability
等场景常量不会再被时间 z-score 消成零。质量 reward 使用同一组 8 个
farthest training probes，在动作前后 50 步比较 confidence-weighted PSNR
和 Laplacian surplus；容量成本为相对净增长 `max(0, Delta N / N)`。
官方逐基元 sensitivity、parent-child credit、PPO、clone/split 以及单独的
prune estimator 均保留。

## Smoke 协议

- 场景：`motion_blurcoffee`、`defocus_cisco`、`tum_fr2_xyz`
- seed：`20260822`
- 训练：3K steps；1K/2K/3K 评测
- 共同配置：Learn2Splat `learned_projected`、FastGS、blur-aware objective、
  Laplacian surplus weight 0.1
- 对照：同 seed、同数据和同评测帧的 exact `--adc legs`

## 结果

| scene | exact 3K PSNR | legs_blur v2 3K PSNR | delta | exact SSIM | v2 SSIM | exact N | v2 N | N delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| motion_blurcoffee | 41.722 | 40.686 | -1.036 | 0.9880 | 0.9858 | 130,486 | 119,316 | -8.6% |
| defocus_cisco | 31.261 | 31.055 | -0.206 | 0.9549 | 0.9522 | 281,236 | 274,894 | -2.3% |
| tum_fr2_xyz | 24.954 | 24.673 | -0.281 | 0.8362 | 0.8218 | 399,169 | 307,023 | -23.1% |
| average / total N | 32.646 | 32.138 | -0.507 | 0.9264 | 0.9199 | 810,891 | 701,233 | -13.5% |

平均 PSNR 差在 1K 为 -0.019 dB，在 2K 为 -0.094 dB，在 3K 为
-0.507 dB。可视化未发现黑块、大面积错误或 NaN。

## 结论

工程接入通过，且 policy 的动作确实受到 blur state/reward 影响；三种域上
都出现了容量控制。当前 3K 证据只支持“接近质量下减少容量”，不支持
“PSNR 已超过 exact LeGS”。因此不能据此启动全量主表或宣称指标提升；
下一步应先在一个 late-training 场景跑到 10K/20K，判断少量早期容量是否
在后期恢复，随后再决定是否做全量。

## 产物

- 首版输出：`outputs/learn2splat_legs_blur_smoke3_3k_s1`
- 修正版输出：`outputs/learn2splat_legs_blur_smoke3_3k_s2`
- 修正版日志：`outputs/logs/learn2splat_legs_blur_smoke3_3k_s2`
- 运行门：`outputs/learn2splat_legs_blur_runtime_gate801_s4`

