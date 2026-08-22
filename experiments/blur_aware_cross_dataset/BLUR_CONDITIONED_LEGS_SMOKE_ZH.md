# Blur-conditioned LeGS matched smoke

日期：2026-08-23

## 方法边界

`--adc legs` 仍是未修改的官方 LeGS 容量机制。实验模式
`--adc legs_blur` 保留其 11-D 逐基元状态、FastGS sensitivity、PPO、
parent-child credit、`keep/clone/split` actor 和 prune estimator，并加入 7-D
场景级模糊状态：

```text
[EVSSM reliability mean/std,
 render-over-EVSSM Laplacian surplus,
 BPN kernel entropy/radius, BPN mask strength,
 primitive pressure]
```

状态只读取固定训练 probe，不读取 hold/test。动作后 50 steps，在同一组 8 个
训练 probe 上测量 confidence-weighted PSNR 与 Laplacian surplus 的变化。全局
质量信号按本次动作的净扩张方向，给 birth 与 prune 相反符号的 credit；逐基元
官方 sensitivity 的 sigmoid 作为局部 support gate，未被局部证据支持的 birth
额外承担相对净容量成本。

模糊分支在 2K 前为零，2K--5K 线性引入。2026-08-23 的审计发现旧 adapter
带可学习 bias，导致输入虽为零但 2K 前 bias 仍被 PPO 更新。当前版本移除了
该无条件 bias，并有回归测试保证 zero-state optimizer update 后仍保持 exact
LeGS 表示。`--adc legs` 回退路径未改变。

## 协议

- 场景：`motion_blurcoffee`、`defocus_cisco`、`tum_fr2_xyz`
- seed：`20260822`
- 训练：10K steps，每 1K 评测
- 共同配置：Learn2Splat `learned_projected`、FastGS、blur-aware objective、
  Laplacian surplus weight `0.1`
- conditioned 参数：quality weight `1.0`、capacity weight `0.10`
- LPIPS：smoke 中跳过
- TUM exact：在当前提交上与 conditioned 同时 fresh 重跑；motion/defocus 使用
  同 seed、同协议且 exact 路径未变的既有冻结 receipt

## 结果

| scene | exact best | conditioned best | delta | exact 10K | conditioned 10K | delta | exact N | conditioned N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| motion_blurcoffee | 46.003 @9K | 45.993 @9K | -0.011 | 45.391 | 45.389 | -0.002 | 289,251 | 285,165 (-1.4%) |
| defocus_cisco | 34.595 @9K | 34.528 @9K | -0.067 | 34.283 | 34.356 | +0.073 | 492,593 | 477,741 (-3.0%) |
| tum_fr2_xyz | 26.550 @9K | 26.608 @9K | +0.057 | 26.396 | 26.457 | +0.061 | 805,431 | 810,784 (+0.7%) |
| average / total N | 35.716 | 35.709 | -0.007 | 35.357 | 35.401 | +0.044 | 1,587,275 | 1,573,690 (-0.9%) |

TUM conditioned 在 8K 曾从约 26.36 dB 降至 25.30 dB，可视化出现黄色拉长
基元，9K 恢复到 26.61 dB。该异常没有被平均值隐藏，说明在线 policy 已能恢复，
但单次结构动作仍可能振荡。

## 结论

无偏 warmup 修复后，blur-conditioned LeGS 不再像早期 3K 版本那样系统性损失
质量。三个域的固定 10K 平均提高 `0.044 dB`，总基元减少 `0.9%`；best 平均
基本持平（`-0.007 dB`）。这支持“模糊状态、全局锐化 reward 与逐基元 LeGS
动作已经正确协同并可跨三域运行”，但单场景单 seed 的差值太小，不能宣称显著
优于 exact LeGS，也不能据此替代论文主结果。下一阶段需要多 seed、全量场景、
LPIPS、运行时间及 TUM 瞬时振荡消融。

## 产物

- conditioned receipts：
  `/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_blur_v7_warmupfix_crossdomain10k_s1`
- fresh TUM exact receipt：
  `/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_exact_current8416a8e_tum10k_s1`
- 汇总 CSV、逐视角表和曲线：
  `/srv2/szha0669/blur_slam_exp/outputs/learn2splat_legs_blur_v7_warmupfix_summary_s1`
- controller receipt version：`blur_conditioned_legs_v6_unbiased_warmup`
- 代码提交：`8416a8e`
