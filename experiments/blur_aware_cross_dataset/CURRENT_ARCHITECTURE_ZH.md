# 当前可复现架构

本文档描述 `agent/blur-aware-cross-dataset` 分支当前已经落地并通过三域
smoke test 的实现。它区分当前主候选、稳定回退版本和历史探索，避免把旧
Triangle、全局规则控制器和 exact LeGS 混写成同一个方法。

## 1. 一句话定义

当前主候选是：

> **Learn2Splat Gaussian 表示与优化 + Blur-aware BPN/Laplacian surplus
> 监督 + exact LeGS 逐基元 PPO 容量控制。**

它不是 Triangle Splatting，也不是此前手写阈值的 direct RL。Learn2Splat
负责优化 Gaussian 参数，BPN 与 surplus loss 负责处理模糊监督，LeGS 只
负责决定每个 Gaussian 的 `keep / clone / split / prune`。

```text
RAW blurry / authoritative sharp images
EVSSM deblurred images
camera intrinsics + poses + trusted depth/SfM points
NIMA > 0.6 sharp manifest + benchmark split
                    |
                    v
        Cross-dataset protocol adapter
        (Motion / Defocus / TUM only provide data and split)
                    |
                    v
       depth/SfM-fused Gaussian initialization (up to 70K)
                    |
                    v
       Learn2Splat Dense proposal optimizer (0--2K)
                    |
                    +-------------------------------+
                    |                               |
                    v                               v
          FastGS differentiable renderer    exact LeGS controller
                    |                       11-D state + PPO policy
                    v                       keep/clone/split/prune
      Blur-aware reconstruction objective           |
      BPN + EVSSM confidence + surplus loss <-------+
                    |
                    v
       objective-consistent Adam projection (>2K)
                    |
                    v
        authoritative RAW hold/test evaluation
        receipt + metrics + grids + controller log
```

## 2. 数据与评测口径

三类数据使用同一个训练程序，场景配置只能提供路径、相机、深度约定和冻结
split，优化器和容量策略不会读取数据集名称。

- **Deblur-NeRF Motion/Defocus**：训练使用全部输入视角；评测使用冻结的
  authoritative sharp/hold manifest，不用 EVSSM 当 GT。
- **I2-SLAM/Unblur-SLAM TUM**：使用官方 mapping video 对应的 42 个
  `fr2_xyz` keyframes；这些索引是过滤后视频的位置，不是原始 TUM 文件名。
- **sharp 训练监督**：NIMA `> 0.6` 的帧进入 sharp pool，以加权公平调度实现
  w10 曝光概率；不会在 loss 中再次乘一次 w10。
- **评测隔离**：LeGS 状态相机、固定训练 probe 和 BPN 都只能读取训练视图，
  最终 hold/test 不参与策略状态、reward 或超参数选择。
- **FrameCrafter**：保留为诊断消融，不属于当前默认输入。现有五帧增强没有
  改善 TUM 主结果，因此不会静默加入主实验。

## 3. 表示与优化器

当前表示是 Learn2Splat 的 **3D Gaussian**，不是 Triangle primitive。

1. 用可信 SfM 点和可用 sensor/sparse depth 初始化，最多使用 Dense checkpoint
   声明的 70K 初始化点；缺失深度时不会伪造深度。
2. `0--2K` 使用官方 Learn2Splat Dense learned optimizer 生成 scene proposal。
3. 超过官方 2K learned horizon 后，保留当前 Gaussian、BPN、sampler 和控制器
   状态，切换到对同一目标函数求解的 Adam residual projection。
4. 只有 Learn2Splat 的 recurrent latent/gradient-normalizer 在 2K 边界按官方
   horizon 重启；场景本身不会从头训练。

因此，`learned_projected` 的含义是“Learn2Splat 提案 + 同目标 Adam 收敛”，
不是把 Learn2Splat 替换回普通 3DGS。

## 4. Blur-aware 监督

### 4.1 静态 EVSSM 可靠度

对每个 RAW/EVSSM 图像对，先计算与表示无关的静态置信度：

```text
fidelity = (color_fidelity * edge_agreement * clipping_safety)^(1/3)
gain     = tanh(max(log(E_lap(EVSSM) / E_lap(RAW)), 0))
c_static = c_min + (1 - c_min) * fidelity * (0.25 + 0.75 * gain)
```

默认 `c_min=0.1`。颜色项检查低频内容是否被改坏，边缘项检查方向是否一致，
clipping 项抑制新产生的黑白饱和区域，Laplacian 项只判断 EVSSM 是否增加了
有用锐度。已知 sharp 帧的置信度直接设为 1。该分数不读取重建 PSNR、数据集
标签或 hold/test。

### 4.2 Factorized BPN

BPN 不输出可记忆纹理的逐像素 kernel，而是：

- 每个训练相机一个正值、归一化的 `9 x 9` kernel；
- 一个共享的低分辨率 mask 网络，输入 RAW、EVSSM、二者残差和可用 depth；
- 图像形成模型：

```text
formed = mask * Kernel(Render) + (1 - mask) * Render
```

kernel center、mask target 和 mask TV 只作为轻量正则。全 sharp batch 会完全
bypass BPN，避免无定义的模糊模型污染直接监督。

### 4.3 重建与 Laplacian surplus

非直接监督帧在 EVSSM target 与 RAW image-formation loss 之间软切换：

```text
w_raw = (1 - direct) * (1 - c_effective) * raw_ramp(t)
L_rec = (1 - w_raw) * L_rgb_ssim(Render, EVSSM)
        + w_raw * L_rgb_ssim(formed, RAW)
```

当前主候选使用三尺度 signed-Laplacian surplus，而不是只把 Render 锁死到
EVSSM：

- 在 RAW/EVSSM 确认有结构的位置，把 EVSSM 边缘强度作为单边 floor；
- 允许 Render 在多视角一致支持下超过 EVSSM；
- 可靠 teacher 的过冲只被软约束；
- 平坦、无证据区域的新高频会进入 artifact penalty；
- render-over-EVSSM surplus 必须在逐视图 EMA 中稳定，并形成场景共识，才会
  降低 teacher 的动态置信度。瞬时单帧提升不能给自己降权。

总目标为：

```text
L_total = L_rec + 0.1 * L_lap_surplus + raw_ramp(t) * L_BPN_reg
```

所有动态置信度均从图像 loss graph detach。精确的三尺度 floor、overshoot、
artifact 和 EMA 定义见 `objective.py` 与 `LAPLACIAN_ABLATION.md`。

## 5. Exact LeGS 容量控制

`--adc legs` 是官方 LeGS **容量机制**在 Learn2Splat runtime 中的精确移植：

- 上游固定到 LeGS commit
  `8eb120b1f0c0fe0727e0440f4e372b412f275572`；
- 两个比较臂都使用 FastGS renderer，避免把 renderer 差异算成 controller
  差异；
- 每次决策随机选 10 个训练相机；
- 每个 Gaussian 的 11-D state 为 XYZ gradient 3 维、scale gradient 3 维、
  opacity gradient 1 维、DC color gradient 3 维、官方 FastGS leave-one-out
  L1 sensitivity 1 维；
- actor 输出 `keep / clone / split`，低 opacity Gaussian 由独立 prune estimator
  决定保留或删除；
- 动作执行 50 step 后，用相同相机重新计算 sensitivity，按 parent-child mapping
  将子 Gaussian 的收益归还给父 Gaussian；
- 两条 transition 后执行 GAE/PPO，两轮 epoch，500K-point chunk；
- 使用官方 `start=500, every=100, stop=15000` 调度，3K opacity reset，15K
  后周期性 final opacity prune；
- exact 模式没有全局 primitive cap。代码中存在的 `cap_max` 只用于显式 adapted
  safety-cap 配置，当前 exact receipt 记录为 `null`。

必须注意：exact LeGS 的 reward 是官方逐基元 sensitivity 变化，不是旧版全局
PSNR/Laplacian probe reward。因此当前组合的 blur awareness 来自监督目标，LeGS
负责学习结构动作；不能把它写成“Laplacian reward 版 LeGS”。

### 5.1 Blur-conditioned LeGS 实验分支

`--adc legs_blur` 在不改动 `--adc legs` 回退路径的前提下，把模糊恢复需求真正
接入 LeGS policy。它保留官方 11-D 逐基元状态、FastGS sensitivity、PPO、
parent-child credit 和 prune estimator，并为每个 Gaussian 追加同一时刻的 7-D
场景状态：

```text
[EVSSM reliability mean/std,
 render-over-EVSSM Laplacian surplus,
 BPN kernel entropy/radius, BPN mask strength,
 primitive pressure]
```

这些量先按各自物理范围变成无量纲的 `[-1, 1]` 特征，而不是套用某个数据集的
PSNR、kernel 或 primitive 阈值。策略 reward 在动作后延迟 50 steps 计算：

```text
r = r_sensitivity
    + w_q * (confidence-weighted multi-view PSNR/surplus improvement)
    - w_c * max(0, relative net primitive growth)
```

质量项始终读取同一组 8 个 farthest training probes；最终 hold/test 视图不会进入
状态或 reward。actor 因而能结合局部 sensitivity 与全局模糊恢复状态选择
`keep/clone/split`，低 opacity 的 `prune` estimator 也读取相同的 18-D 状态。

三域 3K matched smoke 的平均 PSNR 比 exact LeGS 低 `0.507 dB`，总 primitive
数少 `13.5%`。因此它已经是正确接线、可复现的机制消融，但目前只证明了容量
压缩，不证明质量提升，也尚未替代本节的 exact LeGS 主候选。完整表、协议和
产物见 `BLUR_CONDITIONED_LEGS_SMOKE_ZH.md`。

## 6. 当前三域 matched smoke

固定条件：seed `20260822`、10K、每 1K 评测、`learned_projected`、FastGS、
surplus weight `0.1`、同一数据/split/初始化。对照是当前
`adaptive + surplus_probe`，实验臂只换成 exact LeGS 原生 policy/reward。

| 域 / 场景 | Adaptive best | Exact LeGS best | best 差值 | 10K 差值 | 10K primitive 倍数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Motion / blurcoffee | 39.754 @8K | **46.003 @9K** | **+6.250 dB** | **+6.357 dB** | 6.37x |
| Defocus / cisco | 32.798 @10K | **34.595 @9K** | **+1.797 dB** | **+1.485 dB** | 6.39x |
| TUM / fr2_xyz | 25.116 @10K | **26.634 @9K** | **+1.518 dB** | **+1.265 dB** | 8.76x |

三域均提升，说明 exact LeGS 可以在当前 blur-aware Learn2Splat 中工作；但这还
不是最终论文总表：

- 目前只有每域一个代表场景、单 seed、10K；
- smoke 为提速跳过了 LPIPS；
- 三个 exact run 都在 9K 达峰，10K 有轻微回落；
- primitive 数增加约 6.4--8.8 倍，TUM 可视化仍有少量白色散点。

因此当前结论是“机制 smoke 成功”，不是“容量效率和全量泛化已解决”。下一步
需要 50K、多场景、LPIPS、训练时间/显存/primitive 数与 early-stop/cap 消融。

## 7. 运行与回退

### 当前 exact LeGS 主候选

```bash
CUDA_VISIBLE_DEVICES=2 "$ENV" \
  experiments/blur_aware_cross_dataset/run_cross_dataset.py \
  --scene motion_blurcoffee --output-root "$OUT/exact_legs" \
  --steps 10000 \
  --eval-steps 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 \
  --objective blur-aware --optimizer learned_projected \
  --adc legs --decoder-backend fastgs --densification-reward off \
  --laplacian-loss-mode surplus --laplacian-loss-weight 0.1
```

### Blur-conditioned LeGS 实验臂

```text
--adc legs_blur --decoder-backend fastgs
--densification-reward off
--laplacian-loss-mode surplus --laplacian-loss-weight 0.1
```

### 稳定 adaptive 回退

```text
--adc adaptive --decoder-backend fastgs
--densification-reward surplus_probe
--laplacian-loss-mode surplus --laplacian-loss-weight 0.1
```

### 其他消融

- 无容量控制：`--adc none`
- 无 Laplacian：`--laplacian-loss-weight 0`
- 固定 probe 但不把 reward 喂给 ADC：`--densification-reward probe_control`
- 旧 Triangle pipeline：使用原 Triangle 仓库与 launcher；本分支没有修改它。

所有模式通过命令行切换，不需要回滚源码。exact LeGS 与 adaptive 的 checkpoint
不能假定控制器状态兼容，恢复实验时必须保持原 controller 配置。

## 8. 代码入口

| 组件 | 文件 |
| --- | --- |
| 三域 runner、协议门禁、receipt | `run_cross_dataset.py` |
| 场景路径与冻结 split | `scenes.json`、`protocols.py` |
| EVSSM 静态可靠度 | `optgs/experimental/blur_aware/reliability.py` |
| BPN、重建 loss、surplus loss | `optgs/experimental/blur_aware/objective.py` |
| exact LeGS policy/reward/action | `optgs/scene_trainer/adc/legs.py` |
| exact LeGS 配置 | `optgs/scene_trainer/adc/legs_config.py` |
| blur-conditioned LeGS 配置 | `optgs/config/scene_trainer/scene_optimizer/refiner/legs_blur.yaml` |
| blur-conditioned 三域 smoke | `BLUR_CONDITIONED_LEGS_SMOKE_ZH.md` |
| FastGS sensitivity 接口 | `optgs/scene_trainer/adc/fastgs.py` |
| 三域 matched 汇总 | `summarize_adc_pairs.py` |
| 完整 Laplacian 消融 | `LAPLACIAN_ABLATION.md` |
| TUM/Unblur-SLAM 协议审计 | `UPSTREAM_PROTOCOL_AUDIT.md` |

本架构说明只承诺仓库中已有代码与 receipt 能证明的内容。历史 direct RL、
Transformer/TTT 参数预测和 FrameCrafter 思路没有被伪装成当前主方法。
