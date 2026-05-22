"""
Debug script mimicking the learned optimizer training loop with stability loss.
Simulates the meta-training loop with inner iterations and the stability loss.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional


# ─────────────────────────────────────────────
# Minimal stubs mirroring your real classes
# ─────────────────────────────────────────────

@dataclass
class RenderOutput:
    color: torch.Tensor  # [B, V, C, H, W]


@dataclass
class OptimizerOutput:
    context_render_list: List[RenderOutput]
    target_render_list: List[RenderOutput]
    context_index_list: List[Optional[torch.Tensor]]  # list of [B, V] or empty
    target_index_list: List[Optional[torch.Tensor]]

    def get_render_list(self, input_str: str) -> List[RenderOutput]:
        return self.context_render_list if input_str == "context" else self.target_render_list

    def get_index_list(self, input_str: str) -> List[torch.Tensor]:
        lst = self.context_index_list if input_str == "context" else self.target_index_list
        return [x for x in lst if x is not None]


# ─────────────────────────────────────────────
# Tiny "optimizer network" that produces renders
# across inner iterations — all connected in graph
# ─────────────────────────────────────────────

class TinyOptimizerNet(nn.Module):
    """
    Simulates a learned optimizer that refines a rendering across I inner steps.
    Each step: render = prev_render + delta(prev_render, params)
    This creates a graph that chains across iterations, just like your real model.
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(3, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, 3, 1),
        )

    def forward(self, init_render: torch.Tensor, num_inner: int):
        """
        init_render: [B, V, C, H, W]
        Returns list of renders of length num_inner+1 (init + refined)
        """
        renders = [init_render.detach()]  # init is detached, like in your code
        curr = init_render.detach()
        B, V, C, H, W = curr.shape
        for _ in range(num_inner):
            flat = curr.view(B * V, C, H, W)
            delta = self.refine(flat)
            curr = curr + delta.view(B, V, C, H, W)  # connected graph across iters
            renders.append(curr)
        return renders


# ─────────────────────────────────────────────
# Stability loss (copied from your code)
# ─────────────────────────────────────────────

class LossStability(nn.Module):
    def forward(self, optimizer_output: OptimizerOutput, batch: dict) -> torch.Tensor:
        total_loss = 0
        for input_str in ["context", "target"]:
            render_list = optimizer_output.get_render_list(input_str)
            index_list = optimizer_output.get_index_list(input_str)

            predictions = [render.color for render in render_list]
            predictions = torch.stack(predictions, dim=0)  # [I, B, V, C, H, W]
            gt = batch[input_str]["image"]  # [B, V_all, C, H, W]

            if len(index_list) == 0:
                loss = torch.abs(predictions - gt).mean(dim=[3, 4, 5])  # [I, B, V]
                change_in_loss = loss[1:] - loss[:-1].detach()  # [I-1, B, V]
                change_in_loss = torch.relu(change_in_loss)
                total_loss = total_loss + change_in_loss.sum()
                print(f"  Stability loss ({input_str}): {total_loss.item():.6f}")
                continue

            # With index lists
            index_list_padded = [index_list[0]] + index_list  # I tensors
            index_list_t = torch.stack(index_list_padded, dim=0)  # [I, B, V]

            b = gt.shape[0]
            device = gt.device
            batch_idx = torch.arange(b, device=device)[None, :, None]
            gt_indexed = gt[batch_idx, index_list_t]  # [I, B, V, C, H, W]

            loss = torch.abs(predictions - gt_indexed).mean(dim=[3, 4, 5])  # [I, B, V]

            I, B, V_all = predictions.shape[0], gt.shape[0], gt.shape[1]
            loss_full = torch.zeros(I, B, V_all, device=device).scatter(2, index_list_t, loss)

            iter_idx = torch.arange(I, device=device).view(-1, 1, 1)
            visited = loss_full > 0
            visit_ids = torch.where(visited, iter_idx, torch.full_like(iter_idx, -1))
            last_visit = torch.cummax(visit_ids, dim=0).values
            prev_visit = torch.roll(last_visit, shifts=1, dims=0)
            prev_visit[0] = -1
            safe_prev = prev_visit.clamp(min=0)
            prev_loss = loss_full.gather(0, safe_prev).detach()
            has_prev = prev_visit >= 0
            change_in_loss = torch.relu(loss_full - prev_loss)
            change_in_loss = change_in_loss * has_prev.float().detach()
            total_loss = total_loss + change_in_loss.sum()
            print(f"  Stability loss ({input_str}): {change_in_loss.sum().item():.6f}")

        return total_loss


# ─────────────────────────────────────────────
# Other losses (L1 and a fake LPIPS)
# ─────────────────────────────────────────────

def compute_l1_loss(render_color, gt):
    return torch.abs(render_color - gt).mean()

def compute_lpips_loss(render_color, gt):
    # Fake LPIPS: just MSE on downsampled version
    return ((render_color - gt) ** 2).mean()

def compute_meta_losses(optimizer_output, batch, num_inner):
    """Mimics your _calc_opt_loss loop (without stability)."""
    opt_loss = 0
    for i in range(num_inner):
        pred = optimizer_output.context_render_list[i + 1].color
        gt = batch["context"]["image"]
        opt_loss = opt_loss + compute_l1_loss(pred, gt)
        opt_loss = opt_loss + 0.1 * compute_lpips_loss(pred, gt)

        pred = optimizer_output.target_render_list[i + 1].color
        gt = batch["target"]["image"]
        opt_loss = opt_loss + compute_l1_loss(pred, gt)
        opt_loss = opt_loss + 0.1 * compute_lpips_loss(pred, gt)
    return opt_loss


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def make_batch(B=2, V=3, C=3, H=8, W=8, device="cpu"):
    return {
        "context": {"image": torch.rand(B, V, C, H, W, device=device)},
        "target":  {"image": torch.rand(B, V, C, H, W, device=device)},
    }


def run_meta_iteration(net, batch, stability_loss_fn, num_inner=5, use_index_list=False):
    B, V, C, H, W = batch["context"]["image"].shape
    device = batch["context"]["image"].device

    # Simulate init render (detached from network, like 3DGS init)
    init_context = torch.rand(B, V, C, H, W, requires_grad=False, device=device)
    init_target  = torch.rand(B, V, C, H, W, requires_grad=False, device=device)

    context_renders = net(init_context, num_inner)
    target_renders  = net(init_target,  num_inner)

    context_render_list = [RenderOutput(color=r) for r in context_renders]
    target_render_list  = [RenderOutput(color=r) for r in target_renders]

    if use_index_list:
        # Simulate partial view sampling: pick V//2 views each inner iter
        num_views = max(1, V // 2)
        context_index_list = [
            torch.randint(0, V, (B, num_views), device=device)
            for _ in range(num_inner)
        ]
        target_index_list = [
            torch.randint(0, V, (B, num_views), device=device)
            for _ in range(num_inner)
        ]
    else:
        context_index_list = [None] * num_inner
        target_index_list  = [None] * num_inner

    optimizer_output = OptimizerOutput(
        context_render_list=context_render_list,
        target_render_list=target_render_list,
        context_index_list=context_index_list,
        target_index_list=target_index_list,
    )

    # ── Meta losses (L1 + LPIPS across inner iters) ──
    meta_loss = compute_meta_losses(optimizer_output, batch, num_inner)

    # ── Stability loss ──
    stab_loss = stability_loss_fn(optimizer_output, batch)

    total_loss = meta_loss + 0.01 * stab_loss

    return total_loss


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = TinyOptimizerNet(hidden=16).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    stability_loss_fn = LossStability()

    NUM_META_STEPS = 5
    NUM_INNER = 5  # inner iterations (like your 6, minus init)

    for mode in ["no_index_list", "with_index_list"]:
        print(f"\n{'='*50}")
        print(f"Mode: {mode}")
        print(f"{'='*50}")
        use_index = (mode == "with_index_list")

        for step in range(NUM_META_STEPS):
            batch = make_batch(device=device)
            optimizer.zero_grad()

            try:
                total_loss = run_meta_iteration(
                    net, batch, stability_loss_fn,
                    num_inner=NUM_INNER,
                    use_index_list=use_index
                )
                total_loss.backward()
                optimizer.step()
                print(f"  Step {step+1}: total_loss={total_loss.item():.6f} ✓")
            except RuntimeError as e:
                print(f"  Step {step+1}: ERROR - {e}")
                break


if __name__ == "__main__":
    main()