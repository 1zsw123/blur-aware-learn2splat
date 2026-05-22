from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from optgs.dataset.data_types import BatchedExample
from optgs.loss import Loss
from optgs.scene_trainer.optimizer.optimizer import OptimizerOutput


@dataclass
class LossStabilityCfg:
    weight: float | int


@dataclass
class LossStabilityCfgWrapper:
    stability: LossStabilityCfg


class LossStability(Loss[LossStabilityCfg, LossStabilityCfgWrapper]):
    def forward(
            self,
            optimizer_output: OptimizerOutput,
            batch: BatchedExample,
            **kwargs,
    ) -> Float[Tensor, ""]:
        total_loss = torch.tensor(0.0, device=optimizer_output.get_render_list("context")[0].color.device)
        # Stability loss: encourage the model to produce similar outputs for the same input across iterations.
        for input_str in ["context", "target"]:
            render_list = optimizer_output.get_render_list(input_str)
            index_list = optimizer_output.get_index_list(input_str)  # list of I-1 tensors of shape [B, V]

            if len(index_list) == 0:
                predictions = [render.color for render in render_list]
                predictions = torch.stack(predictions, dim=0)  # [I, B, V, C, H, W]
                gt = batch[input_str]["image"]  # [B, V_all, C, H, W]

                # V == V_all
                # Compute l1 loss between predictions and gt for each iteration
                loss = torch.abs(predictions - gt).mean(dim=[3, 4, 5])  # [I, B, V]
                change_in_loss = loss[1:] - loss[:-1].detach()  # [I-1, B, V]
                change_in_loss = torch.relu(change_in_loss)  # Only consider increases in loss as contributing to the stability loss
            else:
                continue

                # Duplicate the first index for the initialization
                index_list = [index_list[0]] + index_list  # Now we have I tensors of shape [B, V]
                index_list = torch.stack(index_list, dim=0)  # [I-1, B, V]

                b = gt.shape[0]
                device = gt.device
                batch_idx = torch.arange(b, device=device)[None, :, None]  # [1, B, 1]
                gt_indexed = gt[batch_idx, index_list]  # [I, B, V, C, H, W]

                # Compute l1 loss between predictions and gt for each iteration
                # Consider the the indexing of the views within the full batch
                loss = torch.abs(predictions - gt_indexed).mean(dim=[3, 4, 5])  # [I, B, V]

                # We want to make sure that the loss decreases across iterations for specific views
                I, B, V_all = predictions.shape[0], gt.shape[0], gt.shape[1]

                # Scatter losses into full view space
                # Don't use scatter_ in-place to enable backpropagation through the loss values
                loss_full = torch.zeros(I, B, V_all, device=loss.device).scatter(2, index_list, loss)  # [I, B, V_all]

                iter_idx = torch.arange(I, device=device).view(-1, 1, 1)  # [I,1,1]

                # mark unvisited as -1
                visited = loss_full > 0  # [I, B, V_all]
                visit_ids = torch.where(visited, iter_idx, torch.full_like(iter_idx, -1))  # [I, B, V]

                # running max gives last visit index
                last_visit = torch.cummax(visit_ids, dim=0).values  # [I,B,V]

                # shift to get strictly previous visit
                prev_visit = torch.roll(last_visit, shifts=1, dims=0)
                prev_visit[0] = -1  # first iter has no previous

                safe_prev = prev_visit.clamp(min=0)

                prev_loss = loss_full.gather(0, safe_prev).detach()

                has_prev = prev_visit >= 0

                change_in_loss = torch.relu(loss_full - prev_loss)
                change_in_loss = change_in_loss * has_prev.detach()


                # # Create a mask to identify views that have been visited in previous iterations (cumulative OR)
                # # Calculate the
                # visited = loss_full > 0  # [I, B, V_all]
                #
                # # Calcaulate the last visited index for each view
                # # Indices along I dimension: shape [I, 1, 1], broadcast over B and v_all
                # indices = torch.arange(I, device=visited.device).view(-1, 1, 1).expand_as(visited)  # [I, 1, 1] -> [I, B, V_all]
                # indices = indices.clone()
                # indices[visited == 0] = 0
                # prev_visit_idx = torch.cummax(indices, dim=0).values - 1  # [I, B, V_all]
                # # valid previous visit exists
                # has_prev = prev_visit_idx >= 0
                # prev_visit_idx = torch.clamp(prev_visit_idx, min=0)  # Ensure indices are non-negative
                #
                # # Loss from the previous visit for each view at each iteration (starting from the second iteration)
                # prev_loss = loss_full.detach().gather(0, prev_visit_idx)[1:]  # [I-1, B, V_all]
                #
                # curr_loss = loss_full[1:]  # [I-1, B, V_all], current loss for each view at each iteration
                #
                # change_in_loss = curr_loss - prev_loss  # [I-1, B, V_all]
                # change_in_loss = torch.relu(change_in_loss)  # Only consider increases in loss as contributing to the stability loss
                #
                # # Valid comparison mask:
                # #   - current iter visited
                # #   - previous visit index is strictly smaller (i.e. a real previous visit exists)
                # mask = visited[1:] & has_prev[:-1]  # [I-1, B, V_all]
                # change_in_loss = change_in_loss * mask.float().detach()  # Zero out change_in_loss for views that haven't been visited in both iterations
                #


            # # Fill in the loss values for the previous visits
            # loss_full_filled = loss_full.gather(0, prev_visit_idx)  # [I, B, V_all], now loss_full[i] contains the loss from the previous visit for each view
            #
            # # Update visited
            # visited_filled = loss_full > 0  # [I, B, V_all], now visited[i] is True for all views visited up to iteration i
            #
            # # Now compute change_in_loss across consecutive iterations
            # change_in_loss = loss_full_filled[1:] - loss_full_filled[:-1].detach()  # [I-1, B, V_all]
            #
            # # Mask change_in_loss to only consider views that have been visited in previous iterations (i.e., views that have a valid loss comparison)
            # # Detach the mask to prevent gradients from flowing through it
            # mask = visited_filled[1:] & visited_filled[:-1]  # [I-1, B, V_all], True for views that have been visited in both iterations being compared
            # mask = mask.detach()
            # change_in_loss = change_in_loss * mask.float()  # [I-1, B, V_all], zero out change_in_loss for views that haven't been visited in both iterations
            #
            # # Apply ReLU to only penalize increases in loss
            # change_in_loss = torch.relu(change_in_loss)  # [I-1, B, V_all], only positive change_in_loss contribute to the loss

            # loss
            total_loss += change_in_loss.sum()
        return total_loss * self.cfg.weight
