import torch


def split_to_minibatch(batch_split, iter_context_idxs):
    minibatch = {
        "image": batch_split["image"][0][iter_context_idxs].unsqueeze(
            0
        ),  # [1, Vc', 3, Hc, Wc]
        "extrinsics": batch_split["extrinsics"][0][iter_context_idxs].unsqueeze(
            0
        ),  # [1, Vc', 4, 4]
        "intrinsics": batch_split["intrinsics"][0][iter_context_idxs].unsqueeze(
            0
        ),  # [1, Vc', 4, 4]
        "near": batch_split["near"][0][iter_context_idxs].unsqueeze(0),  # [1, Vc']
        "far": batch_split["far"][0][iter_context_idxs].unsqueeze(0),  # [1, Vc']
    }
    return minibatch


def batched_select(data, indices):
    """
    Select data[i, indices[i]] for each batch element i.

    Args:
        data: [B, N, ...] input tensor
        indices: [B, K] indices for each batch element

    """

    assert data.shape[0] == indices.shape[0], f"Batch size mismatch {data.shape[0]} vs {indices.shape[0]}"
    assert indices.dim() == 2, f"indices should be 2D, got {indices.shape}"

    B = data.shape[0]
    batch_idx = torch.arange(B, device=data.device)[:, None]
    return data[batch_idx, indices]
