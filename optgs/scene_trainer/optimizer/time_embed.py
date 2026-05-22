# From: https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/time_utils.py

import torch
import torch.nn as nn


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 1,  # time steps are 1D
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class TimeEncodingWrapper:
    def __init__(self, use_time_encoding, time_encoder_fn, t, T, state):
        self.use_time_encoding = use_time_encoding
        self.T = T
        self.time_encoder_fn = time_encoder_fn
        self.state = state
        self.t = t

    def __enter__(self):
        # We are modifying the state only inside the context manager
        state = self.state
        if self.use_time_encoding:
            assert self.time_encoder_fn is not None, "Time encoder function must be defined."

            rel_step = torch.tensor([self.t / self.T], device=state.device)

            time_encoding = self.time_encoder_fn(rel_step)  # [embedding_dim]
            time_encoding = time_encoding.unsqueeze(0).repeat(state.shape[0], 1)  # [N, embedding_dim]

            # Concatenate encoding to state
            state = torch.cat([state, time_encoding], dim=-1)  # [N, c+embedding_dim]

        return state  # returns the modified state

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Do nothing, the original state is preserved outside the context manager
        # Return False to propagate exceptions, if any
        return False


if __name__ == "__main__":
    # Example usage
    embed_fn, output_dim = get_embedder(multires=6)
    print(f"Output embedding dimension: {output_dim}")
    steps = torch.randn(10, 1)  # Example input (steps normalized between 0 and 1)
    print(f"Input shape: {steps.shape}")
    print("steps[0:2]:", steps[0:2])
    embedded_x = embed_fn(steps)
    print(f"Embedded shape: {embedded_x.shape}")
    print("embedded_x[0:2]:", embedded_x[0:2])
