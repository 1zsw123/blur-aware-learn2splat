import torch
import torch.nn as nn

from .layer import QK_Norm_TransformerBlock, init_weights


class LVSMTransformer(nn.Module):
    def __init__(self,
        dim=768,
        d_head=64,
        n_layer=24,
        special_init=True,
        depth_init=True,
        use_qk_norm=True,
        ):
        super().__init__()

        # Create transformer blocks
        self.transformer_blocks = [
            QK_Norm_TransformerBlock(
                dim, d_head, use_qk_norm=use_qk_norm
            ) for _ in range(n_layer)
        ]

        # Apply special initialization if configured
        if special_init:
            for idx, block in enumerate(self.transformer_blocks):
                if depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            for block in self.transformer_blocks:
                block.apply(init_weights)
                
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)


    def forward(self, x):

        for blk in self.transformer_blocks:
            x = blk(x)
        
        return x



if __name__ == '__main__':
    device = torch.device('cuda')
    model = LVSMTransformer().to(device)

    x = torch.randn(2, 64, 768).to(device)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = model(x)

    print(y.shape)




