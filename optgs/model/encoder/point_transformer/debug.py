import torch
import time


from layer import KNNAttention, TransformerBlock, PlainPointTransformer, SubsampleBlock


device = torch.device('cuda')

def test():
    bs = 4
    npts = 1024
    len_xyz = 3
    feat_dims = 64
    num_classes = 23
    coord = torch.rand(bs * npts, len_xyz).cuda()
    feat = torch.rand(bs * npts, feat_dims).cuda()
    offset = [npts * i for i in range(1, bs + 1)]
    offset = torch.tensor(offset).cuda()

    # data_dict = dict(
    #     coord = coord,
    #     feat = feat,
    #     offset = offset
    # )

    # model = PointTransformerSeg26().cuda()

    # model = KNNAttention(feat_dims, num_samples=16).cuda()

    # model = TransformerBlock(feat_dims).cuda()

    # model = PlainPointTransformer(feat_dims, num_blocks=2).cuda()
    model = SubsampleBlock(feat_dims, feat_dims).cuda()
    
    print(model)


    # count time
    # count = 100

    # torch.cuda.synchronize()
    # start = time.time()

    # for _ in range(count):
    #     out = model((coord, feat, offset))

    # torch.cuda.synchronize()
    # print(time.time() - start)


    out = model((coord, feat, offset))
    print(out[0].shape)

    # print(out.shape)



test()
