from ...dataset import DatasetCfg
from .decoder import Decoder
from .gsplat_decoder_splatting_cuda import GSplatDecoderSplattingCUDACfg, GSplatDecoderSplattingCUDA

DECODERS = {
    "gsplat": GSplatDecoderSplattingCUDA,
}

DecoderCfg = GSplatDecoderSplattingCUDACfg

# The inria decoder is optional (it needs diff_gaussian_rasterization).
# Importing this package must NOT require that backend — gsplat is the
# default. If the inria decoder is actually requested while the backend is
# missing, raise a clear, chained ImportError (mirrors the RoMa handling in
# optgs/experimental/edgs/init.py) instead of silently degrading.
try:
    from .decoder_splatting_cuda import DecoderSplattingCUDACfg, DecoderSplattingCUDA
    DECODERS["inria"] = DecoderSplattingCUDA
    DecoderCfg = GSplatDecoderSplattingCUDACfg | DecoderSplattingCUDACfg
except ImportError as _e:
    # `except ... as _e` is auto-deleted at block end; keep a stable ref so the
    # closure below can chain from the original error.
    _INRIA_IMPORT_ERROR = _e

    def _inria_decoder_unavailable(*_args, **_kwargs):
        raise ImportError(
            "The inria decoder requires diff_gaussian_rasterization, which is "
            "not installed. Install it with: "
            "pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
        ) from _INRIA_IMPORT_ERROR

    DECODERS["inria"] = _inria_decoder_unavailable


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    print(f"Using decoder: {decoder_cfg.name}")
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg)
