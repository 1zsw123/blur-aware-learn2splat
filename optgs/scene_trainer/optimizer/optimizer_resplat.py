from typing import Literal

from optgs.scene_trainer.optimizer import KnnBasedOptimizer, KnnBasedOptimizerCfg


class ResplatOptimizerV1(KnnBasedOptimizer):
    OPTIMIZER_NAME = "resplat_v1"

class ResplatOptimizerV2(KnnBasedOptimizer):
    OPTIMIZER_NAME = "resplat_v2"
