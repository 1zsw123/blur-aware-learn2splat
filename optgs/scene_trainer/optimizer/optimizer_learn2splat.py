from optgs.scene_trainer.optimizer.optimizer_knn_based import KnnBasedOptimizer


class Learn2SplatOptimizer(KnnBasedOptimizer):
    OPTIMIZER_NAME = "l2s"
    OPTIMIZER_NAME_ALIASES = ("clogs",)  # TODO (release): remove aliases
