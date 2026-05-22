from dataclasses import dataclass

from optgs.scene_trainer.optimizer.optimizer_utils import Number3DGSCfg, Bool3DGSCfg


@dataclass
class SchedulerCfg:
    name: str
    lr_data: Number3DGSCfg
    apply_scheduler: Bool3DGSCfg


@dataclass
class DDIMSchedulerCfg(SchedulerCfg):
    T: int
    min_lr: float | int
    s: float | int


LrSchedulerCfgType = DDIMSchedulerCfg | SchedulerCfg


class Scheduler[SchedulerCfg]:
    def __init__(self, cfg: SchedulerCfg):
        self.cfg = cfg

    def get_lr(self, t: int, param: str) -> float | int:
        lr = getattr(self.cfg.lr_data, param)
        apply = getattr(self.cfg.apply_scheduler, param)
        if apply:
            return self.scheduler_fn(t, lr)
        else:
            return lr

    def scheduler_fn(self, t: int, base_lr: float | int) -> float | int:
        raise NotImplementedError


class DummyScheduler(Scheduler[SchedulerCfg]):
    def get_lr(self, t: int, param: str) -> float | int:
        return 1


class DDIMCosineScheduler(Scheduler[DDIMSchedulerCfg]):
    def scheduler_fn(self, t: int, base_lr: float | int) -> float | int:
        # Implement DDIM Cosine scheduling logic here
        import math
        t = min(t, self.cfg.T)
        rel_t = t / self.cfg.T
        alpha_bar = math.cos((rel_t + self.cfg.s) / (1 + self.cfg.s) * math.pi / 2) ** 2
        lr = self.cfg.min_lr + alpha_bar * (base_lr - self.cfg.min_lr)
        return lr


SCHEDULERS = {
    "none": Scheduler,
    "ddim": DDIMCosineScheduler,
}

def get_scheduler(cfg: SchedulerCfg) -> Scheduler:
    print(f"Using scheduler: {cfg.name}")
    scheduler_class = SCHEDULERS[cfg.name]
    scheduler = scheduler_class(cfg)
    return scheduler

if __name__ == "__main__":
    cfg = DDIMSchedulerCfg(
        name="DDIMCosineScheduler",
        lr_data=Number3DGSCfg(_base=1.0, _means=1.0, _scales=1.0, _opacities=1.0, _quats=1.0, _sh0=1.0, _shN=1.0),
        apply_scheduler=Bool3DGSCfg(_base=True, _means=True, _scales=True, _opacities=True, _quats=True, _sh0=True,
                                    _shN=True),
        T=24,
        min_lr=0.0,
        s=0.008
    )
    scheduler = DDIMCosineScheduler(cfg)

    iterations = list(range(0, 24))
    lr = [scheduler.get_lr(t, "means") for t in iterations]

    import matplotlib.pyplot as plt

    plt.plot(iterations, lr)
    plt.xlabel("Timestep")
    plt.ylabel("Learning Rate")
    plt.title("DDIM Cosine Learning Rate Schedule")
    plt.grid()
    plt.show()
