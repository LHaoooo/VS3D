import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VS3DConfig:
    stage1_steps: int = 25
    rasi_enabled: bool = True
    rasi_probe_variant: str = "A1"
    pmg_enabled: bool = True
    pmg_num_samples: int = 5
    pmg_partial_samples: int = 2
    pmg_weight: float = 1.2
    pmg_mode: str = "consistency"
    pmg_consistency_tau: float | None = 0.9767255187034607
    pmg_crossfit_kappa_2: float | None = None
    pmg_crossfit_kappa_3: float | None = None
    shape_mode: str = "tar"
    texture_mode: str = "tar"
    tar_lambda: float = 0.5
    tar_clip_tau: float = 10.0
    tar_keep_threshold: float = 0.7
    tar_quantiles: tuple[float, float] = (0.05, 0.95)

    def __post_init__(self) -> None:
        if self.stage1_steps <= 0:
            raise ValueError("stage1_steps must be positive")
        if self.rasi_probe_variant not in {"A1", "A0"}:
            raise ValueError("rasi_probe_variant must be 'A1' or 'A0'")
        if self.pmg_num_samples < 2:
            raise ValueError("pmg_num_samples must be at least 2")
        if self.pmg_partial_samples <= 0:
            raise ValueError("pmg_partial_samples must be positive")
        if self.pmg_partial_samples >= self.pmg_num_samples:
            raise ValueError(
                "pmg_partial_samples must be less than pmg_num_samples"
            )
        if not math.isfinite(self.pmg_weight):
            raise ValueError("pmg_weight must be finite")
        if self.pmg_mode not in {"original", "consistency", "crossfit"}:
            raise ValueError(
                "pmg_mode must be 'original', 'consistency', or 'crossfit'"
            )
        if self.pmg_consistency_tau is not None and (
            not math.isfinite(self.pmg_consistency_tau)
            or not -1.0 <= self.pmg_consistency_tau <= 1.0
        ):
            raise ValueError("pmg_consistency_tau must be finite and in [-1, 1]")
        for name, value in (
            ("pmg_crossfit_kappa_2", self.pmg_crossfit_kappa_2),
            ("pmg_crossfit_kappa_3", self.pmg_crossfit_kappa_3),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.pmg_enabled and self.pmg_mode != "original" and self.pmg_weight < 0.0:
            raise ValueError("gated PMG weight must be non-negative")
        if self.pmg_enabled and self.pmg_mode == "consistency":
            if self.pmg_consistency_tau is None:
                raise ValueError(
                    "pmg_consistency_tau is required for consistency PMG"
                )
        if self.pmg_enabled and self.pmg_mode == "crossfit":
            if self.pmg_num_samples != 5 or self.pmg_partial_samples != 2:
                raise ValueError(
                    "Cross-Fit PMG requires pmg_num_samples=5 and "
                    "pmg_partial_samples=2"
                )
            if (
                self.pmg_crossfit_kappa_2 is None
                or self.pmg_crossfit_kappa_3 is None
            ):
                raise ValueError(
                    "pmg_crossfit_kappa_2 and pmg_crossfit_kappa_3 are required"
                )
        if self.shape_mode not in {"tar", "native"}:
            raise ValueError("shape_mode must be 'tar' or 'native'")
        if self.texture_mode not in {"tar", "native"}:
            raise ValueError("texture_mode must be 'tar' or 'native'")
        if not 0.0 < self.tar_lambda <= 1.0:
            raise ValueError("tar_lambda must be in (0, 1]")
        if not math.isfinite(self.tar_clip_tau) or self.tar_clip_tau <= 0.0:
            raise ValueError("tar_clip_tau must be finite and positive")
        if not 0.0 <= self.tar_keep_threshold <= 1.0:
            raise ValueError("tar_keep_threshold must be in [0, 1]")

        if len(self.tar_quantiles) != 2:
            raise ValueError("tar_quantiles must contain exactly two values")
        lower_quantile, upper_quantile = self.tar_quantiles
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError(
                "tar_quantiles must be ordered values in [0, 1]"
            )
