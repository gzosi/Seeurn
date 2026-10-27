"""
Configurazioni della pipeline di calibrazione (Modulo 02).
Carica e valida i parametri da file YAML (es. configs/calibration_config.yaml)
permettendo il controllo completo senza toccare il codice degli algoritmi.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Union
import yaml


@dataclass(frozen=True)
class TargetConfig:
    """Specifiche geometriche della scacchiera di calibrazione."""

    pattern_size: Tuple[int, int] = (15, 9)
    square_size_mm: float = 5.0

    @property
    def num_corners(self) -> int:
        return self.pattern_size[0] * self.pattern_size[1]


@dataclass(frozen=True)
class SamplingConfig:
    """Parametri per il campionamento disgiunto dei fotogrammi."""

    random_seed: int = 42
    min_frame_spacing: int = 5
    frames_per_group: Dict[str, int] = field(
        default_factory=lambda: {
            "mono_train": 50,
            "mono_test": 50,
            "stereo_train": 50,
            "stereo_test": 50,
            "ba_train": 50,
            "final_test": 50,
        }
    )


@dataclass(frozen=True)
class Phase1Config:
    """Parametri per la Fase 1: rilevamento spigoli, check di qualità e visualizzazione."""

    output_dir: Path = Path("tests/output/phase1_corner_check")
    save_diagnostic_images: bool = True
    max_diagnostic_images_per_cam: int = 3
    clahe_clip_limit: float = 3.0
    clahe_grid_size: Tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class CalibrationConfig:
    """Configurazione globale della pipeline di calibrazione."""

    target: TargetConfig = field(default_factory=TargetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CalibrationConfig":
        """Carica la configurazione da file YAML."""
        yaml_path = Path(path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Target
        raw_target = data.get("target", {})
        pat_size = raw_target.get("pattern_size", [15, 9])
        target_cfg = TargetConfig(
            pattern_size=(int(pat_size[0]), int(pat_size[1])),
            square_size_mm=float(raw_target.get("square_size_mm", 5.0)),
        )

        # Sampling
        raw_samp = data.get("sampling", {})
        samp_cfg = SamplingConfig(
            random_seed=int(raw_samp.get("random_seed", 42)),
            min_frame_spacing=int(raw_samp.get("min_frame_spacing", 5)),
            frames_per_group=raw_samp.get(
                "frames_per_group",
                {
                    "mono_train": 50,
                    "mono_test": 50,
                    "stereo_train": 50,
                    "stereo_test": 50,
                    "ba_train": 50,
                    "final_test": 50,
                },
            ),
        )

        # Phase 1
        raw_p1 = data.get("phase1", {})
        grid_size = raw_p1.get("clahe_grid_size", [8, 8])
        p1_cfg = Phase1Config(
            output_dir=Path(raw_p1.get("output_dir", "tests/output/phase1_corner_check")),
            save_diagnostic_images=bool(raw_p1.get("save_diagnostic_images", True)),
            max_diagnostic_images_per_cam=int(raw_p1.get("max_diagnostic_images_per_cam", 3)),
            clahe_clip_limit=float(raw_p1.get("clahe_clip_limit", 3.0)),
            clahe_grid_size=(int(grid_size[0]), int(grid_size[1])),
        )

        return cls(target=target_cfg, sampling=samp_cfg, phase1=p1_cfg)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Esporta la configurazione in un file YAML leggibile."""
        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "target": {
                "pattern_size": list(self.target.pattern_size),
                "square_size_mm": self.target.square_size_mm,
            },
            "sampling": {
                "random_seed": self.sampling.random_seed,
                "min_frame_spacing": self.sampling.min_frame_spacing,
                "frames_per_group": self.sampling.frames_per_group,
            },
            "phase1": {
                "output_dir": str(self.phase1.output_dir),
                "save_diagnostic_images": self.phase1.save_diagnostic_images,
                "max_diagnostic_images_per_cam": self.phase1.max_diagnostic_images_per_cam,
                "clahe_clip_limit": self.phase1.clahe_clip_limit,
                "clahe_grid_size": list(self.phase1.clahe_grid_size),
            },
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

