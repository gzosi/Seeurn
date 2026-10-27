"""
Modulo 02: Calibrazione Ottica Multi-Camera, Rilevamento Target e Campionamento Disgiunto.
"""

from seeurn.calibration.target_detector import (
    CheckerboardConfig,
    DetectionResult,
    RobustCheckerboardDetector,
)
from seeurn.calibration.sampler import (
    CalibrationSplitGroups,
    CalibrationFrameSampler,
)

__all__ = [
    "CheckerboardConfig",
    "DetectionResult",
    "RobustCheckerboardDetector",
    "CalibrationSplitGroups",
    "CalibrationFrameSampler",
]
