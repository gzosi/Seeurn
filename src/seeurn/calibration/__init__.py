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
from seeurn.calibration.config import (
    TargetConfig,
    SamplingConfig,
    Phase1Config,
    CalibrationConfig,
)
from seeurn.calibration.visualizer import CornerVisualizer
from seeurn.calibration.checker import CornerQualityReport, CornerQualityChecker
from seeurn.calibration.monocular import (
    DistortionModelFlags,
    CameraIntrinsics,
    MonocularCalibrator,
)
from seeurn.calibration.stereo import (
    StereoExtrinsics,
    compute_epipolar_distance,
    EpipolarGraph,
    StereoPairCalibrator,
)

__all__ = [
    "CheckerboardConfig",
    "DetectionResult",
    "RobustCheckerboardDetector",
    "CalibrationSplitGroups",
    "CalibrationFrameSampler",
    "TargetConfig",
    "SamplingConfig",
    "Phase1Config",
    "CalibrationConfig",
    "CornerVisualizer",
    "CornerQualityReport",
    "CornerQualityChecker",
    "DistortionModelFlags",
    "CameraIntrinsics",
    "MonocularCalibrator",
    "StereoExtrinsics",
    "compute_epipolar_distance",
    "EpipolarGraph",
    "StereoPairCalibrator",
]
