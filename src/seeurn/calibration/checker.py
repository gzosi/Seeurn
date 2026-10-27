"""
Validatore e orchestratore della Fase 1 di Calibrazione (Modulo 02).
Esegue il controllo di correttezza dei corner sui frame campionati, calcola la copertura
del sensore e salva i report diagnostici e visivi.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from seeurn.data.catalog import CalibrationCatalog
from seeurn.data.preprocessor import ImagePreprocessor
from seeurn.calibration.config import CalibrationConfig
from seeurn.calibration.target_detector import RobustCheckerboardDetector, CheckerboardConfig
from seeurn.calibration.visualizer import CornerVisualizer


@dataclass(frozen=True)
class CornerQualityReport:
    """Rapporto metrico e visivo della qualità dei corner per una telecamera."""

    camera_id: int
    dataset_name: str
    sensor_size: Tuple[int, int]
    total_frames_sampled: int
    valid_detections_count: int
    success_rate: float
    coverage_ratio: float
    methods_breakdown: Dict[str, int]
    diagnostic_image_paths: List[Path] = field(default_factory=list)
    coverage_map_path: Optional[Path] = None


class CornerQualityChecker:
    """
    Orchestra la Fase 1 di calibrazione:
    1. Riceve i frame campionati per una telecamera.
    2. Rileva i 135 corner su immagini raddrizzate da ImagePreprocessor.
    3. Verifica la consistenza geometrica (135 punti, orientamento canonico).
    4. Genera e salva le immagini diagnostiche annotate e la mappa di copertura del sensore.
    """

    def __init__(
        self,
        config: Optional[CalibrationConfig] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        self.config = config or CalibrationConfig()
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.target_cfg = CheckerboardConfig(
            pattern_size=self.config.target.pattern_size,
            square_size_mm=self.config.target.square_size_mm,
        )
        self.detector = RobustCheckerboardDetector(
            config=self.target_cfg,
            clahe_clip_limit=self.config.phase1.clahe_clip_limit,
            clahe_grid_size=self.config.phase1.clahe_grid_size,
            inverted_cameras=(),  # Le immagini arrivano già raddrizzate da ImagePreprocessor
        )
        self.visualizer = CornerVisualizer()

    def check_camera_frames(
        self,
        image_paths: Sequence[Path],
        camera_id: int,
        dataset_name: str = "dataset3",
    ) -> CornerQualityReport:
        """
        Esegue il controllo di qualità su una sequenza di percorsi immagine per una telecamera.

        Args:
            image_paths: Lista di percorsi dei frame campionati.
            camera_id: ID della telecamera (1, 2 o 3).
            dataset_name: Nome identificativo del dataset.

        Returns:
            Istanza di CornerQualityReport con statistiche e percorsi dei grafici salvati.
        """
        output_dir = Path(self.config.phase1.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        valid_corners: List[np.ndarray] = []
        methods_count: Dict[str, int] = {}
        diagnostic_paths: List[Path] = []
        sensor_size: Optional[Tuple[int, int]] = None

        total_sampled = len(image_paths)
        diag_limit = self.config.phase1.max_diagnostic_images_per_cam

        for idx, path in enumerate(image_paths):
            # Carica immagine con orientamento già normalizzato
            img = self.preprocessor.load_image(path, camera_id=camera_id, grayscale=True)
            if sensor_size is None:
                sensor_size = (img.shape[1], img.shape[0])  # (width, height)

            res = self.detector.detect(img, camera_id=camera_id)
            if res.is_valid and res.image_points is not None:
                valid_corners.append(res.image_points)
                methods_count[res.method_used] = methods_count.get(res.method_used, 0) + 1

                # Salva le prime 'diag_limit' immagini annotate
                if len(diagnostic_paths) < diag_limit and self.config.phase1.save_diagnostic_images:
                    annotated = self.visualizer.draw_annotated_corners(
                        image=img,
                        corners=res.image_points,
                        pattern_size=self.config.target.pattern_size,
                        frame_idx=idx,
                        camera_id=camera_id,
                        method=res.method_used,
                    )
                    diag_path = output_dir / f"cam{camera_id}_{dataset_name}_sample_{idx:03d}.png"
                    cv2.imwrite(str(diag_path), annotated)
                    diagnostic_paths.append(diag_path)

        sensor_w, sensor_h = sensor_size or (1280, 800)
        valid_count = len(valid_corners)
        success_rate = (valid_count / max(total_sampled, 1)) * 100.0

        # Genera la mappa di copertura del sensore
        coverage_map, coverage_ratio = self.visualizer.create_sensor_coverage_map(
            sensor_size=(sensor_w, sensor_h),
            all_corners_list=valid_corners,
            camera_id=camera_id,
            total_frames_sampled=total_sampled,
            camera_name=f"Cam {camera_id}",
        )

        coverage_path = output_dir / f"cam{camera_id}_{dataset_name}_coverage_map.png"
        cv2.imwrite(str(coverage_path), coverage_map)

        return CornerQualityReport(
            camera_id=camera_id,
            dataset_name=dataset_name,
            sensor_size=(sensor_w, sensor_h),
            total_frames_sampled=total_sampled,
            valid_detections_count=valid_count,
            success_rate=success_rate,
            coverage_ratio=coverage_ratio,
            methods_breakdown=methods_count,
            diagnostic_image_paths=diagnostic_paths,
            coverage_map_path=coverage_path,
        )

