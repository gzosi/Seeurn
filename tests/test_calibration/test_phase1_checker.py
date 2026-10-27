"""
Test di Fase 1 della Calibrazione:
Verifica configurazione YAML, visualizzazione diagnostica, calcolo della mappa di copertura del sensore
e controllo di qualità sui 50 fotogrammi reali campionati per ciascuna telecamera.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from seeurn.data.catalog import CalibrationCatalog
from seeurn.data.preprocessor import ImagePreprocessor
from seeurn.calibration.config import CalibrationConfig
from seeurn.calibration.sampler import CalibrationFrameSampler
from seeurn.calibration.visualizer import CornerVisualizer
from seeurn.calibration.checker import CornerQualityChecker, CornerQualityReport


class TestPhase1Config:
    """Verifica il caricamento e la serializzazione delle configurazioni di calibrazione."""

    def test_load_default_yaml_config(self):
        config_path = Path("/home/gzosi/Documents/Activities/Seeurn/configs/calibration_config.yaml")
        if not config_path.exists():
            pytest.skip("calibration_config.yaml non trovato")

        config = CalibrationConfig.from_yaml(config_path)
        assert config.target.pattern_size == (15, 9)
        assert config.target.square_size_mm == 5.0
        assert config.sampling.frames_per_group["mono_train"] == 50
        assert config.phase1.save_diagnostic_images is True

    def test_yaml_roundtrip(self, tmp_path: Path):
        cfg = CalibrationConfig()
        out_yaml = tmp_path / "test_calib_cfg.yaml"
        cfg.to_yaml(out_yaml)

        reloaded = CalibrationConfig.from_yaml(out_yaml)
        assert reloaded.target.pattern_size == cfg.target.pattern_size
        assert reloaded.sampling.random_seed == cfg.sampling.random_seed


class TestCornerVisualizer:
    """Verifica unitaria dei moduli grafici di rendering."""

    def test_draw_annotated_corners(self):
        vis = CornerVisualizer()
        dummy_img = np.zeros((400, 600), dtype=np.uint8)
        # 135 corner fittizi distribuiti a griglia 15x9
        nx, ny = 15, 9
        corners = np.mgrid[50:350:complex(0, nx), 50:250:complex(0, ny)].T.reshape(-1, 2).astype(np.float32)

        annotated = vis.draw_annotated_corners(
            dummy_img, corners, pattern_size=(nx, ny), frame_idx=10, camera_id=1
        )
        assert annotated.shape == (400, 600, 3)
        # Verifica che non sia tutto nero (sono stati disegnati elementi colorati)
        assert np.any(annotated > 0)

    def test_create_sensor_coverage_map(self):
        vis = CornerVisualizer()
        nx, ny = 15, 9
        corners1 = np.mgrid[100:600:complex(0, nx), 100:400:complex(0, ny)].T.reshape(-1, 2).astype(np.float32)
        corners2 = np.mgrid[400:1000:complex(0, nx), 300:700:complex(0, ny)].T.reshape(-1, 2).astype(np.float32)

        canvas, coverage_pct = vis.create_sensor_coverage_map(
            sensor_size=(1280, 800),
            all_corners_list=[corners1, corners2],
            camera_id=1,
            total_frames_sampled=2,
        )
        assert canvas.ndim == 3
        assert coverage_pct > 0.0


class TestRealDataPhase1QualityCheck:
    """
    Esegue la Fase 1 su 50 frame reali campionati da wake2/dataset3 per Cam 1, Cam 2 e Cam 3.
    Verifica che il tasso di rilevamento sia elevato e genera i file visivi per l'ispezione.
    """

    @pytest.fixture
    def catalog(self) -> CalibrationCatalog:
        calib_root = Path("/home/gzosi/Documents/Activities/Seeurn/data/raw/calibration")
        if not calib_root.exists():
            pytest.skip("data/raw/calibration non disponibile")
        return CalibrationCatalog(calib_root)

    def test_run_phase1_check_on_50_frames_per_camera(self, catalog: CalibrationCatalog):
        config_path = Path("/home/gzosi/Documents/Activities/Seeurn/configs/calibration_config.yaml")
        config = CalibrationConfig.from_yaml(config_path)

        sampler = CalibrationFrameSampler(
            min_frame_spacing=config.sampling.min_frame_spacing,
            seed=config.sampling.random_seed,
        )
        checker = CornerQualityChecker(config=config)

        # Mappatura dei dataset ottimali per ciascuna telecamera:
        # Cam 1 e 2 sono ottimizzate in dataset2 (scacchiera frontale)
        # Cam 3 è ottimizzata in dataset3 nella finestra angolata verso la parete laterale
        camera_specs = [
            (1, "dataset2", range(0, 500), 50),
            (2, "dataset2", range(0, 500), 50),
            (3, "dataset3", range(350, 480), 25),
        ]

        for cam_id, dataset_name, frame_range, count in camera_specs:
            all_files = catalog.get_camera_files("wake2", dataset_name, cam_id)
            assert len(all_files) > max(frame_range), f"Cam {cam_id} ha frame insufficienti"

            sampled_indices = sampler.sample_spaced_indices(frame_range, count=count)
            assert len(sampled_indices) == count

            selected_paths = [all_files[idx] for idx in sampled_indices]
            report = checker.check_camera_frames(
                image_paths=selected_paths,
                camera_id=cam_id,
                dataset_name=dataset_name,
            )

            # Assert di correttezza metrologica
            assert report.total_frames_sampled == count
            assert report.valid_detections_count >= int(count * 0.65), (
                f"Camera {cam_id}: success rate troppo basso ({report.success_rate:.1f}%)"
            )
            assert report.coverage_ratio > 10.0, (
                f"Camera {cam_id}: copertura insufficiente ({report.coverage_ratio:.1f}%)"
            )

            # Assert sui feed visivi salvati
            assert len(report.diagnostic_image_paths) == config.phase1.max_diagnostic_images_per_cam
            for diag_p in report.diagnostic_image_paths:
                assert diag_p.exists(), f"File annotato non trovato: {diag_p}"

            assert report.coverage_map_path is not None
            assert report.coverage_map_path.exists(), f"Mappa di copertura non trovata: {report.coverage_map_path}"

            print(
                f"\n[Fase 1 OK] Cam {cam_id} ({dataset_name}): {report.valid_detections_count}/{count} rilevati "
                f"({report.success_rate:.1f}%) | Copertura: {report.coverage_ratio:.1f}% | "
                f"Mappa: {report.coverage_map_path.name}"
            )

